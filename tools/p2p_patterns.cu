// GPU0 <-> GPU1 peer access patterns, one at a time, for a set time. Finds the traffic a pair cannot carry.
// Build: nvcc -O2 -arch=native p2p_patterns.cu -o p2p_patterns
//
// usage: p2p_patterns <mode> <seconds> [bytes]          (bytes default 1 MiB)
//   r01    GPU0's kernels read GPU1's memory through the peer pointer (16-byte loads)
//   r10    GPU1's kernels read GPU0's memory
//   rbi    both at the same time: each GPU's kernels read the other GPU's memory
//   rw     GPU0's kernels read GPU1's memory while GPU1's kernels write into GPU0's memory
//   cerbi  copy engines, both at the same time, each GPU's copy engine reads the other's memory
//          (cudaMemcpyPeerAsync on the destination GPU's stream)
//   cewbi  copy engines, both at the same time, each GPU's copy engine writes into the other's memory
//          (cudaMemcpyPeerAsync on the source GPU's stream; the integrity test's "both ways at once")
// BLOCKS and THREADS in the environment set the kernel size, default 36 x 512 (the grid of vLLM's custom all-reduce).
// One line per second. Two different Turing chips need the chipid shim for peer access (docs/turing.md, T5).
//
// WARNING: on the test rig (Ryzen 9 3900X, B550, RTX 2060 in the chipset slot + GTX 1650 SUPER), `rbi` with 36 x 512
// reset the whole machine within a second, twice ("data fabric sync flood"). Save your work and run `sync` first.
// Every other mode, and `rbi` up to 24 x 512, ran clean there. docs/troubleshooting.md has the details.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x)                                                                                \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));      \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

__global__ void read_peer(const int4* src, size_t n, int4* sink) {
  int4 acc = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = src[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;  // keeps the loads from being optimized away
}

__global__ void write_peer(int4* dst, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = make_int4((int)i, 1, 2, 3);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <r01|r10|rbi|rw|cerbi|cewbi> <seconds> [bytes]\n", argv[0]);
    return 2;
  }
  const char* mode = argv[1];
  const char* modes[] = {"r01", "r10", "rbi", "rw", "cerbi", "cewbi"};
  bool known = false;
  for (const char* m : modes) known = known || !strcmp(mode, m);
  if (!known) {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }
  double seconds = atof(argv[2]);
  size_t bytes = argc > 3 ? strtoull(argv[3], nullptr, 0) : (1u << 20);
  size_t n4 = bytes / 16;
  int blocks = getenv("BLOCKS") ? atoi(getenv("BLOCKS")) : 36;
  int threads = getenv("THREADS") ? atoi(getenv("THREADS")) : 512;

  int can01 = 0, can10 = 0;
  CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
  CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
  if (!can01 || !can10) {
    fprintf(stderr, "no peer access between GPU0 and GPU1 (two different Turing chips need the shim)\n");
    return 1;
  }

  void *src[2], *dst[2];
  int4* sink[2];
  cudaStream_t stream[2];
  for (int d = 0; d < 2; d++) {
    CHECK(cudaSetDevice(d));
    CHECK(cudaDeviceEnablePeerAccess(1 - d, 0));
    CHECK(cudaMalloc(&src[d], bytes));
    CHECK(cudaMalloc(&dst[d], bytes));
    CHECK(cudaMemset(src[d], d + 1, bytes));
    CHECK(cudaMalloc(&sink[d], (size_t)blocks * threads * sizeof(int4)));
    CHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
  }
  printf("mode %s, %zu bytes, %.0f s, kernels %d x %d\n", mode, bytes, seconds, blocks, threads);
  fflush(stdout);

  auto t0 = std::chrono::steady_clock::now(), last = t0;
  long iters = 0, iters_last = 0;
  while (true) {
    for (int d = 0; d < 2; d++) {
      CHECK(cudaSetDevice(d));
      cudaStream_t s = stream[d];
      if (!strcmp(mode, "cerbi")) {
        CHECK(cudaMemcpyPeerAsync(dst[d], d, src[1 - d], 1 - d, bytes, s));  // d's copy engine pulls
      } else if (!strcmp(mode, "cewbi")) {
        CHECK(cudaMemcpyPeerAsync(dst[1 - d], 1 - d, src[d], d, bytes, s));  // d's copy engine pushes
      } else if (!strcmp(mode, "rw") && d == 1) {
        write_peer<<<blocks, threads, 0, s>>>((int4*)dst[0], n4);
      } else if (!strcmp(mode, "rbi") || (d == 0 && (!strcmp(mode, "r01") || !strcmp(mode, "rw"))) ||
                 (d == 1 && !strcmp(mode, "r10"))) {
        read_peer<<<blocks, threads, 0, s>>>((const int4*)src[1 - d], n4, sink[d]);
      }
      CHECK(cudaGetLastError());
    }
    iters++;
    if (iters % 16 == 0) {
      for (int d = 0; d < 2; d++) CHECK(cudaStreamSynchronize(stream[d]));
      auto now = std::chrono::steady_clock::now();
      double dt = std::chrono::duration<double>(now - last).count();
      if (dt >= 1.0) {
        double total = std::chrono::duration<double>(now - t0).count();
        printf("t=%5.1f s  %6.2f GB/s per transfer\n", total, (iters - iters_last) * (double)bytes / dt / 1e9);
        fflush(stdout);
        iters_last = iters;
        last = now;
        if (total >= seconds) break;
      }
    }
  }
  printf("DONE %s\n", mode);
  return 0;
}
