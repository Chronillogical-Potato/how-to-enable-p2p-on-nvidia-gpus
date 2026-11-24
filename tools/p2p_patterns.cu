// GPU0 <-> GPU1 peer access patterns, one at a time, for a set time, with the data checked. Finds the traffic a pair
// cannot carry: transfers that bring back wrong data, and transfers that bring down the machine.
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
// Data check: every 16-byte word a kernel reads from the peer is compared with the known fill pattern, so a board that
// does not forward peer reads (they come back as 0xffffffff) shows up as wrong data. Destinations of writes and copies
// are cleared first and compared on their own GPU at the end. Exit status 1 and "DATA WRONG" on any mismatch. This
// does not replace the integrity test, which covers sizes, offsets, allocators and patterns.
//
// WARNING: on the test rig (Ryzen 9 3900X, B550, RTX 2060 in the chipset slot + GTX 1650 SUPER), `rbi` with 36 x 512
// reset the whole machine within a second, twice ("data fabric sync flood"). Save your work and run `sync` first.
// Every other mode, and `rbi` up to 24 x 512, ran clean there. That pair is two different chips run through the shim,
// with one card behind the chipset; either may be the reason. docs/troubleshooting.md has details.

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

struct Mismatch {
  unsigned long long count;  // wrong 16-byte words
  int4 first;                // one wrong word, for the report
};

// Source buffers hold one byte value everywhere (GPU0: 0x01, GPU1: 0x02), so every word read must be `expect`.
__global__ void read_peer(const int4* src, size_t n, unsigned expect, int4* sink, Mismatch* bad) {
  unsigned long long nbad = 0;
  int4 last = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = src[i];
    if (((unsigned)v.x ^ expect) | ((unsigned)v.y ^ expect) | ((unsigned)v.z ^ expect) | ((unsigned)v.w ^ expect)) {
      nbad++;
      last = v;
    }
  }
  if (nbad && atomicAdd(&bad->count, nbad) == 0) bad->first = last;
  sink[blockIdx.x * blockDim.x + threadIdx.x] = last;  // keeps the loads from being optimized away
}

__global__ void write_peer(int4* dst, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = make_int4((int)i, 1, 2, 3);
}

// Runs on the GPU that owns `buf`: local reads only. written: the pattern of write_peer, else one byte value.
__global__ void check_local(const int4* buf, size_t n, bool written, unsigned expect, Mismatch* bad) {
  unsigned long long nbad = 0;
  int4 last = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = buf[i];
    bool ok = written ? (v.x == (int)i && v.y == 1 && v.z == 2 && v.w == 3)
                      : ((unsigned)v.x == expect && (unsigned)v.y == expect && (unsigned)v.z == expect &&
                         (unsigned)v.w == expect);
    if (!ok) {
      nbad++;
      last = v;
    }
  }
  if (nbad && atomicAdd(&bad->count, nbad) == 0) bad->first = last;
}

static unsigned fill_word(int d) { return 0x01010101u * (unsigned)(d + 1); }  // cudaMemset(src[d], d + 1)

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
  Mismatch* bad[2];
  cudaStream_t stream[2];
  for (int d = 0; d < 2; d++) {
    CHECK(cudaSetDevice(d));
    CHECK(cudaDeviceEnablePeerAccess(1 - d, 0));
    CHECK(cudaMalloc(&src[d], bytes));
    CHECK(cudaMalloc(&dst[d], bytes));
    CHECK(cudaMemset(src[d], d + 1, bytes));
    CHECK(cudaMemset(dst[d], 0, bytes));
    CHECK(cudaMalloc(&sink[d], (size_t)blocks * threads * sizeof(int4)));
    CHECK(cudaMalloc(&bad[d], sizeof(Mismatch)));
    CHECK(cudaMemset(bad[d], 0, sizeof(Mismatch)));
    CHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
    CHECK(cudaDeviceSynchronize());
  }
  printf("mode %s, %zu bytes, %.0f s, kernels %d x %d\n", mode, bytes, seconds, blocks, threads);
  fflush(stdout);

  auto mismatches = [&](Mismatch* out) {
    unsigned long long total = 0;
    for (int d = 0; d < 2; d++) {
      CHECK(cudaMemcpy(&out[d], bad[d], sizeof(Mismatch), cudaMemcpyDeviceToHost));
      total += out[d].count;
    }
    return total;
  };

  auto t0 = std::chrono::steady_clock::now(), last = t0;
  long iters = 0, iters_last = 0;
  Mismatch m[2];
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
        read_peer<<<blocks, threads, 0, s>>>((const int4*)src[1 - d], n4, fill_word(1 - d), sink[d], bad[d]);
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
        printf("t=%5.1f s  %6.2f GB/s per transfer, %llu wrong words so far\n", total,
               (iters - iters_last) * (double)bytes / dt / 1e9, mismatches(m));
        fflush(stdout);
        iters_last = iters;
        last = now;
        if (total >= seconds) break;
      }
    }
  }

  // Destinations of copies and writes, checked once at the end on the GPU that owns them.
  for (int d = 0; d < 2; d++) {
    bool copied = !strcmp(mode, "cerbi") || !strcmp(mode, "cewbi");
    bool written = !strcmp(mode, "rw") && d == 0;
    if (copied || written) {
      CHECK(cudaSetDevice(d));
      check_local<<<64, 256, 0, stream[d]>>>((const int4*)dst[d], n4, written, fill_word(1 - d), bad[d]);
      CHECK(cudaStreamSynchronize(stream[d]));
    }
  }
  unsigned long long wrong = mismatches(m);
  if (wrong) {
    for (int d = 0; d < 2; d++)
      if (m[d].count)
        printf("DATA WRONG on GPU%d: %llu wrong 16-byte words, e.g. %08x %08x %08x %08x\n", d, m[d].count,
               (unsigned)m[d].first.x, (unsigned)m[d].first.y, (unsigned)m[d].first.z, (unsigned)m[d].first.w);
    printf("FAIL %s\n", mode);
    return 1;
  }
  printf("DATA OK, DONE %s\n", mode);
  return 0;
}
