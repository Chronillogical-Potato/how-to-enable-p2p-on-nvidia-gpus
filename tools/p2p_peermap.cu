// Peer access between every pair of GPUs at once, then one allocation per GPU, in turn: how much memory each card can
// still allocate when every other card maps it. Made for three or more cards (docs/three-cards.md).
// Build: nvcc -O2 -arch=native p2p_peermap.cu -o p2p_peermap -lnvidia-ml
//
// usage: p2p_peermap <MiB> [GPU...]          (default: every GPU)
//
// With peer access on, CUDA maps every new allocation into every peer, so each cudaMalloc either succeeds or fails.
// Up to fork commit 25c929e the patched driver failed it with out of memory once a card would map more than one
// card's BAR1 of all the other cards' memory together, while the card itself had room (docs/troubleshooting.md, "Three
// or more cards"). Prints each allocation's result and each card's BAR1 use. Then every GPU's kernel writes 1 MiB into
// the END of every other GPU's allocation, through the peer pointer, and the owner compares it word by word: the whole
// allocation is reachable, not just its start.
// Exit status: 0 every allocation made and every block exact ("RESULT OK"); 1 an allocation failed; 2 "DATA WRONG";
// 3 usage or setup (fewer than two GPUs, or a pair without peer access). Two different Turing chips need the chipid
// shim for peer access (docs/turing.md, T5).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <nvml.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(3); } } while (0)

static const size_t BLOCK = 1 << 20;   // bytes each writer puts into each peer's allocation

__global__ void fill(unsigned *dst, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = seed ^ (unsigned)(i * 2654435761u);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MIB [GPU...]\n", argv[0]); return 3; }
    size_t mib = strtoull(argv[1], nullptr, 10), bytes = mib << 20;
    int count = 0; CK(cudaGetDeviceCount(&count));
    std::vector<int> g;
    for (int a = 2; a < argc; a++) g.push_back(atoi(argv[a]));
    if (g.empty()) for (int d = 0; d < count; d++) g.push_back(d);
    int n = (int)g.size();
    if (n < 2 || bytes < (size_t)(n + 1) * BLOCK) { fprintf(stderr, "need two GPUs and MIB >= %d\n", n + 1); return 3; }

    nvmlInit();
    // peer access, every ordered pair
    for (int i = 0; i < n; i++) {
        CK(cudaSetDevice(g[i]));
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            int can = 0; CK(cudaDeviceCanAccessPeer(&can, g[i], g[j]));
            if (!can) { printf("GPU %d -> GPU %d: no peer access\n", g[i], g[j]); return 3; }
            CK(cudaDeviceEnablePeerAccess(g[j], 0));
        }
    }
    printf("peer access on between all %d GPUs; %zu MiB per GPU, in turn\n", n, mib);

    std::vector<void *> p(n, nullptr);
    int failed = 0;
    for (int i = 0; i < n; i++) {
        CK(cudaSetDevice(g[i]));
        size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
        cudaError_t e = cudaMalloc(&p[i], bytes);
        printf("cudaMalloc %zu MiB on GPU %d (%zu MiB free): %s\n", mib, g[i], fr >> 20,
               e == cudaSuccess ? "ok" : cudaGetErrorString(e));
        if (e != cudaSuccess) { cudaGetLastError(); p[i] = nullptr; failed = 1; break; }
    }
    for (int i = 0; i < n; i++) {
        nvmlDevice_t h; char bus[32]; nvmlBAR1Memory_t b{};
        cudaDeviceGetPCIBusId(bus, sizeof bus, g[i]);
        if (nvmlDeviceGetHandleByPciBusId_v2(bus, &h) == NVML_SUCCESS && nvmlDeviceGetBAR1MemoryInfo(h, &b) == NVML_SUCCESS)
            printf("GPU %d %s BAR1 used %llu of %llu MiB\n", g[i], bus, b.bar1Used >> 20, b.bar1Total >> 20);
    }
    if (failed) return 1;

    // every writer w fills block w+1 from the end of every other GPU's allocation, through the peer mapping
    for (int w = 0; w < n; w++) {
        CK(cudaSetDevice(g[w]));
        for (int o = 0; o < n; o++) {
            if (o == w) continue;
            unsigned *dst = (unsigned *)((char *)p[o] + bytes - (size_t)(w + 1) * BLOCK);
            fill<<<64, 256>>>(dst, BLOCK / 4, 0x9e3779b9u * (unsigned)(w * 16 + o + 1));
            CK(cudaGetLastError());
        }
        CK(cudaDeviceSynchronize());
    }
    std::vector<unsigned> h(BLOCK / 4);
    int wrong = 0;
    for (int o = 0; o < n; o++) {
        CK(cudaSetDevice(g[o]));
        for (int w = 0; w < n; w++) {
            if (w == o) continue;
            CK(cudaMemcpy(h.data(), (char *)p[o] + bytes - (size_t)(w + 1) * BLOCK, BLOCK, cudaMemcpyDeviceToHost));
            unsigned seed = 0x9e3779b9u * (unsigned)(w * 16 + o + 1);
            size_t bad = 0;
            for (size_t i = 0; i < h.size(); i++) bad += h[i] != (seed ^ (unsigned)(i * 2654435761u));
            printf("GPU %d wrote the end of GPU %d's allocation: %s\n", g[w], g[o], bad ? "DATA WRONG" : "DATA OK");
            if (bad) { printf("  %zu of %zu words wrong\n", bad, h.size()); wrong = 1; }
        }
    }
    for (int i = 0; i < n; i++) { cudaSetDevice(g[i]); cudaFree(p[i]); }
    nvmlShutdown();
    printf("%s\n", wrong ? "RESULT DATA WRONG" : "RESULT OK");
    return wrong ? 2 : 0;
}
