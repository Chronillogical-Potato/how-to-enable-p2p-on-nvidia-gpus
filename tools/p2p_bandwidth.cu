// GPU0 <-> GPU1 peer copy bandwidth (1 GiB x 20 each way). Exits 1 if peer access is not offered.
// Build: nvcc -O2 p2p_bandwidth.cu -o p2p_bandwidth

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::cerr << cudaGetErrorString(e) << "\n"; return 1; } } while(0)

int main() {
    const size_t SIZE = 1024ULL * 1024 * 1024;
    const int ITERS = 20;
    int can01 = 0, can10 = 0;
    CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    std::cout << "CanAccessPeer 0->1: " << can01 << ", 1->0: " << can10 << "\n";
    if (!can01 || !can10) return 1;

    CHECK(cudaSetDevice(0));
    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    void *d0; CHECK(cudaMalloc(&d0, SIZE));

    CHECK(cudaSetDevice(1));
    CHECK(cudaDeviceEnablePeerAccess(0, 0));
    void *d1; CHECK(cudaMalloc(&d1, SIZE));

    CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; ++i) CHECK(cudaMemcpyPeer(d1, 1, d0, 0, SIZE));
    CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "P2P GPU0 -> GPU1: " << (SIZE * ITERS / std::chrono::duration<double>(t1 - t0).count() / 1e9) << " GB/s\n";

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; ++i) CHECK(cudaMemcpyPeer(d0, 0, d1, 1, SIZE));
    CHECK(cudaDeviceSynchronize());
    t1 = std::chrono::steady_clock::now();
    std::cout << "P2P GPU1 -> GPU0: " << (SIZE * ITERS / std::chrono::duration<double>(t1 - t0).count() / 1e9) << " GB/s\n";

    cudaFree(d0); cudaFree(d1);
    return 0;
}
