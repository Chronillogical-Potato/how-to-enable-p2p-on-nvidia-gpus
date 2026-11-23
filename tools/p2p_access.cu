// Prints cudaDeviceCanAccessPeer for every ordered pair of GPUs.
// Build: nvcc -O2 p2p_access.cu -o p2p_access
// Stock GeForce driver: "NO" everywhere. Patched driver: "YES".

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        // For example one GPU that failed to initialize (dmesg: "RmInitAdapter failed") makes CUDA fail for all.
        fprintf(stderr, "cudaGetDeviceCount: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < count; j++) {
            if (i == j) continue;

            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);

            printf("GPU %d -> GPU %d: %s\n",
                   i, j, canAccess ? "YES" : "NO");
        }
    }
}
