// Prints cudaDeviceCanAccessPeer for every ordered pair of GPUs.
// Build: nvcc -O2 p2p_access.cu -o p2p_access
// Stock GeForce driver: "NO" everywhere. Patched driver: "YES".

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int count;
    cudaGetDeviceCount(&count);

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
