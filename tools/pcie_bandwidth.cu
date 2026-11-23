// Host <-> GPU copy bandwidth per GPU (1 GiB x 20, pinned host memory). Shows each card's real PCIe link.
// Build: nvcc -O2 pcie_bandwidth.cu -o pcie_bandwidth -lnvidia-ml
// Note: prints GPU UUIDs. Remove them before posting the output publicly.

#include <cuda_runtime.h>
#include <nvml.h>
#include <iostream>
#include <chrono>
#include <cstdio>

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        std::cerr << cudaGetErrorString(e) << std::endl; \
        return 1; \
    } \
} while(0)

#define NVML_CHECK(x) do { \
    nvmlReturn_t r = (x); \
    if (r != NVML_SUCCESS) { \
        std::cerr << nvmlErrorString(r) << std::endl; \
        nvmlShutdown(); \
        return 1; \
    } \
} while(0)

int main() {
    const size_t SIZE = 1024ULL * 1024 * 1024;
    const int ITERATIONS = 20;

    int count;
    CHECK(cudaGetDeviceCount(&count));

    NVML_CHECK(nvmlInit());

    std::cout << "Detected GPUs: " << count << std::endl;

    for (int gpu = 0; gpu < count; gpu++) {
        CHECK(cudaSetDevice(gpu));

        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, gpu));

        nvmlDevice_t nvmlDevice;
        NVML_CHECK(nvmlDeviceGetHandleByIndex(gpu, &nvmlDevice));

        char nvmlUUID[NVML_DEVICE_UUID_BUFFER_SIZE];
        NVML_CHECK(nvmlDeviceGetUUID(
            nvmlDevice, nvmlUUID, sizeof(nvmlUUID)
        ));

        nvmlPciInfo_t pciInfo;
        NVML_CHECK(nvmlDeviceGetPciInfo(
            nvmlDevice, &pciInfo
        ));

        std::cout << "\n====================================" << std::endl;
        std::cout << "CUDA GPU ID: " << gpu << std::endl;
        std::cout << "Name:        " << prop.name << std::endl;
        std::cout << "UUID:        " << nvmlUUID << std::endl;
        std::cout << "PCI Bus ID:  " << pciInfo.busId << std::endl;
        std::cout << "====================================" << std::endl;

        char *host, *device;

        CHECK(cudaMallocHost(&host, SIZE));
        CHECK(cudaMalloc(&device, SIZE));

        for (size_t i = 0; i < SIZE; i++)
            host[i] = i % 256;

        CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < ITERATIONS; i++)
            CHECK(cudaMemcpy(device, host, SIZE,
                             cudaMemcpyHostToDevice));

        CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::steady_clock::now();

        double seconds =
            std::chrono::duration<double>(end-start).count();

        std::cout << "Host -> GPU: "
                  << (SIZE * ITERATIONS / seconds / 1e9)
                  << " GB/s" << std::endl;

        start = std::chrono::steady_clock::now();

        for (int i = 0; i < ITERATIONS; i++)
            CHECK(cudaMemcpy(host, device, SIZE,
                             cudaMemcpyDeviceToHost));

        CHECK(cudaDeviceSynchronize());

        end = std::chrono::steady_clock::now();

        seconds =
            std::chrono::duration<double>(end-start).count();

        std::cout << "GPU -> Host: "
                  << (SIZE * ITERATIONS / seconds / 1e9)
                  << " GB/s" << std::endl;

        cudaFree(device);
        cudaFreeHost(host);
    }

    nvmlShutdown();

    return 0;
}
