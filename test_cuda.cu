#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("Found %d CUDA device(s):\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    }
    
    return 0;
}
