#include <cuda_runtime.h>
#include <iostream>

int main() {
    int size = 1024 * sizeof(int);
    int* devicePtr;

    cudaMalloc((void**)&devicePtr, size);

    cudaMemsetAsync(devicePtr, 0, size); // <- the API not implemented

    cudaDeviceSynchronize();

    cudaFree(devicePtr);

    return 0;
}