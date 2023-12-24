#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    cudnnHandle_t handle;
    cudaStream_t stream;

    // Create descriptors
    cudnnCreate(&handle);

    // Set stream
    cudaStreamCreate(&stream);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++)
        cudnnSetStream(handle, stream);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Time elapsed: " << diff.count() << " second" << std::endl;

    // Destroy handle
    cudnnDestroy(handle);

    return 0;
}