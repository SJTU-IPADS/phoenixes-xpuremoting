#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

int main()
{
    const int iterations = 0;
    int count = 0;
    int device;

    cudaGetDeviceCount(&count);
    std::cout << "Number of CUDA devices: " << count << std::endl;
    cudaGetDevice(&device);
    std::cout << "Current CUDA device: " << device << std::endl;
    cudaSetDevice(count -1 );
    cudaGetDevice(&device);
    std::cout << "Current CUDA device: " << device << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cudaGetDevice(&device);
        cudaSetDevice(0);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double totalElapsedTime = elapsed.count();

    //  Calculate the average elapsed time
    double averageElapsedTime = totalElapsedTime / iterations;

    std::cout << "Average elapsed time: " << averageElapsedTime << " ms" << std::endl;

    return 0;
}
