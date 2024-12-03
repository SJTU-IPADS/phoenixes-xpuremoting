#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

int main()
{
    const int iterations = 1000000;
    int device;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cudaGetDevice(&device);
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
