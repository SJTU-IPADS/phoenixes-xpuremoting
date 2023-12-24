#include <iostream>
#include <chrono>

int main() {
    // Number of iterations
    const int numIterations = 1000000;
    
    double totalElapsedTime = 0.0;
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        // Synchronize the default stream
        cudaStreamSynchronize(cudaStreamLegacy);
    }
    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time in milliseconds
    std::chrono::duration<double, std::milli> elapsed = end - start;
    totalElapsedTime += elapsed.count();

    // Calculate the average elapsed time
    double averageElapsedTime = totalElapsedTime / numIterations;

    std::cout << "Average elapsed time: " << averageElapsedTime << " ms" << std::endl;

    return 0;
}
