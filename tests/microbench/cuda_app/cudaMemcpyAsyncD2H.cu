#include <iostream>
#include <chrono>
#include <cstdlib>

int getMemorySize() {
    char* envVar = std::getenv("MEMORY_SIZE");
    if (envVar != nullptr) {
        int memorySize = std::stoi(envVar);
        return memorySize;
    }
    return 1024;
}

int main() {
    int n = getMemorySize();
    char *h_data = (char*)malloc(n);
    char *d_data;

    std::cout << "test with memory size: " << n << std::endl;

    cudaStream_t stream;

    // Set stream
    cudaStreamCreate(&stream);

    for (int i = 0; i < n; i++) {
        h_data[i] = (char) (i % 128);
    }

    cudaMalloc((void**)&d_data, n * sizeof(int));

    // remove initial overhead
    for (int i = 0; i < 10; i++) {
        cudaMemcpyAsync(h_data, d_data, n, cudaMemcpyDeviceToHost, stream);
    }

    // Number of iterations
    const int numIterations = 10000;
    
    double totalElapsedTime = 0.0;
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        cudaMemcpyAsync(h_data, d_data, n, cudaMemcpyDeviceToHost, stream);
        // cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice);
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
