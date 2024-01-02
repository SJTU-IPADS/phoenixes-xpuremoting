#include <iostream>
#include <chrono>

int main() {
    int n = 16*1024*1024;
    char *h_data = (char*)malloc(n);
    char *d_data;

    cudaStream_t stream;

    // Set stream
    cudaStreamCreate(&stream);

    for (int i = 0; i < n; i++) {
        h_data[i] = (char) (i % 128);
    }

    cudaMalloc((void**)&d_data, n * sizeof(int));

    // remove initial overhead
    for (int i = 0; i < 10; i++) {
        cudaMemcpyAsync(d_data, h_data, n, cudaMemcpyHostToDevice, stream);
    }

    // Number of iterations
    const int numIterations = 10000;
    
    double totalElapsedTime = 0.0;
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice);
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
