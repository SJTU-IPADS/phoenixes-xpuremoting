#include <cuda_runtime.h>
#include <cudnn.h>
#include <chrono>
#include <iostream>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (unlikely(err != cudaSuccess)) { \
    std::cout \
        << __FILE__ << ":" << __LINE__ << ": " << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (unlikely(err != CUDNN_STATUS_SUCCESS)) { \
    std::cout \
        << __FILE__ << ":" << __LINE__ << ": " << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

int main() {
    // Number of iterations
    const int numIterations = 10000000;
    cudnnTensorDescriptor_t desc;

    // remove initial overhead
    for (int i = 0; i < numIterations; ++i) {
        // Synchronize the default stream
        CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    }
    
    double totalElapsedTime = 0.0;
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        // Synchronize the default stream
        CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
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
