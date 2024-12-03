#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <chrono>

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
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 224, 224);

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 224, 224);

    cudnnTensorDescriptor_t bn_param_desc;
    cudnnCreateTensorDescriptor(&bn_param_desc);
    cudnnSetTensor4dDescriptor(bn_param_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 1, 1);

    // Initialize input_data, bn_scale, bn_bias, bn_mean, bn_var
    float *input_data = new float[1 * 3 * 224 * 224];
    float *output_data = new float[1 * 3 * 224 * 224];
    float *bn_scale = new float[3];
    float *bn_bias = new float[3];
    float *bn_mean = new float[3];
    float *bn_var = new float[3];


    cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
    double epsilon = 1e-5;
    float alpha = 1.0f;
    float beta = 0.0f;

    // remove initial overhead
    for (int i = 0; i < 10; i++) {
        CUDNN_CALL(
            cudnnBatchNormalizationForwardInference(
                cudnn,
                mode,
                &alpha,
                &beta,
                input_desc,
                input_data,
                output_desc,
                output_data,
                bn_param_desc,
                bn_scale,
                bn_bias,
                bn_mean,
                bn_var,
                epsilon
            )
        );
    }

    const int iterations = 1000000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        CUDNN_CALL(
            cudnnBatchNormalizationForwardInference(
                cudnn,
                mode,
                &alpha,
                &beta,
                input_desc,
                input_data,
                output_desc,
                output_data,
                bn_param_desc,
                bn_scale,
                bn_bias,
                bn_mean,
                bn_var,
                epsilon
            )
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double totalElapsedTime = elapsed.count();

    //  Calculate the average elapsed time
    double averageElapsedTime = totalElapsedTime / iterations;

    std::cout << "Average elapsed time: " << averageElapsedTime << " ms" << std::endl;

    // Do something with output_data

    delete[] input_data;
    delete[] output_data;
    delete[] bn_scale;
    delete[] bn_bias;
    delete[] bn_mean;
    delete[] bn_var;

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bn_param_desc);

    cudnnDestroy(cudnn);

    return 0;
}
