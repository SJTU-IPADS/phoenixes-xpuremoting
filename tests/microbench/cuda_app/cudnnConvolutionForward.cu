// https://gist.github.com/odashi/1c20ba90388cf02330e1b95963d78039

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <chrono>

#include <cuda.h>
#include <cudnn.h>

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

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid;
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

int main() {
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // input
  const int in_n = 1;
  const int in_c = 1;
  const int in_h = 5;
  const int in_w = 5;
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

  // filter
  const int filt_k = 1;
  const int filt_c = 1;
  const int filt_h = 2;
  const int filt_w = 2;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  float *filt_data;
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  int padA[] = {pad_h, pad_w};
  int filterStrideA[] = {str_h, str_w};
  int dilationA[] = {dil_h, dil_w};
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(
        conv_desc, 2,
        padA, filterStrideA, dilationA,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  int out[4];
  CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        4, out));
  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;

  out_n = out[0];
  out_c = out[1];
  out_h = out[2];
  out_w = out[3];

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  cudnnConvolutionFwdAlgoPerf_t algo_perf;
  algo_perf.algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
//   int requestedAlgoCount = 1;
//   int returnedAlgoCount = 0;
//   CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
//         cudnn,
//         in_desc, filt_desc, conv_desc, out_desc,
//         requestedAlgoCount, &returnedAlgoCount, &algo));

  std::cout << "Convolution algorithm: " << algo_perf.algo << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo_perf.algo, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  // perform
  float alpha = 1.f;
  float beta = 0.f;
  dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
  dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);

  // remove initial overhead
  for (int i = 0; i < 10; i++) {
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo_perf.algo, ws_data, ws_size,
        &beta, out_desc, out_data));
  }

  const int numIterations = 100000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < numIterations; ++i) {
    CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo_perf.algo, ws_data, ws_size,
      &beta, out_desc, out_data));
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  double totalElapsedTime = elapsed.count();
  // Calculate the average elapsed time
  double averageElapsedTime = totalElapsedTime / numIterations;
  std::cout << "Average elapsed time: " << averageElapsedTime << " ms" << std::endl;
  
  // results
  std::cout << "in_data:" << std::endl;
  print(in_data, in_n, in_c, in_h, in_w);
  
  std::cout << "filt_data:" << std::endl;
  print(filt_data, filt_k, filt_c, filt_h, filt_w);
  
  std::cout << "out_data:" << std::endl;
  print(out_data, out_n, out_c, out_h, out_w);

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));
  return 0;
}
