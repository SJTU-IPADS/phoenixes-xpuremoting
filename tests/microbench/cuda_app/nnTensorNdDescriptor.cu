#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>

int main()
{
    const int iterations = 10000;
    cudnnTensorDescriptor_t desc;
    const int nbDims = 400;
    const int dimA[nbDims] = { 1, 1, 1, 1 }, strideA[nbDims] = { 1, 1, 1, 1 };
    
    for (int i = 0; i < iterations; ++i) {
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, nbDims, dimA, strideA);
        cudnnDestroyTensorDescriptor(desc);
    }
    

    return 0;
}
