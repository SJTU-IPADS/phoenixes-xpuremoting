#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>

int main()
{
    const int iterations = 10000;
    cudnnTensorDescriptor_t x_desc;
    for (int i = 0; i < iterations; ++i) {
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnDestroyTensorDescriptor(x_desc);
    }

    return 0;
}
