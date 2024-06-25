remoting_type_dict = {
    # [remoting_type, payload_backward]
    "__cudaPushCallConfiguration": ["LOCAL", 0],
    "__cudaPopCallConfiguration": ["LOCAL", 0],
    "cublasCreate_v2": ["SYNC", 12],
    "cublasGetMathMode": ["SYNC", 4],
    "cublasGemmEx": ["SYNC", 4],
    "cublasSetMathMode": ["ASYNC", 0],
    "cublasSetStream_v2": ["ASYNC", 0],
    "cublasSgemm_v2": ["ASYNC", 0],
    "cublasSgemmStridedBatched": ["ASYNC", 0],
    "cublasGemmStridedBatchedEx": ["ASYNC", 0],
    "cudaDeviceGetAttribute": ["SYNC", 4],
    "cudaFree": ["SYNC", 4],
    "cudaGetDevice": ["LOCAL", 0],
    "cudaGetLastError": ["SYNC", 4],
    "cudaLaunchKernel": ["ASYNC", 0],
    "cudaMalloc": ["SYNC", 12],
    "cudaMemcpyAsync": ["ASYNC", 0],
    "cudaMemcpyAsyncDeviceToDevice": ["ASYNC", 0],
    "cudaMemcpyAsyncDeviceToHost": ["ASYNC", 0],
    "cudaMemcpyAsyncHostToDevice": ["ASYNC", 0],
    "cudaMemsetAsync": ["ASYNC", 0],
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": ["LOCAL", 0],
    "cudaPeekAtLastError": ["SYNC", 4],
    "cudaSetDevice": ["LOCAL", 0],
    "cudaStreamIsCapturing": ["SYNC", 8],
    "cudaStreamSynchronize": ["SYNC", 4],
    "cudnnCreate": ["SYNC", 12],
    "cudnnBatchNormalizationBackwardEx": ["SYNC", 4],
    "cudnnBatchNormalizationForwardInference": ["ASYNC", 0],
    "cudnnBatchNormalizationForwardTrainingEx": ["SYNC", 4],
    "cudnnConvolutionBackwardData": ["SYNC", 4],
    "cudnnConvolutionBackwardFilter": ["SYNC", 4],
    "cudnnConvolutionForward": ["ASYNC", 0],
    "cudnnCreateConvolutionDescriptor": ["ASYNC", 0],
    "cudnnCreateFilterDescriptor": ["ASYNC", 0],
    "cudnnCreateTensorDescriptor": ["ASYNC", 0],
    "cudnnDestroyConvolutionDescriptor": ["ASYNC", 0],
    "cudnnDestroyFilterDescriptor": ["ASYNC", 0],
    "cudnnDestroyTensorDescriptor": ["ASYNC", 0],
    "cudnnGetBatchNormalizationBackwardExWorkspaceSize": ["SYNC", 12],
    "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize": ["SYNC", 12],
    "cudnnGetBatchNormalizationTrainingExReserveSpaceSize": ["SYNC", 12],
    "cudnnGetConvolutionBackwardDataAlgorithm_v7": ["SYNC", 304],
    "cudnnGetConvolutionBackwardFilterAlgorithm_v7": ["SYNC", 304],
    "cudnnGetConvolutionForwardAlgorithm_v7": ["SYNC", 400],
    "cudnnInitTransformDest": ["ASYNC", 0],
    "cudnnSetConvolutionGroupCount": ["ASYNC", 0],
    "cudnnSetConvolutionMathType": ["ASYNC", 0],
    "cudnnSetConvolutionNdDescriptor": ["ASYNC", 0],
    "cudnnSetFilterNdDescriptor": ["ASYNC", 0],
    "cudnnSetTensorNdDescriptor": ["ASYNC", 0],
    "cudnnSetTensorNdDescriptorEx": ["ASYNC", 0],
    "cudnnSetTensorTransformDescriptor": ["ASYNC", 0],
    "cudnnSetStream": ["ASYNC", 0],
    "nvmlDeviceGetCount_v2": ["SYNC", 4],
    "nvmlInit_v2": ["SYNC", 4],
    "nvmlInitWithFlags": ["SYNC", 4],
}


def get_remoting_type(api_name: str) -> str:
    # if not find then error
    if api_name not in remoting_type_dict:
        # raise ValueError(f"Execution type '{api_name}' not found in remoting_type_dict.")
        print(f"Execution type '{api_name}' not found in remoting_type_dict.")
        return None
    return remoting_type_dict[api_name]

