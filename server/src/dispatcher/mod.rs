use super::*;

mod cuda_exe;
mod cuda_exe_custom;
mod cudart_exe;
mod cudart_exe_custom;
mod nvml_exe;
mod cudnn_exe_custom;
mod cudnn_exe;
mod cublas_exe;
mod cublas_exe_custom;


use self::cuda_exe::*;
use self::cuda_exe_custom::*;
use self::cudart_exe::*;
use self::cudart_exe_custom::*;
use self::nvml_exe::*;
use self::cudnn_exe_custom::*;
use self::cudnn_exe::*;
use self::cublas_exe::*;
use self::cublas_exe_custom::*;

pub fn dispatch<T: CommChannel>(proc_id: i32, channel_sender: &mut T, channel_receiver: &mut T) {
    // let start = network::NsTimestamp::now();
    match proc_id {
        0 => cudaGetDeviceExe(channel_sender, channel_receiver),
        1 => cudaSetDeviceExe(channel_sender, channel_receiver),
        2 => cudaGetDeviceCountExe(channel_sender, channel_receiver),
        3 => cudaGetLastErrorExe(channel_sender, channel_receiver),
        4 => cudaPeekAtLastErrorExe(channel_sender, channel_receiver),
        5 => cudaStreamSynchronizeExe(channel_sender, channel_receiver),
        6 => cudaMallocExe(channel_sender, channel_receiver),
        7 => cudaMemcpyExe(channel_sender, channel_receiver),
        8 => cudaFreeExe(channel_sender, channel_receiver),
        9 => cudaStreamIsCapturingExe(channel_sender, channel_receiver),
        10 => cudaGetDevicePropertiesExe(channel_sender, channel_receiver),
        11 => cudaMallocManagedExe(channel_sender, channel_receiver),
        12 => cudaPointerGetAttributesExe(channel_sender, channel_receiver),
        13 => cudaHostAllocExe(channel_sender, channel_receiver),
        14 => cudaFuncGetAttributesExe(channel_sender, channel_receiver),
        15 => cudaDeviceGetStreamPriorityRangeExe(channel_sender, channel_receiver),
        16 => cudaMemsetAsyncExe(channel_sender, channel_receiver),
        17 => cudaGetErrorStringExe(channel_sender, channel_receiver),
        18 => cudaMemGetInfoExe(channel_sender, channel_receiver),
        100 => __cudaRegisterFatBinaryExe(channel_sender, channel_receiver),
        101 => __cudaUnregisterFatBinaryExe(channel_sender, channel_receiver),
        102 => __cudaRegisterFunctionExe(channel_sender, channel_receiver),
        103 => __cudaRegisterVarExe(channel_sender, channel_receiver),
        200 => cudaLaunchKernelExe(channel_sender, channel_receiver),
        300 => cuDevicePrimaryCtxGetStateExe(channel_sender, channel_receiver),
        500 => cuGetProcAddressExe(channel_sender, channel_receiver),
        501 => cuDriverGetVersionExe(channel_sender, channel_receiver),
        502 => cuInitExe(channel_sender, channel_receiver),
        503 => cuGetExportTableExe(channel_sender, channel_receiver),
        1000 => nvmlInit_v2Exe(channel_sender, channel_receiver),
        1001 => nvmlDeviceGetCount_v2Exe(channel_sender, channel_receiver),
        1002 => nvmlInitWithFlagsExe(channel_sender, channel_receiver),
        1500 => cudnnCreateExe(channel_sender, channel_receiver),
        1501 => cudnnCreateTensorDescriptorExe(channel_sender, channel_receiver),
        1502 => cudnnSetTensor4dDescriptorExe(channel_sender, channel_receiver),
        1503 => cudnnCreateActivationDescriptorExe(channel_sender, channel_receiver),
        1504 => cudnnSetActivationDescriptorExe(channel_sender, channel_receiver),
        1505 => cudnnActivationForwardExe(channel_sender, channel_receiver),
        1506 => cudnnDestroyExe(channel_sender, channel_receiver),
        1507 => cudnnSetConvolution2dDescriptorExe(channel_sender, channel_receiver),
        1508 => cudnnSetStreamExe(channel_sender, channel_receiver),
        1509 => cudnnSetTensorNdDescriptorExe(channel_sender, channel_receiver),
        1510 => cudnnDestroyTensorDescriptorExe(channel_sender, channel_receiver),
        1511 => cudnnCreateFilterDescriptorExe(channel_sender, channel_receiver),
        1512 => cudnnDestroyFilterDescriptorExe(channel_sender, channel_receiver),
        1513 => cudnnSetFilterNdDescriptorExe(channel_sender, channel_receiver),
        1514 => cudnnCreateConvolutionDescriptorExe(channel_sender, channel_receiver),
        1515 => cudnnDestroyConvolutionDescriptorExe(channel_sender, channel_receiver),
        1516 => cudnnSetConvolutionNdDescriptorExe(channel_sender, channel_receiver),
        1517 => cudnnSetConvolutionGroupCountExe(channel_sender, channel_receiver),
        1518 => cudnnSetConvolutionMathTypeExe(channel_sender, channel_receiver),
        1519 => cudnnSetConvolutionReorderTypeExe(channel_sender, channel_receiver),
        1520 => cudnnGetConvolutionForwardAlgorithm_v7Exe(channel_sender, channel_receiver),
        1521 => cudnnConvolutionForwardExe(channel_sender, channel_receiver),
        1522 => cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeExe(channel_sender, channel_receiver),
        1523 => cudnnGetBatchNormalizationTrainingExReserveSpaceSizeExe(channel_sender, channel_receiver),
        1524 => cudnnBatchNormalizationForwardTrainingExExe(channel_sender, channel_receiver),
        1525 => cudnnGetBatchNormalizationBackwardExWorkspaceSizeExe(channel_sender, channel_receiver),
        1526 => cudnnBatchNormalizationBackwardExExe(channel_sender, channel_receiver),
        1527 => cudnnGetConvolutionBackwardDataAlgorithm_v7Exe(channel_sender, channel_receiver),
        1528 => cudnnConvolutionBackwardDataExe(channel_sender, channel_receiver),
        1529 => cudnnGetConvolutionBackwardFilterAlgorithm_v7Exe(channel_sender, channel_receiver),
        1530 => cudnnConvolutionBackwardFilterExe(channel_sender, channel_receiver),
        1531 => cudnnBatchNormalizationForwardInferenceExe(channel_sender, channel_receiver),
        1532 => cudnnSetFilter4dDescriptorExe(channel_sender, channel_receiver),
        1533 => cudnnGetConvolutionNdForwardOutputDimExe(channel_sender, channel_receiver),
        1534 => cudnnGetConvolutionForwardWorkspaceSizeExe(channel_sender, channel_receiver),
        1535 => cudnnGetErrorStringExe(channel_sender, channel_receiver),
        2000 => cublasCreate_v2Exe(channel_sender, channel_receiver),
        2001 => cublasDestroy_v2Exe(channel_sender, channel_receiver),
        2002 => cublasSetStream_v2Exe(channel_sender, channel_receiver),
        2003 => cublasSetMathModeExe(channel_sender, channel_receiver),
        2004 => cublasSgemm_v2Exe(channel_sender, channel_receiver),
        2005 => cublasSgemmStridedBatchedExe(channel_sender, channel_receiver), 
        2006 => cublasGetMathModeExe(channel_sender, channel_receiver), 
        2007 => cublasGemmExExe(channel_sender, channel_receiver), 
        2008 => cublasGemmStridedBatchedExExe(channel_sender, channel_receiver),
        other => {
            error!(
                "[{}:{}] invalid proc_id: {}",
                std::file!(),
                std::line!(),
                other
            );
        }
    }
    // let end = network::NsTimestamp::now();
    // let elapsed = (end.sec_timestamp - start.sec_timestamp) as f64 * 1000000000.0
    //             + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as f64;
    // info!("exe: {}", elapsed / 1000.0);
}
