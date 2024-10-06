#![allow(non_snake_case)]
#![allow(unused_variables)]

use super::*;
use cudasys::cudnn::*;
use std::os::raw::*;

pub fn cudnnCreateExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudnnCreate", std::file!(), std::line!());
    let mut cuda_handle: cudnnHandle_t = Default::default();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = {
        // cudnnCreate(&mut cuda_handle)
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1500u64,
                0u64,
                vec![
                    get_address(&mut cuda_handle), cuda_handle.mem_size(),
                ]
            )
        )
    };

    cuda_handle.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

#[cfg(feature = "shadow_desc")]
pub fn cudnnCreateTensorDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnCreateTensorDescriptor",
        std::file!(),
        std::line!()
    );

    let mut resource_idx: usize = Default::default();
    match resource_idx.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving tensor_desc_addr: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let mut tensorDesc: cudnnTensorDescriptor_t = Default::default();
    {
        // cudnnCreateTensorDescriptor(&mut tensorDesc)
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1501u64,
                0u64,
                vec![
                    get_address(&mut tensorDesc), tensorDesc.mem_size(),
                ]
            )
        )
    };
    add_resource(resource_idx, tensorDesc as usize);
}
#[cfg(not(feature = "shadow_desc"))]
pub fn cudnnCreateTensorDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnCreateTensorDescriptor",
        std::file!(),
        std::line!()
    );

    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let mut tensorDesc: cudnnTensorDescriptor_t = Default::default();
    let result: cudnnStatus_t = {
        // cudnnCreateTensorDescriptor(&mut tensorDesc)
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1501u64,
                0u64,
                vec![
                    get_address(&mut tensorDesc), tensorDesc.mem_size(),
                ]
            )
        )
    };

    tensorDesc.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cudnnCreateActivationDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnCreateActivationDescriptor",
        std::file!(),
        std::line!()
    );
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let mut activationDesc: cudnnActivationDescriptor_t = Default::default();
    let result = {
        // cudnnCreateActivationDescriptor(&mut activationDesc)
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1503u64,
                0u64,
                vec![
                    get_address(&mut activationDesc), activationDesc.mem_size(),
                ]
            )
        )
    };

    activationDesc.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cudnnActivationForwardExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudnnActivationForward", std::file!(), std::line!());
    let mut handle: cudnnHandle_t = Default::default();
    let mut activationDesc: cudnnActivationDescriptor_t = Default::default();
    let mut alpha: MemPtr = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut x: MemPtr = Default::default();
    let mut beta: MemPtr = Default::default();
    let mut yDesc: cudnnTensorDescriptor_t = Default::default();
    let mut y: MemPtr = Default::default();
    match handle.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving handle: {:?}", e);
        }
    }
    match activationDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving activationDesc: {:?}", e);
        }
    }
    match alpha.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving alpha: {:?}", e);
        }
    }
    match xDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving xDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    match x.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving x: {:?}", e);
        }
    }
    match beta.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving beta: {:?}", e);
        }
    }
    match yDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving yDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let yDesc = get_resource(yDesc as usize);
    match y.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving y: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result: cudnnStatus_t = {
        // cudnnActivationForward(
        //     handle,
        //     activationDesc,
        //     alpha as *const c_void,
        //     xDesc,
        //     x as *const c_void,
        //     beta as *const c_void,
        //     yDesc,
        //     y as *mut c_void,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1505u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&activationDesc), activationDesc.mem_size(),
                    get_address(&alpha), alpha.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&x), x.mem_size(),
                    get_address(&beta), beta.mem_size(),
                    get_address(&yDesc), yDesc.mem_size(),
                    get_address(&y), y.mem_size(),
                ]
            )
        )
    };
    y.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cudnnSetTensorNdDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnSetTensorNdDescriptor",
        std::file!(),
        std::line!()
    );
    let mut tensorDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dataType: cudnnDataType_t = Default::default();
    let mut nbDims: c_int = Default::default();
    let mut dimA: Vec<i32> = Default::default();
    let mut strideA: Vec<i32> = Default::default();
    match tensorDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving tensorDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let tensorDesc = get_resource(tensorDesc as usize);
    match dataType.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving dataType: {:?}", e);
        }
    }
    match nbDims.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving nbDims: {:?}", e);
        }
    }
    match dimA.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving dimA: {:?}", e);
        }
    }
    match strideA.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving strideA: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let result: cudnnStatus_t = {
        // cudnnSetTensorNdDescriptor(
        //     tensorDesc,
        //     dataType,
        //     nbDims,
        //     dimA.as_ptr() as *const c_int,
        //     strideA.as_ptr() as *const c_int,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1509u64,
                0u64,
                vec![
                    get_address(&tensorDesc), tensorDesc.mem_size(),
                    get_address(&dataType), dataType.mem_size(),
                    get_address(&nbDims), nbDims.mem_size(),
                    get_address(&dimA), dimA.mem_size(),
                    get_address(&strideA), strideA.mem_size(),
                ]
            )
        )
    };

    #[cfg(not(feature = "async_api"))]
    {
        result.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
}

#[cfg(feature = "shadow_desc")]
pub fn cudnnCreateFilterDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnCreateFilterDescriptor",
        std::file!(),
        std::line!()
    );
    let mut resource_idx: usize = Default::default();
    match resource_idx.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving tensor_desc_addr: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let mut filterDesc: cudnnFilterDescriptor_t = Default::default();
    {
        // cudnnCreateFilterDescriptor(&mut filterDesc)
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1511u64,
                0u64,
                vec![
                    get_address(&mut filterDesc), filterDesc.mem_size(),
                ]
            )
        )
    };
    add_resource(resource_idx, filterDesc as usize);
}
#[cfg(not(feature = "shadow_desc"))]
pub fn cudnnCreateFilterDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnCreateFilterDescriptor",
        std::file!(),
        std::line!()
    );
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let mut filterDesc: cudnnFilterDescriptor_t = Default::default();
    let result: cudnnStatus_t = {
        // cudnnCreateFilterDescriptor(&mut filterDesc)
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1511u64,
                0u64,
                vec![
                    get_address(&mut filterDesc), filterDesc.mem_size(),
                ]
            )
        )
    };
    filterDesc.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cudnnSetFilterNdDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnSetFilterNdDescriptor",
        std::file!(),
        std::line!()
    );
    let mut filterDesc: cudnnFilterDescriptor_t = Default::default();
    let mut dataType: cudnnDataType_t = Default::default();
    let mut format: cudnnTensorFormat_t = Default::default();
    let mut nbDims: c_int = Default::default();
    let mut filterDimA: Vec<i32> = Default::default();
    match filterDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving filterDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let filterDesc = get_resource(filterDesc as usize);
    match dataType.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving dataType: {:?}", e);
        }
    }
    match format.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving format: {:?}", e);
        }
    }
    match nbDims.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving nbDims: {:?}", e);
        }
    }
    match filterDimA.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving filterDimA: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = {
        // cudnnSetFilterNdDescriptor(
        //     filterDesc,
        //     dataType,
        //     format,
        //     nbDims,
        //     filterDimA.as_ptr() as *const c_int,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1513u64,
                0u64,
                vec![
                    get_address(&filterDesc), filterDesc.mem_size(),
                    get_address(&dataType), dataType.mem_size(),
                    get_address(&format), format.mem_size(),
                    get_address(&nbDims), nbDims.mem_size(),
                    get_address(&filterDimA), filterDimA.mem_size(),
                ]
            )
        )
    };
    #[cfg(not(feature = "async_api"))]
    {
        match result.send(channel_sender) {
            Ok(()) => {}
            Err(e) => {
                error!("Error sending result: {:?}", e);
            }
        }
        channel_sender.flush_out().unwrap();
    }
}

#[cfg(feature = "shadow_desc")]
pub fn cudnnCreateConvolutionDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnCreateConvolutionDescriptor",
        std::file!(),
        std::line!()
    );
    let mut resource_idx: usize = Default::default();
    match resource_idx.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving tensor_desc_addr: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    {
        // cudnnCreateConvolutionDescriptor(&mut convDesc)
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1514u64,
                0u64,
                vec![
                    get_address(&mut convDesc), convDesc.mem_size(),
                ]
            )
        )
    };
    add_resource(resource_idx, convDesc as usize);
}
#[cfg(not(feature = "shadow_desc"))]
pub fn cudnnCreateConvolutionDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnCreateConvolutionDescriptor",
        std::file!(),
        std::line!()
    );
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    let result: cudnnStatus_t = {
        // cudnnCreateConvolutionDescriptor(&mut convDesc)
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1514u64,
                0u64,
                vec![
                    get_address(&mut convDesc), convDesc.mem_size(),
                ]
            )
        )
    };
    convDesc.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cudnnSetConvolutionNdDescriptorExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnSetConvolutionNdDescriptor",
        std::file!(),
        std::line!()
    );
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    let mut arrayLength: c_int = Default::default();
    let mut padA: Vec<i32> = Default::default();
    let mut filterStrideA: Vec<i32> = Default::default();
    let mut upscaleA: Vec<i32> = Default::default();
    let mut mode: cudnnConvolutionMode_t = Default::default();
    let mut computeType: cudnnDataType_t = Default::default();
    match convDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving convDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let convDesc = get_resource(convDesc as usize);
    match arrayLength.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving arrayLength: {:?}", e);
        }
    }
    match padA.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving padA: {:?}", e);
        }
    }
    match filterStrideA.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving filterStrideA: {:?}", e);
        }
    }
    match upscaleA.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving upscaleA: {:?}", e);
        }
    }
    match mode.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving mode: {:?}", e);
        }
    }
    match computeType.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving computeType: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = {
        // cudnnSetConvolutionNdDescriptor(
        //     convDesc,
        //     arrayLength,
        //     padA.as_ptr() as *const c_int,
        //     filterStrideA.as_ptr() as *const c_int,
        //     upscaleA.as_ptr() as *const c_int,
        //     mode,
        //     computeType,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1516u64,
                0u64,
                vec![
                    get_address(&convDesc), convDesc.mem_size(),
                    get_address(&arrayLength), arrayLength.mem_size(),
                    get_address(&padA), padA.mem_size(),
                    get_address(&filterStrideA), filterStrideA.mem_size(),
                    get_address(&upscaleA), upscaleA.mem_size(),
                    get_address(&mode), mode.mem_size(),
                    get_address(&computeType), computeType.mem_size(),
                ]
            )
        )
    };
    #[cfg(not(feature = "async_api"))]
    {
        match result.send(channel_sender) {
            Ok(()) => {}
            Err(e) => {
                error!("Error sending result: {:?}", e);
            }
        }
        channel_sender.flush_out().unwrap();
    }
}

pub fn cudnnGetConvolutionForwardAlgorithm_v7Exe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnGetConvolutionForwardAlgorithm_v7",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut wDesc: cudnnFilterDescriptor_t = Default::default();
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    let mut yDesc: cudnnTensorDescriptor_t = Default::default();
    let mut requestedAlgoCount: c_int = Default::default();
    let mut returnedAlgoCount: c_int = Default::default();
    let mut perfResults: Vec<cudnnConvolutionFwdAlgoPerf_t> = Default::default();
    match handle.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving handle: {:?}", e);
        }
    }
    match xDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving xDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    match wDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving wDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let wDesc = get_resource(wDesc as usize);
    match convDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving convDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let convDesc = get_resource(convDesc as usize);
    match yDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving yDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let yDesc = get_resource(yDesc as usize);
    match requestedAlgoCount.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving requestedAlgoCount: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    perfResults.resize(requestedAlgoCount as usize, Default::default());
    let result = {
        // cudnnGetConvolutionForwardAlgorithm_v7(
        //     handle,
        //     xDesc,
        //     wDesc,
        //     convDesc,
        //     yDesc,
        //     requestedAlgoCount,
        //     &mut returnedAlgoCount,
        //     perfResults.as_mut_ptr(),
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1520u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&wDesc), wDesc.mem_size(),
                    get_address(&convDesc), convDesc.mem_size(),
                    get_address(&yDesc), yDesc.mem_size(),
                    get_address(&requestedAlgoCount), requestedAlgoCount.mem_size(),
                    get_address(&mut returnedAlgoCount), returnedAlgoCount.mem_size(),
                    get_address(&perfResults), perfResults.mem_size(),
                ]
            )
        )
    };
    match returnedAlgoCount.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("Error sending returnedAlgoCount: {:?}", e);
        }
    }
    match perfResults.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("Error sending perfResults: {:?}", e);
        }
    }
    match result.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("Error sending result: {:?}", e);
        }
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudnnConvolutionForwardExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnConvolutionForward",
        std::file!(),
        std::line!()
    );

    let mut handle: cudnnHandle_t = Default::default();
    let mut alpha: f64 = Default::default(); // currently, we assume that alpha is f64
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut x: MemPtr = Default::default();
    let mut wDesc: cudnnFilterDescriptor_t = Default::default();
    let mut w: MemPtr = Default::default();
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    let mut algo: cudnnConvolutionFwdAlgo_t = Default::default();
    let mut workSpace: MemPtr = Default::default();
    let mut workSpaceSizeInBytes: size_t = Default::default();
    let mut beta: f64 = Default::default(); // currently, we assume that beta is f64
    let mut yDesc: cudnnTensorDescriptor_t = Default::default();
    let mut y: MemPtr = Default::default();
    match handle.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving handle: {:?}", e);
        }
    }
    match alpha.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving alpha: {:?}", e);
        }
    }
    match xDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving xDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    match x.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving x: {:?}", e);
        }
    }
    match wDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving wDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let wDesc = get_resource(wDesc as usize);
    match w.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving w: {:?}", e);
        }
    }
    match convDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving convDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let convDesc = get_resource(convDesc as usize);
    match algo.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving algo: {:?}", e);
        }
    }
    match workSpace.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving workSpace: {:?}", e);
        }
    }
    match workSpaceSizeInBytes.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving workSpaceSizeInBytes: {:?}", e);
        }
    }
    match beta.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving beta: {:?}", e);
        }
    }
    match yDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving yDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let yDesc = get_resource(yDesc as usize);
    match y.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving y: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let alpha_ = &alpha as *const f64;
    let beta_ = &beta as *const f64;

    let result: cudnnStatus_t = {
        // cudnnConvolutionForward(
        //     handle,
        //     alpha_ as *const c_void,
        //     xDesc,
        //     x as *const c_void,
        //     wDesc,
        //     w as *const c_void,
        //     convDesc,
        //     algo,
        //     workSpace as *mut c_void,
        //     workSpaceSizeInBytes,
        //     beta_ as *const c_void,
        //     yDesc,
        //     y as *mut c_void,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1521u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&alpha_), alpha_.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&x), x.mem_size(),
                    get_address(&wDesc), wDesc.mem_size(),
                    get_address(&w), w.mem_size(),
                    get_address(&convDesc), convDesc.mem_size(),
                    get_address(&algo), algo.mem_size(),
                    get_address(&workSpace), workSpace.mem_size(),
                    get_address(&workSpaceSizeInBytes), workSpaceSizeInBytes.mem_size(),
                    get_address(&beta_), beta_.mem_size(),
                    get_address(&yDesc), yDesc.mem_size(),
                    get_address(&y), y.mem_size(),
                ]
            )
        )
    };
    
    #[cfg(not(feature = "async_api"))]
    {
        match result.send(channel_sender) {
            Ok(()) => {}
            Err(e) => {
                error!("Error sending result: {:?}", e);
            }
        }
        channel_sender.flush_out().unwrap();
    }
}

pub fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut mode: cudnnBatchNormMode_t = Default::default();
    let mut bnOps: cudnnBatchNormOps_t = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut zDesc: cudnnTensorDescriptor_t = Default::default();
    let mut yDesc: cudnnTensorDescriptor_t = Default::default();
    let mut bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t = Default::default();
    let mut activationDesc: cudnnActivationDescriptor_t = Default::default();
    let mut sizeInBytes: size_t = Default::default();

    match handle.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving handle: {:?}", e),
    }
    match mode.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving mode: {:?}", e),
    }
    match bnOps.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving bnOps: {:?}", e),
    }
    match xDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving xDesc: {:?}", e),
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    match zDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving zDesc: {:?}", e),
    }
    #[cfg(feature = "shadow_desc")]
    let zDesc = get_resource(zDesc as usize);
    match yDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving yDesc: {:?}", e),
    }
    #[cfg(feature = "shadow_desc")]
    let yDesc = get_resource(yDesc as usize);
    match bnScaleBiasMeanVarDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving bnScaleBiasMeanVarDesc: {:?}", e),
    }
    #[cfg(feature = "shadow_desc")]
    let bnScaleBiasMeanVarDesc = get_resource(bnScaleBiasMeanVarDesc as usize);
    match activationDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving activationDesc: {:?}", e),
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let result = {
        // cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        //     handle,
        //     mode,
        //     bnOps,
        //     xDesc,
        //     zDesc,
        //     yDesc,
        //     bnScaleBiasMeanVarDesc,
        //     activationDesc,
        //     &mut sizeInBytes,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1522u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&mode), mode.mem_size(),
                    get_address(&bnOps), bnOps.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&zDesc), zDesc.mem_size(),
                    get_address(&yDesc), yDesc.mem_size(),
                    get_address(&bnScaleBiasMeanVarDesc), bnScaleBiasMeanVarDesc.mem_size(),
                    get_address(&activationDesc), activationDesc.mem_size(),
                    get_address(&mut sizeInBytes), sizeInBytes.mem_size(),
                ]
            )
        )
    };
    match sizeInBytes.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending sizeInBytes: {:?}", e),
    }
    match result.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending result: {:?}", e),
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudnnGetBatchNormalizationTrainingExReserveSpaceSizeExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnGetBatchNormalizationTrainingExReserveSpaceSize",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut mode: cudnnBatchNormMode_t = Default::default();
    let mut bnOps: cudnnBatchNormOps_t = Default::default();
    let mut activationDesc: cudnnActivationDescriptor_t = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    match handle.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving handle: {:?}", e),
    }
    match mode.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving mode: {:?}", e),
    }
    match bnOps.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving bnOps: {:?}", e),
    }
    match activationDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving activationDesc: {:?}", e),
    }
    match xDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving xDesc: {:?}", e),
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let mut sizeInBytes: size_t = Default::default();
    let result = {
        // cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        //     handle,
        //     mode,
        //     bnOps,
        //     activationDesc,
        //     xDesc,
        //     &mut sizeInBytes,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1523u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&mode), mode.mem_size(),
                    get_address(&bnOps), bnOps.mem_size(),
                    get_address(&activationDesc), activationDesc.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&mut sizeInBytes), sizeInBytes.mem_size(),
                ]
            )
        )
    };
    match sizeInBytes.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending sizeInBytes: {:?}", e),
    }
    match result.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending result: {:?}", e),
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudnnBatchNormalizationForwardTrainingExExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnBatchNormalizationForwardTrainingEx",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut mode: cudnnBatchNormMode_t = Default::default();
    let mut bnOps: cudnnBatchNormOps_t = Default::default();
    let mut alpha: f64 = Default::default();
    let mut beta: f64 = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut x: MemPtr = Default::default();
    let mut zDesc: cudnnTensorDescriptor_t = Default::default();
    let mut z: MemPtr = Default::default();
    let mut yDesc: cudnnTensorDescriptor_t = Default::default();
    let mut y: MemPtr = Default::default();
    let mut bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t = Default::default();
    let mut bnScale: MemPtr = Default::default();
    let mut bnBias: MemPtr = Default::default();
    let mut exponentialAverageFactor: f64 = Default::default();
    let mut resultRunningMean: MemPtr = Default::default();
    let mut resultRunningVariance: MemPtr = Default::default();
    let mut epsilon: f64 = Default::default();
    let mut saveMean: MemPtr = Default::default();
    let mut saveInvVariance: MemPtr = Default::default();
    let mut activationDesc: cudnnActivationDescriptor_t = Default::default();
    let mut workSpace: MemPtr = Default::default();
    let mut workSpaceSizeInBytes: size_t = Default::default();
    let mut reserveSpace: MemPtr = Default::default();
    let mut reserveSpaceSizeInBytes: size_t = Default::default();
    match handle.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving handle: {:?}", e),
    }
    match mode.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving mode: {:?}", e),
    }
    match bnOps.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving bnOps: {:?}", e),
    }
    match alpha.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving alpha: {:?}", e),
    }
    match beta.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving beta: {:?}", e),
    }
    match xDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving xDesc: {:?}", e),
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    match x.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving x: {:?}", e),
    }
    match zDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving zDesc: {:?}", e),
    }
    #[cfg(feature = "shadow_desc")]
    let zDesc = get_resource(zDesc as usize);
    match z.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving z: {:?}", e),
    }
    match yDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving yDesc: {:?}", e),
    }
    #[cfg(feature = "shadow_desc")]
    let yDesc = get_resource(yDesc as usize);
    match y.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving y: {:?}", e),
    }
    match bnScaleBiasMeanVarDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving bnScaleBiasMeanVarDesc: {:?}", e),
    }
    #[cfg(feature = "shadow_desc")]
    let bnScaleBiasMeanVarDesc = get_resource(bnScaleBiasMeanVarDesc as usize);
    match bnScale.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving bnScale: {:?}", e),
    }
    match bnBias.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving bnBias: {:?}", e),
    }
    match exponentialAverageFactor.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving exponentialAverageFactor: {:?}", e),
    }
    match resultRunningMean.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving resultRunningMean: {:?}", e),
    }
    match resultRunningVariance.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving resultRunningVariance: {:?}", e),
    }
    match epsilon.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving epsilon: {:?}", e),
    }
    match saveMean.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving saveMean: {:?}", e),
    }
    match saveInvVariance.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving saveInvVariance: {:?}", e),
    }
    match activationDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving activationDesc: {:?}", e),
    }
    match workSpace.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving workSpace: {:?}", e),
    }
    match workSpaceSizeInBytes.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving workSpaceSizeInBytes: {:?}", e),
    }
    match reserveSpace.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving reserveSpace: {:?}", e),
    }
    match reserveSpaceSizeInBytes.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving reserveSpaceSizeInBytes: {:?}", e),
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let alpha_ = &alpha as *const f64;
    let beta_ = &beta as *const f64;
    let result = {
        // cudnnBatchNormalizationForwardTrainingEx(
        //     handle,
        //     mode,
        //     bnOps,
        //     alpha_ as *const c_void,
        //     beta_ as *const c_void,
        //     xDesc,
        //     x as *const c_void,
        //     zDesc,
        //     z as *const c_void,
        //     yDesc,
        //     y as *mut c_void,
        //     bnScaleBiasMeanVarDesc,
        //     bnScale as *const c_void,
        //     bnBias as *const c_void,
        //     exponentialAverageFactor,
        //     resultRunningMean as *mut c_void,
        //     resultRunningVariance as *mut c_void,
        //     epsilon,
        //     saveMean as *mut c_void,
        //     saveInvVariance as *mut c_void,
        //     activationDesc,
        //     workSpace as *mut c_void,
        //     workSpaceSizeInBytes,
        //     reserveSpace as *mut c_void,
        //     reserveSpaceSizeInBytes,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1524u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&mode), mode.mem_size(),
                    get_address(&bnOps), bnOps.mem_size(),
                    get_address(&alpha_), alpha_.mem_size(),
                    get_address(&beta_), beta_.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&x), x.mem_size(),
                    get_address(&zDesc), zDesc.mem_size(),
                    get_address(&z), z.mem_size(),
                    get_address(&yDesc), yDesc.mem_size(),
                    get_address(&y), y.mem_size(),
                    get_address(&bnScaleBiasMeanVarDesc), bnScaleBiasMeanVarDesc.mem_size(),
                    get_address(&bnScale), bnScale.mem_size(),
                    get_address(&bnBias), bnBias.mem_size(),
                    get_address(&exponentialAverageFactor), exponentialAverageFactor.mem_size(),
                    get_address(&resultRunningMean), resultRunningMean.mem_size(),
                    get_address(&resultRunningVariance), resultRunningVariance.mem_size(),
                    get_address(&epsilon), epsilon.mem_size(),
                    get_address(&saveMean), saveMean.mem_size(),
                    get_address(&saveInvVariance), saveInvVariance.mem_size(),
                    get_address(&activationDesc), activationDesc.mem_size(),
                    get_address(&workSpace), workSpace.mem_size(),
                    get_address(&workSpaceSizeInBytes), workSpaceSizeInBytes.mem_size(),
                    get_address(&reserveSpace), reserveSpace.mem_size(),
                    get_address(&reserveSpaceSizeInBytes), reserveSpaceSizeInBytes.mem_size(),
                ]
            )
        )
    };
    #[cfg(not(feature = "async_api"))]
    {
        match result.send(channel_sender) {
            Ok(()) => {}
            Err(e) => error!("Error sending result: {:?}", e),
        }
        channel_sender.flush_out().unwrap();
    }
}

pub fn cudnnGetBatchNormalizationBackwardExWorkspaceSizeExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnGetBatchNormalizationBackwardExWorkspaceSize",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut mode: cudnnBatchNormMode_t = Default::default();
    let mut bnOps: cudnnBatchNormOps_t = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut yDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dyDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dzDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dxDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dbnScaleBiasDesc: cudnnTensorDescriptor_t = Default::default();
    let mut activationDesc: cudnnActivationDescriptor_t = Default::default();
    let mut sizeInBytes: size_t = Default::default();
    if let Err(e) = handle.recv(channel_receiver) {
        error!("Error receiving handle: {:?}", e);
    }
    if let Err(e) = mode.recv(channel_receiver) {
        error!("Error receiving mode: {:?}", e);
    }
    if let Err(e) = bnOps.recv(channel_receiver) {
        error!("Error receiving bnOps: {:?}", e);
    }
    if let Err(e) = xDesc.recv(channel_receiver) {
        error!("Error receiving xDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    if let Err(e) = yDesc.recv(channel_receiver) {
        error!("Error receiving yDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let yDesc = get_resource(yDesc as usize);
    if let Err(e) = dyDesc.recv(channel_receiver) {
        error!("Error receiving dyDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dyDesc = get_resource(dyDesc as usize);
    if let Err(e) = dzDesc.recv(channel_receiver) {
        error!("Error receiving dzDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dzDesc = get_resource(dzDesc as usize);
    if let Err(e) = dxDesc.recv(channel_receiver) {
        error!("Error receiving dxDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dxDesc = get_resource(dxDesc as usize);
    if let Err(e) = dbnScaleBiasDesc.recv(channel_receiver) {
        error!("Error receiving dbnScaleBiasDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dbnScaleBiasDesc = get_resource(dbnScaleBiasDesc as usize);
    if let Err(e) = activationDesc.recv(channel_receiver) {
        error!("Error receiving activationDesc: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = {
        // cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        //     handle,
        //     mode,
        //     bnOps,
        //     xDesc,
        //     yDesc,
        //     dyDesc,
        //     dzDesc,
        //     dxDesc,
        //     dbnScaleBiasDesc,
        //     activationDesc,
        //     &mut sizeInBytes,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1525u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&mode), mode.mem_size(),
                    get_address(&bnOps), bnOps.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&yDesc), yDesc.mem_size(),
                    get_address(&dyDesc), dyDesc.mem_size(),
                    get_address(&dzDesc), dzDesc.mem_size(),
                    get_address(&dxDesc), dxDesc.mem_size(),
                    get_address(&dbnScaleBiasDesc), dbnScaleBiasDesc.mem_size(),
                    get_address(&activationDesc), activationDesc.mem_size(),
                    get_address(&mut sizeInBytes), sizeInBytes.mem_size(),
                ]
            )
        )
    };
    if let Err(e) = sizeInBytes.send(channel_sender) {
        error!("Error sending sizeInBytes: {:?}", e);
    }
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudnnBatchNormalizationBackwardExExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnBatchNormalizationBackwardEx",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut mode: cudnnBatchNormMode_t = Default::default();
    let mut bnOps: cudnnBatchNormOps_t = Default::default();
    let mut alphaDataDiff: f64 = Default::default();
    let mut betaDataDiff: f64 = Default::default();
    let mut alphaParamDiff: f64 = Default::default();
    let mut betaParamDiff: f64 = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut xData: MemPtr = Default::default();
    let mut yDesc: cudnnTensorDescriptor_t = Default::default();
    let mut yData: MemPtr = Default::default();
    let mut dyDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dyData: MemPtr = Default::default();
    let mut dzDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dzData: MemPtr = Default::default();
    let mut dxDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dxData: MemPtr = Default::default();
    let mut dbnScaleBiasDesc: cudnnTensorDescriptor_t = Default::default();
    let mut bnScaleData: MemPtr = Default::default();
    let mut bnBiasData: MemPtr = Default::default();
    let mut dBnScaleData: MemPtr = Default::default();
    let mut dBnBiasData: MemPtr = Default::default();
    let mut epsilon: f64 = Default::default();
    let mut saveMean: MemPtr = Default::default();
    let mut saveInvVariance: MemPtr = Default::default();
    let mut activationDesc: cudnnActivationDescriptor_t = Default::default();
    let mut workSpace: MemPtr = Default::default();
    let mut workSpaceSizeInBytes: size_t = Default::default();
    let mut reserveSpace: MemPtr = Default::default();
    let mut reserveSpaceSizeInBytes: size_t = Default::default();
    if let Err(e) = handle.recv(channel_receiver) {
        error!("Error receiving handle: {:?}", e);
    }
    if let Err(e) = mode.recv(channel_receiver) {
        error!("Error receiving mode: {:?}", e);
    }
    if let Err(e) = bnOps.recv(channel_receiver) {
        error!("Error receiving bnOps: {:?}", e);
    }
    if let Err(e) = alphaDataDiff.recv(channel_receiver) {
        error!("Error receiving alphaDataDiff: {:?}", e);
    }
    if let Err(e) = betaDataDiff.recv(channel_receiver) {
        error!("Error receiving betaDataDiff: {:?}", e);
    }
    if let Err(e) = alphaParamDiff.recv(channel_receiver) {
        error!("Error receiving alphaParamDiff: {:?}", e);
    }
    if let Err(e) = betaParamDiff.recv(channel_receiver) {
        error!("Error receiving betaParamDiff: {:?}", e);
    }
    if let Err(e) = xDesc.recv(channel_receiver) {
        error!("Error receiving xDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    if let Err(e) = xData.recv(channel_receiver) {
        error!("Error receiving xData: {:?}", e);
    }
    if let Err(e) = yDesc.recv(channel_receiver) {
        error!("Error receiving yDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let yDesc = get_resource(yDesc as usize);
    if let Err(e) = yData.recv(channel_receiver) {
        error!("Error receiving yData: {:?}", e);
    }
    if let Err(e) = dyDesc.recv(channel_receiver) {
        error!("Error receiving dyDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dyDesc = get_resource(dyDesc as usize);
    if let Err(e) = dyData.recv(channel_receiver) {
        error!("Error receiving dyData: {:?}", e);
    }
    if let Err(e) = dzDesc.recv(channel_receiver) {
        error!("Error receiving dzDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dzDesc = get_resource(dzDesc as usize);
    if let Err(e) = dzData.recv(channel_receiver) {
        error!("Error receiving dzData: {:?}", e);
    }
    if let Err(e) = dxDesc.recv(channel_receiver) {
        error!("Error receiving dxDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dxDesc = get_resource(dxDesc as usize);
    if let Err(e) = dxData.recv(channel_receiver) {
        error!("Error receiving dxData: {:?}", e);
    }
    if let Err(e) = dbnScaleBiasDesc.recv(channel_receiver) {
        error!("Error receiving dbnScaleBiasDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dbnScaleBiasDesc = get_resource(dbnScaleBiasDesc as usize);
    if let Err(e) = bnScaleData.recv(channel_receiver) {
        error!("Error receiving bnScaleData: {:?}", e);
    }
    if let Err(e) = bnBiasData.recv(channel_receiver) {
        error!("Error receiving bnBiasData: {:?}", e);
    }
    if let Err(e) = dBnScaleData.recv(channel_receiver) {
        error!("Error receiving dBnScaleData: {:?}", e);
    }
    if let Err(e) = dBnBiasData.recv(channel_receiver) {
        error!("Error receiving dBnBiasData: {:?}", e);
    }
    if let Err(e) = epsilon.recv(channel_receiver) {
        error!("Error receiving epsilon: {:?}", e);
    }
    if let Err(e) = saveMean.recv(channel_receiver) {
        error!("Error receiving saveMean: {:?}", e);
    }
    if let Err(e) = saveInvVariance.recv(channel_receiver) {
        error!("Error receiving saveInvVariance: {:?}", e);
    }
    if let Err(e) = activationDesc.recv(channel_receiver) {
        error!("Error receiving activationDesc: {:?}", e);
    }
    if let Err(e) = workSpace.recv(channel_receiver) {
        error!("Error receiving workSpace: {:?}", e);
    }
    if let Err(e) = workSpaceSizeInBytes.recv(channel_receiver) {
        error!("Error receiving workSpaceSizeInBytes: {:?}", e);
    }
    if let Err(e) = reserveSpace.recv(channel_receiver) {
        error!("Error receiving reserveSpace: {:?}", e);
    }
    if let Err(e) = reserveSpaceSizeInBytes.recv(channel_receiver) {
        error!("Error receiving reserveSpaceSizeInBytes: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let alphaDataDiff_ = &alphaDataDiff as *const f64;
    let betaDataDiff_ = &betaDataDiff as *const f64;
    let alphaParamDiff_ = &alphaParamDiff as *const f64;
    let betaParamDiff_ = &betaParamDiff as *const f64;
    let result = {
        // cudnnBatchNormalizationBackwardEx(
        //     handle,
        //     mode,
        //     bnOps,
        //     alphaDataDiff_ as *const c_void,
        //     betaDataDiff_ as *const c_void,
        //     alphaParamDiff_ as *const c_void,
        //     betaParamDiff_ as *const c_void,
        //     xDesc,
        //     xData as *const c_void,
        //     yDesc,
        //     yData as *const c_void,
        //     dyDesc,
        //     dyData as *const c_void,
        //     dzDesc,
        //     dzData as *mut c_void,
        //     dxDesc,
        //     dxData as *mut c_void,
        //     dbnScaleBiasDesc,
        //     bnScaleData as *const c_void,
        //     bnBiasData as *const c_void,
        //     dBnScaleData as *mut c_void,
        //     dBnBiasData as *mut c_void,
        //     epsilon,
        //     saveMean as *const c_void,
        //     saveInvVariance as *const c_void,
        //     activationDesc,
        //     workSpace as *mut c_void,
        //     workSpaceSizeInBytes,
        //     reserveSpace as *mut c_void,
        //     reserveSpaceSizeInBytes,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1526u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&mode), mode.mem_size(),
                    get_address(&bnOps), bnOps.mem_size(),
                    get_address(&alphaDataDiff_), alphaDataDiff_.mem_size(),
                    get_address(&betaDataDiff_), betaDataDiff_.mem_size(),
                    get_address(&alphaParamDiff_), alphaParamDiff_.mem_size(),
                    get_address(&betaParamDiff_), betaParamDiff_.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&xData), xData.mem_size(),
                    get_address(&yDesc), yDesc.mem_size(),
                    get_address(&yData), yData.mem_size(),
                    get_address(&dyDesc), dyDesc.mem_size(),
                    get_address(&dyData), dyData.mem_size(),
                    get_address(&dzDesc), dzDesc.mem_size(),
                    get_address(&dzData), dzData.mem_size(),
                    get_address(&dxDesc), dxDesc.mem_size(),
                    get_address(&dxData), dxData.mem_size(),
                    get_address(&dbnScaleBiasDesc), dbnScaleBiasDesc.mem_size(),
                    get_address(&bnScaleData), bnScaleData.mem_size(),
                    get_address(&bnBiasData), bnBiasData.mem_size(),
                    get_address(&dBnScaleData), dBnScaleData.mem_size(),
                    get_address(&dBnBiasData), dBnBiasData.mem_size(),
                    get_address(&epsilon), epsilon.mem_size(),
                    get_address(&saveMean), saveMean.mem_size(),
                    get_address(&saveInvVariance), saveInvVariance.mem_size(),
                    get_address(&activationDesc), activationDesc.mem_size(),
                    get_address(&workSpace), workSpace.mem_size(),
                    get_address(&workSpaceSizeInBytes), workSpaceSizeInBytes.mem_size(),
                    get_address(&reserveSpace), reserveSpace.mem_size(),
                    get_address(&reserveSpaceSizeInBytes), reserveSpaceSizeInBytes.mem_size(),
                ]
            )
        )
    };
    #[cfg(not(feature = "async_api"))]
    {
        if let Err(e) = result.send(channel_sender) {
            error!("Error sending result: {:?}", e);
        }
        channel_sender.flush_out().unwrap();
    }
}

pub fn cudnnGetConvolutionBackwardDataAlgorithm_v7Exe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnGetConvolutionBackwardDataAlgorithm_v7",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut wDesc: cudnnFilterDescriptor_t = Default::default();
    let mut dyDesc: cudnnTensorDescriptor_t = Default::default();
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    let mut dxDesc: cudnnTensorDescriptor_t = Default::default();
    let mut requestedAlgoCount: c_int = Default::default();
    let mut returnedAlgoCount: c_int = Default::default();
    let mut perfResults: Vec<cudnnConvolutionBwdDataAlgoPerf_t> = Default::default();
    match handle.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving handle: {:?}", e);
        }
    }
    match wDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving wDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let wDesc = get_resource(wDesc as usize);
    match dyDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving xDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let dyDesc = get_resource(dyDesc as usize);
    match convDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving convDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let convDesc = get_resource(convDesc as usize);
    match dxDesc.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving yDesc: {:?}", e);
        }
    }
    #[cfg(feature = "shadow_desc")]
    let dxDesc = get_resource(dxDesc as usize);
    match requestedAlgoCount.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving requestedAlgoCount: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    perfResults.resize(requestedAlgoCount as usize, Default::default());
    let result = {
        // cudnnGetConvolutionBackwardDataAlgorithm_v7(
        //     handle,
        //     wDesc,
        //     dyDesc,
        //     convDesc,
        //     dxDesc,
        //     requestedAlgoCount,
        //     &mut returnedAlgoCount,
        //     perfResults.as_mut_ptr(),
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1527u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&wDesc), wDesc.mem_size(),
                    get_address(&dyDesc), dyDesc.mem_size(),
                    get_address(&convDesc), convDesc.mem_size(),
                    get_address(&dxDesc), dxDesc.mem_size(),
                    get_address(&requestedAlgoCount), requestedAlgoCount.mem_size(),
                    get_address(&mut returnedAlgoCount), returnedAlgoCount.mem_size(),
                    get_address(&perfResults), perfResults.mem_size(),
                ]
            )
        )
    };
    match returnedAlgoCount.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("Error sending returnedAlgoCount: {:?}", e);
        }
    }
    match perfResults.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("Error sending perfResults: {:?}", e);
        }
    }
    match result.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("Error sending result: {:?}", e);
        }
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudnnConvolutionBackwardDataExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnConvolutionBackwardData",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut alpha: f64 = Default::default();
    let mut wDesc: cudnnFilterDescriptor_t = Default::default();

    let mut w: MemPtr = Default::default();
    let mut dyDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dy: MemPtr = Default::default();
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    let mut algo: cudnnConvolutionBwdDataAlgo_t = Default::default();
    let mut workSpace: MemPtr = Default::default();
    let mut workSpaceSizeInBytes: size_t = Default::default();
    let mut beta: f64 = Default::default();
    let mut dxDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dx: MemPtr = Default::default();
    if let Err(e) = handle.recv(channel_receiver) {
        error!("Error receiving handle: {:?}", e);
    }
    if let Err(e) = alpha.recv(channel_receiver) {
        error!("Error receiving alpha: {:?}", e);
    }
    if let Err(e) = wDesc.recv(channel_receiver) {
        error!("Error receiving wDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let wDesc = get_resource(wDesc as usize);
    if let Err(e) = w.recv(channel_receiver) {
        error!("Error receiving w: {:?}", e);
    }
    if let Err(e) = dyDesc.recv(channel_receiver) {
        error!("Error receiving dyDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dyDesc = get_resource(dyDesc as usize);
    if let Err(e) = dy.recv(channel_receiver) {
        error!("Error receiving dy: {:?}", e);
    }
    if let Err(e) = convDesc.recv(channel_receiver) {
        error!("Error receiving convDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let convDesc = get_resource(convDesc as usize);
    if let Err(e) = algo.recv(channel_receiver) {
        error!("Error receiving algo: {:?}", e);
    }
    if let Err(e) = workSpace.recv(channel_receiver) {
        error!("Error receiving workSpace: {:?}", e);
    }
    if let Err(e) = workSpaceSizeInBytes.recv(channel_receiver) {
        error!("Error receiving workSpaceSizeInBytes: {:?}", e);
    }
    if let Err(e) = beta.recv(channel_receiver) {
        error!("Error receiving beta: {:?}", e);
    }
    if let Err(e) = dxDesc.recv(channel_receiver) {
        error!("Error receiving dxDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dxDesc = get_resource(dxDesc as usize);
    if let Err(e) = dx.recv(channel_receiver) {
        error!("Error receiving dx: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let alpha_ = &alpha as *const f64;
    let beta_ = &beta as *const f64;
    let result = {
        // cudnnConvolutionBackwardData(
        //     handle,
        //     alpha_ as *const c_void,
        //     wDesc,
        //     w as *const c_void,
        //     dyDesc,
        //     dy as *const c_void,
        //     convDesc,
        //     algo,
        //     workSpace as *mut c_void,
        //     workSpaceSizeInBytes,
        //     beta_ as *const c_void,
        //     dxDesc,
        //     dx as *mut c_void,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1528u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&alpha_), alpha_.mem_size(),
                    get_address(&wDesc), wDesc.mem_size(),
                    get_address(&w), w.mem_size(),
                    get_address(&dyDesc), dyDesc.mem_size(),
                    get_address(&dy), dy.mem_size(),
                    get_address(&convDesc), convDesc.mem_size(),
                    get_address(&algo), algo.mem_size(),
                    get_address(&workSpace), workSpace.mem_size(),
                    get_address(&workSpaceSizeInBytes), workSpaceSizeInBytes.mem_size(),
                    get_address(&beta_), beta_.mem_size(),
                    get_address(&dxDesc), dxDesc.mem_size(),
                    get_address(&dx), dx.mem_size(),
                ]
            )
        )
    };
    #[cfg(not(feature = "async_api"))]
    {
        if let Err(e) = result.send(channel_sender) {
            error!("Error sending result: {:?}", e);
        }
        channel_sender.flush_out().unwrap();
    }
}

pub fn cudnnGetConvolutionBackwardFilterAlgorithm_v7Exe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnGetConvolutionBackwardFilterAlgorithm_v7",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dyDesc: cudnnTensorDescriptor_t = Default::default();
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    let mut dwDesc: cudnnFilterDescriptor_t = Default::default();
    let mut requestedAlgoCount: c_int = Default::default();
    let mut returnedAlgoCount: c_int = Default::default();
    let mut perfResults: Vec<cudnnConvolutionBwdFilterAlgoPerf_t> = Default::default();

    if let Err(e) = handle.recv(channel_receiver) {
        error!("Error receiving handle: {:?}", e);
    }
    if let Err(e) = xDesc.recv(channel_receiver) {
        error!("Error receiving xDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    if let Err(e) = dyDesc.recv(channel_receiver) {
        error!("Error receiving dyDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dyDesc = get_resource(dyDesc as usize);
    if let Err(e) = convDesc.recv(channel_receiver) {
        error!("Error receiving convDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let convDesc = get_resource(convDesc as usize);
    if let Err(e) = dwDesc.recv(channel_receiver) {
        error!("Error receiving dwDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dwDesc = get_resource(dwDesc as usize);
    if let Err(e) = requestedAlgoCount.recv(channel_receiver) {
        error!("Error receiving requestedAlgoCount: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    perfResults.resize(requestedAlgoCount as usize, Default::default());
    let result = {
        // cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        //     handle,
        //     xDesc,
        //     dyDesc,
        //     convDesc,
        //     dwDesc,
        //     requestedAlgoCount,
        //     &mut returnedAlgoCount,
        //     perfResults.as_mut_ptr(),
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1529u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&dyDesc), dyDesc.mem_size(),
                    get_address(&convDesc), convDesc.mem_size(),
                    get_address(&dwDesc), dwDesc.mem_size(),
                    get_address(&requestedAlgoCount), requestedAlgoCount.mem_size(),
                    get_address(&mut returnedAlgoCount), returnedAlgoCount.mem_size(),
                    get_address(&perfResults), perfResults.mem_size(),
                ]
            )
        )
    };
    if let Err(e) = returnedAlgoCount.send(channel_sender) {
        error!("Error sending returnedAlgoCount: {:?}", e);
    }
    if let Err(e) = perfResults.send(channel_sender) {
        error!("Error sending perfResults: {:?}", e);
    }
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudnnConvolutionBackwardFilterExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnConvolutionBackwardFilter",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut alpha: f64 = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut x: MemPtr = Default::default();
    let mut dyDesc: cudnnTensorDescriptor_t = Default::default();
    let mut dy: MemPtr = Default::default();
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    let mut algo: cudnnConvolutionBwdFilterAlgo_t = Default::default();
    let mut workSpace: MemPtr = Default::default();
    let mut workSpaceSizeInBytes: size_t = Default::default();
    let mut beta: f64 = Default::default();
    let mut dwDesc: cudnnFilterDescriptor_t = Default::default();
    let mut dw: MemPtr = Default::default();
    if let Err(e) = handle.recv(channel_receiver) {
        error!("Error receiving handle: {:?}", e);
    }
    if let Err(e) = alpha.recv(channel_receiver) {
        error!("Error receiving alpha: {:?}", e);
    }
    if let Err(e) = xDesc.recv(channel_receiver) {
        error!("Error receiving xDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    if let Err(e) = x.recv(channel_receiver) {
        error!("Error receiving x: {:?}", e);
    }
    if let Err(e) = dyDesc.recv(channel_receiver) {
        error!("Error receiving dyDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dyDesc = get_resource(dyDesc as usize);
    if let Err(e) = dy.recv(channel_receiver) {
        error!("Error receiving dy: {:?}", e);
    }
    if let Err(e) = convDesc.recv(channel_receiver) {
        error!("Error receiving convDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let convDesc = get_resource(convDesc as usize);
    if let Err(e) = algo.recv(channel_receiver) {
        error!("Error receiving algo: {:?}", e);
    }
    if let Err(e) = workSpace.recv(channel_receiver) {
        error!("Error receiving workSpace: {:?}", e);
    }
    if let Err(e) = workSpaceSizeInBytes.recv(channel_receiver) {
        error!("Error receiving workSpaceSizeInBytes: {:?}", e);
    }
    if let Err(e) = beta.recv(channel_receiver) {
        error!("Error receiving beta: {:?}", e);
    }
    if let Err(e) = dwDesc.recv(channel_receiver) {
        error!("Error receiving dwDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let dwDesc = get_resource(dwDesc as usize);
    if let Err(e) = dw.recv(channel_receiver) {
        error!("Error receiving dw: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let alpha_ = &alpha as *const f64;
    let beta_ = &beta as *const f64;
    let result = {
        // cudnnConvolutionBackwardFilter(
        //     handle,
        //     alpha_ as *const c_void,
        //     xDesc,
        //     x as *const c_void,
        //     dyDesc,
        //     dy as *const c_void,
        //     convDesc,
        //     algo,
        //     workSpace as *mut c_void,
        //     workSpaceSizeInBytes,
        //     beta_ as *const c_void,
        //     dwDesc,
        //     dw as *mut c_void,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1530u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&alpha_), alpha_.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&x), x.mem_size(),
                    get_address(&dyDesc), dyDesc.mem_size(),
                    get_address(&dy), dy.mem_size(),
                    get_address(&convDesc), convDesc.mem_size(),
                    get_address(&algo), algo.mem_size(),
                    get_address(&workSpace), workSpace.mem_size(),
                    get_address(&workSpaceSizeInBytes), workSpaceSizeInBytes.mem_size(),
                    get_address(&beta_), beta_.mem_size(),
                    get_address(&dwDesc), dwDesc.mem_size(),
                    get_address(&dw), dw.mem_size(),
                ]
            )
        )
    };
    #[cfg(not(feature = "async_api"))]
    {
        if let Err(e) = result.send(channel_sender) {
            error!("Error sending result: {:?}", e);
        }
        channel_sender.flush_out().unwrap();
    }
}

pub fn cudnnBatchNormalizationForwardInferenceExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudnnBatchNormalizationForwardInference",
        std::file!(),
        std::line!()
    );
    let mut handle: cudnnHandle_t = Default::default();
    let mut mode: cudnnBatchNormMode_t = Default::default();
    let mut alpha: f64 = Default::default();
    let mut beta: f64 = Default::default();
    let mut xDesc: cudnnTensorDescriptor_t = Default::default();
    let mut x: MemPtr = Default::default();
    let mut yDesc: cudnnTensorDescriptor_t = Default::default();
    let mut y: MemPtr = Default::default();
    let mut bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t = Default::default();
    let mut bnScale: MemPtr = Default::default();
    let mut bnBias: MemPtr = Default::default();
    let mut estimatedMean: MemPtr = Default::default();
    let mut estimatedVariance: MemPtr = Default::default();
    let mut epsilon: f64 = Default::default();
    if let Err(e) = handle.recv(channel_receiver) {
        error!("Error receiving handle: {:?}", e);
    }
    if let Err(e) = mode.recv(channel_receiver) {
        error!("Error receiving mode: {:?}", e);
    }
    if let Err(e) = alpha.recv(channel_receiver) {
        error!("Error receiving alpha: {:?}", e);
    }
    if let Err(e) = beta.recv(channel_receiver) {
        error!("Error receiving beta: {:?}", e);
    }
    if let Err(e) = xDesc.recv(channel_receiver) {
        error!("Error receiving xDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let xDesc = get_resource(xDesc as usize);
    if let Err(e) = x.recv(channel_receiver) {
        error!("Error receiving x: {:?}", e);
    }
    if let Err(e) = yDesc.recv(channel_receiver) {
        error!("Error receiving yDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let yDesc = get_resource(yDesc as usize);
    if let Err(e) = y.recv(channel_receiver) {
        error!("Error receiving y: {:?}", e);
    }
    if let Err(e) = bnScaleBiasMeanVarDesc.recv(channel_receiver) {
        error!("Error receiving bnScaleBiasMeanVarDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let bnScaleBiasMeanVarDesc = get_resource(bnScaleBiasMeanVarDesc as usize);
    if let Err(e) = bnScale.recv(channel_receiver) {
        error!("Error receiving bnScale: {:?}", e);
    }
    if let Err(e) = bnBias.recv(channel_receiver) {
        error!("Error receiving bnBias: {:?}", e);
    }
    if let Err(e) = estimatedMean.recv(channel_receiver) {
        error!("Error receiving estimatedMean: {:?}", e);
    }
    if let Err(e) = estimatedVariance.recv(channel_receiver) {
        error!("Error receiving estimatedVariance: {:?}", e);
    }
    if let Err(e) = epsilon.recv(channel_receiver) {
        error!("Error receiving epsilon: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let alpha_ = &alpha as *const f64;
    let beta_ = &beta as *const f64;
    let result = {
        // cudnnBatchNormalizationForwardInference(
        //     handle,
        //     mode,
        //     alpha_ as *const c_void,
        //     beta_ as *const c_void,
        //     xDesc,
        //     x as *const c_void,
        //     yDesc,
        //     y as *mut c_void,
        //     bnScaleBiasMeanVarDesc,
        //     bnScale as *const c_void,
        //     bnBias as *const c_void,
        //     estimatedMean as *const c_void,
        //     estimatedVariance as *const c_void,
        //     epsilon,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1531u64,
                0u64,
                vec![
                    get_address(&handle), handle.mem_size(),
                    get_address(&mode), mode.mem_size(),
                    get_address(&alpha_), alpha_.mem_size(),
                    get_address(&beta_), beta_.mem_size(),
                    get_address(&xDesc), xDesc.mem_size(),
                    get_address(&x), x.mem_size(),
                    get_address(&yDesc), yDesc.mem_size(),
                    get_address(&y), y.mem_size(),
                    get_address(&bnScaleBiasMeanVarDesc), bnScaleBiasMeanVarDesc.mem_size(),
                    get_address(&bnScale), bnScale.mem_size(),
                    get_address(&bnBias), bnBias.mem_size(),
                    get_address(&estimatedMean), estimatedMean.mem_size(),
                    get_address(&estimatedVariance), estimatedVariance.mem_size(),
                    get_address(&epsilon), epsilon.mem_size(),
                ]
            )
        )
    };
    #[cfg(not(feature = "async_api"))]
    {
        if let Err(e) = result.send(channel_sender) {
            error!("Error sending result: {:?}", e);
        }
        channel_sender.flush_out().unwrap();
    }
}



pub fn cudnnGetConvolutionNdForwardOutputDimExe<T: CommChannel> (
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!("[{}:{}] cudnnGetConvolutionNdForwardOutputDim", std::file!(), std::line!());
    let mut convDesc: cudnnConvolutionDescriptor_t = Default::default();
    let mut inputTensorDesc: cudnnTensorDescriptor_t = Default::default();
    let mut filterDesc: cudnnFilterDescriptor_t = Default::default();
    let mut nbDims: c_int = Default::default();

    if let Err(e) = convDesc.recv(channel_receiver) {
        error!("Error receiving convDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let convDesc = get_resource(convDesc as usize);
    if let Err(e) = inputTensorDesc.recv(channel_receiver) {
        error!("Error receiving inputTensorDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let inputTensorDesc = get_resource(inputTensorDesc as usize);
    if let Err(e) = filterDesc.recv(channel_receiver) {
        error!("Error receiving wDesc: {:?}", e);
    }
    #[cfg(feature = "shadow_desc")]
    let filterDesc = get_resource(filterDesc as usize);
    if let Err(e) = nbDims.recv(channel_receiver) {
        error!("Error receiving n: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let mut tensorOutputDim_ = vec![0; nbDims as usize];
    let result = {
        // cudnnGetConvolutionNdForwardOutputDim(
        //     convDesc,
        //     inputTensorDesc,
        //     filterDesc,
        //     nbDims,
        //     tensorOutputDim_.as_mut_ptr() as *mut c_int,
        // )
        cudnnStatus_t::from(
            pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                1533u64,
                0u64,
                vec![
                    get_address(&convDesc), convDesc.mem_size(),
                    get_address(&inputTensorDesc), inputTensorDesc.mem_size(),
                    get_address(&filterDesc), filterDesc.mem_size(),
                    get_address(&nbDims), nbDims.mem_size(),
                    get_address(&mut tensorOutputDim_), tensorOutputDim_.mem_size(),
                ]
            )
        )
    };
    if let Err(e) = tensorOutputDim_.send(channel_sender) {
        error!("Error sending tensorDimA: {:?}", e);
    }
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudnnGetErrorStringExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!("[{}:{}] cudnnGetErrorString", std::file!(), std::line!());
    let mut status: cudnnStatus_t = Default::default();
    if let Err(e) = status.recv(channel_receiver) {
        error!("Error receiving status: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = {
        // cudnnGetErrorString(status)
        pos_process(
            POS_CUDA_WS.lock().unwrap().get_ptr(),
            1535u64,
            0u64,
            vec![
                get_address(&status), status.mem_size(),
            ]
        ) as *const i8
    };
    // transfer to Vec<u8>
    let result = unsafe { std::ffi::CStr::from_ptr(result).to_bytes().to_vec() };
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}
