extern crate log;
mod cuda_lib;
mod dispatcher;

use cuda_lib::*;
extern crate codegen;
extern crate network;

use codegen::gen_exe;
use dispatcher::dispatch;
use network::{
    ringbufferchannel::{
        RingBuffer, SHMChannelBufferManager, SHM_NAME_CTOS, SHM_NAME_STOC, SHM_SIZE,
    },
    type_impl::{
        basic::MemPtr,
        cuda::{CUdevice, CUdeviceptr, CUfunction, CUmodule, CUresult, CUstream},
        cudart::{
            cudaDeviceProp, cudaError_t, cudaMemcpyKind, cudaStreamCaptureStatus, cudaStream_t,
            dim3,
        },
        nvml::nvmlReturn_t,
    },
    CommChannel, CommChannelError, Transportable,
};

#[allow(unused_imports)]
use log::{debug, error, info, log_enabled, Level};

extern crate lazy_static;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    // client_address -> module
    static ref MODULES: Mutex<HashMap<MemPtr, CUmodule>> = Mutex::new(HashMap::new());
    // host_func -> device_func
    static ref FUNCTIONS: Mutex<HashMap<MemPtr, CUfunction>> = Mutex::new(HashMap::new());
    // host_var -> device_var
    static ref VARIABLES: Mutex<HashMap<MemPtr, CUdeviceptr>> = Mutex::new(HashMap::new());
}

fn add_module(client_address: MemPtr, module: CUmodule) {
    MODULES.lock().unwrap().insert(client_address, module);
}

fn get_module(client_address: MemPtr) -> Option<CUmodule> {
    MODULES.lock().unwrap().get(&client_address).cloned()
}

fn add_function(host_func: MemPtr, device_func: CUfunction) {
    FUNCTIONS.lock().unwrap().insert(host_func, device_func);
}

fn get_function(host_func: MemPtr) -> Option<CUfunction> {
    FUNCTIONS.lock().unwrap().get(&host_func).cloned()
}

fn add_variable(host_var: MemPtr, device_var: CUdeviceptr) {
    VARIABLES.lock().unwrap().insert(host_var, device_var);
}

fn get_variable(host_var: MemPtr) -> Option<CUdeviceptr> {
    VARIABLES.lock().unwrap().get(&host_var).cloned()
}

fn create_buffer() -> (
    RingBuffer<SHMChannelBufferManager>,
    RingBuffer<SHMChannelBufferManager>,
) {
    let manager_sender = SHMChannelBufferManager::new_server(SHM_NAME_STOC, SHM_SIZE).unwrap();
    let manager_receiver = SHMChannelBufferManager::new_server(SHM_NAME_CTOS, SHM_SIZE).unwrap();
    (
        RingBuffer::new(manager_sender),
        RingBuffer::new(manager_receiver),
    )
}

fn receive_request<T: CommChannel>(channel_receiver: &mut T) -> Result<i32, CommChannelError> {
    let mut proc_id = 0;
    if let Ok(()) = proc_id.recv(channel_receiver) {
        Ok(proc_id)
    } else {
        Err(CommChannelError::IoError)
    }
}

pub fn launch_server() {
    let (mut channel_sender, mut channel_receiver) = create_buffer();
    info!("[{}:{}] shm buffer created", std::file!(), std::line!());
    if let cudaError_t::cudaSuccess = unsafe { cudaDeviceSynchronize() } {
        info!(
            "[{}:{}] cuda driver initialized",
            std::file!(),
            std::line!()
        );
    } else {
        error!(
            "[{}:{}] failed to initialize cuda driver",
            std::file!(),
            std::line!()
        );
        panic!();
    }

    loop {
        if let Ok(proc_id) = receive_request(&mut channel_receiver) {
            dispatch(proc_id, &mut channel_sender, &mut channel_receiver);
        } else {
            error!(
                "[{}:{}] failed to receive request",
                std::file!(),
                std::line!()
            );
            break;
        }
    }

    info!("[{}:{}] server terminated", std::file!(), std::line!());
}