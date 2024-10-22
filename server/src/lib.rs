#![allow(dead_code)]
extern crate log;
mod dispatcher;

extern crate codegen;
extern crate cudasys;
extern crate network;

use codegen::gen_exe;
#[cfg(feature = "async_api")]
use codegen::gen_exe_async;
use cudasys::{
    cuda::{CUdeviceptr, CUfunction, CUmodule},
    cudart::{cudaDeviceSynchronize, cudaError_t, cudaGetDeviceCount, cudaSetDevice},
};
use dispatcher::dispatch;

#[cfg(feature = "emulator")]
use network::ringbufferchannel::EmulatorChannel;

#[allow(unused_imports)]
use network::{
    ringbufferchannel::{RDMAChannel, SHMChannel},
    type_impl::MemPtr,
    Channel, CommChannel, CommChannelError, Transportable, MemorySize, CONFIG,
};

#[allow(unused_imports)]
use log::{debug, error, info, log_enabled, Level};

extern crate lazy_static;
use lazy_static::lazy_static;
use std::{boxed::Box, ptr};
use std::collections::HashMap;
use std::sync::Mutex;

extern {
    // fn new_pos_workspace(argc: i32, argv: *const *const i8) -> *mut std::ffi::c_void;
    fn pos_create_workspace_cuda() -> *mut std::ffi::c_void;
    // fn pos_workspace_init(pos_cuda_ws: *mut std::ffi::c_void);
    fn call_pos_process(pos_cuda_ws: *mut std::ffi::c_void, api_id: u64, uuid: u64, param_desps: *mut std::ffi::c_void, param_num: i32) -> i32;
    fn pos_destory_workspace_cuda(pos_cuda_ws: *mut std::ffi::c_void) -> i32;
}

pub struct POSWorkspace(*mut std::ffi::c_void);
unsafe impl Send for POSWorkspace {}
unsafe impl Sync for POSWorkspace {}
impl POSWorkspace {
    pub fn get_ptr(&self) -> *mut std::ffi::c_void {
        self.0
    }
}

lazy_static! {
    // client_address -> module
    static ref MODULES: Mutex<HashMap<MemPtr, CUmodule>> = Mutex::new(HashMap::new());
    // host_func -> device_func
    static ref FUNCTIONS: Mutex<HashMap<MemPtr, CUfunction>> = Mutex::new(HashMap::new());
    // host_var -> device_var
    static ref VARIABLES: Mutex<HashMap<MemPtr, CUdeviceptr>> = Mutex::new(HashMap::new());

    static ref RESOURCES: Mutex<HashMap<usize, usize>> = Mutex::new(HashMap::new());

    pub static ref POS_CUDA_WS: Mutex<POSWorkspace> = {
        // let args: Vec<String> = std::env::args().collect();
        // let argc = args.len() as i32;
        // let argv: Vec<*const i8> = args.iter()
        //     .map(|arg| std::ffi::CString::new(arg.as_str()).unwrap().into_raw() as *const i8)
        //     .collect();
        let pos = unsafe { pos_create_workspace_cuda() };
        // unsafe { pos_workspace_init(pos) }
        Mutex::new(POSWorkspace(pos))
    };
}

// pub fn get_address<T>(value: &mut T) -> usize {
//     let ptr: *mut T = value as *mut T;
//     ptr as usize
// }
pub fn get_address<T>(value: &T) -> usize {
    let ptr: *const T = ptr::addr_of!(*value);
    ptr as usize
}

pub fn pos_process(pos_cuda_ws: *mut std::ffi::c_void, api_id: u64, uuid: u64, mut param_desps: Vec<usize>) -> i32 {
    unsafe { call_pos_process(pos_cuda_ws, api_id, uuid, param_desps.as_mut_ptr() as *mut std::ffi::c_void, (param_desps.len() / 2) as i32) }
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

fn add_resource(id: usize, res: usize) {
    RESOURCES.lock().unwrap().insert(id, res);
}

fn get_resource(id: usize) -> usize {
    RESOURCES.lock().unwrap().get(&id).cloned().unwrap()
}

fn remove_resource(id: usize) -> usize {
    RESOURCES.lock().unwrap().remove(&id).unwrap()
}

#[cfg(not(feature = "emulator"))]
fn create_buffer() -> (Channel, Channel) {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the server side, the sender's name is stoc_channel_name,
    // receiver's name is ctos_channel_name.
    match CONFIG.comm_type.as_str() {
        #[cfg(feature = "shm")]
        "shm" => {
            let sender = SHMChannel::new_server(&CONFIG.stoc_channel_name, CONFIG.buf_size).unwrap();
            let receiver = SHMChannel::new_server(&CONFIG.ctos_channel_name, CONFIG.buf_size).unwrap();
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
        }
        #[cfg(feature = "rdma")]
        "rdma" => {
            // Make sure to new receiver first! Client side sender will handshake with it first.
            let receiver = RDMAChannel::new_server(
                &CONFIG.ctos_channel_name,
                CONFIG.buf_size,
                CONFIG.receiver_socket.parse().unwrap(),
            ).unwrap();
            let sender = RDMAChannel::new_server(
                &CONFIG.stoc_channel_name,
                CONFIG.buf_size,
                CONFIG.sender_socket.parse().unwrap(),
            ).unwrap();
            (Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
        }
        &_ => panic!("Unsupported communication type in config"),
    }
}

#[cfg(feature = "emulator")]
fn create_buffer() -> (Channel, Channel) {
    let sender = SHMChannel::new_server(&CONFIG.stoc_channel_name, CONFIG.buf_size).unwrap();
    let receiver = SHMChannel::new_server(&CONFIG.ctos_channel_name, CONFIG.buf_size).unwrap();
    (
        Channel::new(Box::new(EmulatorChannel::new(Box::new(sender)))),
        Channel::new(Box::new(EmulatorChannel::new(Box::new(receiver)))),
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
    info!(
        "[{}:{}] {} buffer created",
        std::file!(),
        std::line!(),
        CONFIG.comm_type
    );
    let mut max_devices = 0;
    if let cudaError_t::cudaSuccess =
        unsafe { cudaGetDeviceCount(&mut max_devices as *mut ::std::os::raw::c_int) }
    {
        info!(
            "[{}:{}] found {} cuda devices",
            std::file!(),
            std::line!(),
            max_devices
        );
    } else {
        error!(
            "[{}:{}] failed to find cuda devices",
            std::file!(),
            std::line!()
        );
        panic!();
    }
    if let cudaError_t::cudaSuccess = unsafe { cudaSetDevice(0) } {
        info!("[{}:{}] cuda device set to 0", std::file!(), std::line!());
    } else {
        error!(
            "[{}:{}] failed to set cuda device",
            std::file!(),
            std::line!()
        );
        panic!();
    }
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
