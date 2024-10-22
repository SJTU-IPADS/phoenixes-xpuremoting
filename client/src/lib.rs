#[macro_use]
extern crate lazy_static;

extern crate cudasys;
extern crate network;

extern crate log;
#[allow(unused_imports)]
use log::{debug, error, info, log_enabled, Level};

#[allow(unused_imports)]
use network::{
    ringbufferchannel::{EmulatorChannel, RDMAChannel, SHMChannel},
    type_impl::MemPtr,
    Channel, CommChannel, CommChannelInner, Transportable, CONFIG,
};

extern crate codegen;
use codegen::gen_hijack;
#[cfg(feature = "async_api")]
use codegen::gen_hijack_async;
#[cfg(feature = "local")]
use codegen::gen_hijack_local;
use codegen::gen_unimplement;

pub mod hijack;
pub use hijack::*;

pub mod elf;
use elf::interfaces::{fat_header, kernel_info_t};
use elf::ElfController;

pub mod dl;
pub use dl::*;

use std::boxed::Box;
use std::collections::HashMap;
use std::{
    sync::Mutex,
};

extern {
    fn pos_create_agent() -> *mut std::ffi::c_void;
    fn pos_destory_agent(pos_agent: *mut std::ffi::c_void) -> i32;
}

lazy_static! {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the client side, the sender's name is ctos_channel_name,
    // receiver's name is stoc_channel_name.
    static ref CHANNEL_SENDER: Mutex<Channel> = {
        let c: Box<dyn CommChannelInner> = match CONFIG.comm_type.as_str() {
            #[cfg(feature = "shm")]
            "shm" => {
                Box::new(SHMChannel::new_client(&CONFIG.ctos_channel_name, CONFIG.buf_size).unwrap())
            },
            #[cfg(feature = "rdma")]
            "rdma" => {
                // client side sender should connect to server's receiver socket.
                Box::new(RDMAChannel::new_client(&CONFIG.ctos_channel_name, CONFIG.buf_size, CONFIG.receiver_socket.parse().unwrap(), 1).unwrap())
            }
            &_ => panic!("Unsupported communication type in config"),
        };
        unsafe{ pos_create_agent() };
        if cfg!(feature = "emulator") {
            Mutex::new(Channel::new(Box::new(EmulatorChannel::new(c))))
        } else {
            Mutex::new(Channel::new(c))
        }
    };
    static ref CHANNEL_RECEIVER: Mutex<Channel> = {
        let c: Box<dyn CommChannelInner> = match CONFIG.comm_type.as_str() {
            #[cfg(feature = "shm")]
            "shm" => {
                Box::new(SHMChannel::new_client(&CONFIG.stoc_channel_name, CONFIG.buf_size).unwrap())
            }
            #[cfg(feature = "rdma")]
            "rdma" => {
                // client side receiver should connect to server's sender socket.
                Box::new(RDMAChannel::new_client(&CONFIG.stoc_channel_name, CONFIG.buf_size, CONFIG.sender_socket.parse().unwrap(), 1).unwrap())
            }
            &_ => panic!("Unsupported communication type in config"),
        };
        if cfg!(feature = "emulator") {
            Mutex::new(Channel::new(Box::new(EmulatorChannel::new(c))))
        } else {
            Mutex::new(Channel::new(c))
        }
    };

    static ref ELF_CONTROLLER: ElfController = ElfController::new();
    static ref ENABLE_LOG: bool = {
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "debug");
        }
        env_logger::init();
        true
    };

    static ref RESOURCE_IDX: Mutex<usize> = Mutex::new(0);

    static ref LOCAL_INFO: Mutex<HashMap<usize, usize>> = Mutex::new(HashMap::new());
}

#[cfg(feature = "local")]
fn add_local_info(proc_id: usize, info: usize) {
    LOCAL_INFO.lock().unwrap().insert(proc_id, info);
}

#[cfg(feature = "local")]
fn get_local_info(proc_id: usize) -> Option<usize> {
    LOCAL_INFO.lock().unwrap().get(&proc_id).cloned()
}

// #[ctor]
// fn init() {
//     core_affinity::set_for_current(1);
// }