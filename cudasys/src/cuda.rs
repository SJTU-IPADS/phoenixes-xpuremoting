use super::*;
pub use crate::types::cuda::*;
include!("bindings/funcs/cuda.rs");

impl CUresult {
    pub fn from(value: i32) -> Self {
        match value {
            0 => CUresult::CUDA_SUCCESS,
            1 => CUresult::CUDA_ERROR_INVALID_VALUE,
            2 => CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            3 => CUresult::CUDA_ERROR_NOT_INITIALIZED,
            4 => CUresult::CUDA_ERROR_DEINITIALIZED,
            5 => CUresult::CUDA_ERROR_PROFILER_DISABLED,
            6 => CUresult::CUDA_ERROR_PROFILER_NOT_INITIALIZED,
            7 => CUresult::CUDA_ERROR_PROFILER_ALREADY_STARTED,
            8 => CUresult::CUDA_ERROR_PROFILER_ALREADY_STOPPED,
            34 => CUresult::CUDA_ERROR_STUB_LIBRARY,
            100 => CUresult::CUDA_ERROR_NO_DEVICE,
            101 => CUresult::CUDA_ERROR_INVALID_DEVICE,
            102 => CUresult::CUDA_ERROR_DEVICE_NOT_LICENSED,
            200 => CUresult::CUDA_ERROR_INVALID_IMAGE,
            201 => CUresult::CUDA_ERROR_INVALID_CONTEXT,
            202 => CUresult::CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
            205 => CUresult::CUDA_ERROR_MAP_FAILED,
            206 => CUresult::CUDA_ERROR_UNMAP_FAILED,
            207 => CUresult::CUDA_ERROR_ARRAY_IS_MAPPED,
            208 => CUresult::CUDA_ERROR_ALREADY_MAPPED,
            209 => CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU,
            210 => CUresult::CUDA_ERROR_ALREADY_ACQUIRED,
            211 => CUresult::CUDA_ERROR_NOT_MAPPED,
            212 => CUresult::CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
            213 => CUresult::CUDA_ERROR_NOT_MAPPED_AS_POINTER,
            214 => CUresult::CUDA_ERROR_ECC_UNCORRECTABLE,
            215 => CUresult::CUDA_ERROR_UNSUPPORTED_LIMIT,
            216 => CUresult::CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
            217 => CUresult::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
            218 => CUresult::CUDA_ERROR_INVALID_PTX,
            219 => CUresult::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
            220 => CUresult::CUDA_ERROR_NVLINK_UNCORRECTABLE,
            221 => CUresult::CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
            222 => CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
            223 => CUresult::CUDA_ERROR_JIT_COMPILATION_DISABLED,
            300 => CUresult::CUDA_ERROR_INVALID_SOURCE,
            301 => CUresult::CUDA_ERROR_FILE_NOT_FOUND,
            302 => CUresult::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
            303 => CUresult::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
            304 => CUresult::CUDA_ERROR_OPERATING_SYSTEM,
            400 => CUresult::CUDA_ERROR_INVALID_HANDLE,
            401 => CUresult::CUDA_ERROR_ILLEGAL_STATE,
            500 => CUresult::CUDA_ERROR_NOT_FOUND,
            600 => CUresult::CUDA_ERROR_NOT_READY,
            700 => CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            701 => CUresult::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
            702 => CUresult::CUDA_ERROR_LAUNCH_TIMEOUT,
            703 => CUresult::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
            704 => CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
            705 => CUresult::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
            708 => CUresult::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
            709 => CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED,
            710 => CUresult::CUDA_ERROR_ASSERT,
            711 => CUresult::CUDA_ERROR_TOO_MANY_PEERS,
            712 => CUresult::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
            713 => CUresult::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
            714 => CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR,
            715 => CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION,
            716 => CUresult::CUDA_ERROR_MISALIGNED_ADDRESS,
            717 => CUresult::CUDA_ERROR_INVALID_ADDRESS_SPACE,
            718 => CUresult::CUDA_ERROR_INVALID_PC,
            719 => CUresult::CUDA_ERROR_LAUNCH_FAILED,
            720 => CUresult::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
            800 => CUresult::CUDA_ERROR_NOT_PERMITTED,
            801 => CUresult::CUDA_ERROR_NOT_SUPPORTED,
            802 => CUresult::CUDA_ERROR_SYSTEM_NOT_READY,
            803 => CUresult::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
            804 => CUresult::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE,
            900 => CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
            901 => CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
            902 => CUresult::CUDA_ERROR_STREAM_CAPTURE_MERGE,
            903 => CUresult::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED,
            904 => CUresult::CUDA_ERROR_STREAM_CAPTURE_UNJOINED,
            905 => CUresult::CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
            906 => CUresult::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
            907 => CUresult::CUDA_ERROR_CAPTURED_EVENT,
            908 => CUresult::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD,
            909 => CUresult::CUDA_ERROR_TIMEOUT,
            910 => CUresult::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE,
            999 => CUresult::CUDA_ERROR_UNKNOWN,
            _ => panic!(format!("unsupported CUresult {}", value))
        }
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    use network::{
        ringbufferchannel::{META_AREA, LocalChannel},
        Channel,
    };
    use std::boxed::Box;

    #[test]
    fn test_CUresult_io() {
        let mut buffer: Channel =
            Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = CUresult::CUDA_ERROR_ALREADY_ACQUIRED;
        let mut b = CUresult::CUDA_SUCCESS;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }
}
