use super::*;
pub use crate::types::nvml::*;
include!("bindings/funcs/nvml.rs");

impl nvmlReturn_t {
    pub fn from(value: i32) -> Self {
        match value {
            0 => nvmlReturn_t::NVML_SUCCESS,
            1 => nvmlReturn_t::NVML_ERROR_UNINITIALIZED,
            2 => nvmlReturn_t::NVML_ERROR_INVALID_ARGUMENT,
            3 => nvmlReturn_t::NVML_ERROR_NOT_SUPPORTED,
            4 => nvmlReturn_t::NVML_ERROR_NO_PERMISSION,
            5 => nvmlReturn_t::NVML_ERROR_ALREADY_INITIALIZED,
            6 => nvmlReturn_t::NVML_ERROR_NOT_FOUND,
            7 => nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE,
            8 => nvmlReturn_t::NVML_ERROR_INSUFFICIENT_POWER,
            9 => nvmlReturn_t::NVML_ERROR_DRIVER_NOT_LOADED,
            10 => nvmlReturn_t::NVML_ERROR_TIMEOUT,
            11 => nvmlReturn_t::NVML_ERROR_IRQ_ISSUE,
            12 => nvmlReturn_t::NVML_ERROR_LIBRARY_NOT_FOUND,
            13 => nvmlReturn_t::NVML_ERROR_FUNCTION_NOT_FOUND,
            14 => nvmlReturn_t::NVML_ERROR_CORRUPTED_INFOROM,
            15 => nvmlReturn_t::NVML_ERROR_GPU_IS_LOST,
            16 => nvmlReturn_t::NVML_ERROR_RESET_REQUIRED,
            17 => nvmlReturn_t::NVML_ERROR_OPERATING_SYSTEM,
            18 => nvmlReturn_t::NVML_ERROR_LIB_RM_VERSION_MISMATCH,
            19 => nvmlReturn_t::NVML_ERROR_IN_USE,
            20 => nvmlReturn_t::NVML_ERROR_MEMORY,
            21 => nvmlReturn_t::NVML_ERROR_NO_DATA,
            22 => nvmlReturn_t::NVML_ERROR_VGPU_ECC_NOT_SUPPORTED,
            23 => nvmlReturn_t::NVML_ERROR_INSUFFICIENT_RESOURCES,
            999 => nvmlReturn_t::NVML_ERROR_UNKNOWN,
            _ => panic!(format!("unsupported nvmlReturn_t {}", value))
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::FromPrimitive;
    use network::{
        ringbufferchannel::{META_AREA, LocalChannel},
        Channel,
    };

    #[test]
    fn test_num_derive() {
        let x: u32 = nvmlReturn_t::NVML_SUCCESS as u32;
        assert_eq!(x, 0);
        match nvmlReturn_t::from_u32(1) {
            Some(v) => assert_eq!(v, nvmlReturn_t::NVML_ERROR_UNINITIALIZED),
            None => panic!("failed to convert from u32"),
        }
    }

    #[test]
    fn test_nvmlReturn_t_io() {
        let mut buffer: Channel =
            Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = nvmlReturn_t::NVML_ERROR_UNINITIALIZED;
        let mut b = nvmlReturn_t::NVML_SUCCESS;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }
}
