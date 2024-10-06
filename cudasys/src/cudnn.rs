use super::*;
pub use crate::types::cudnn::*;
include!("bindings/funcs/cudnn.rs");

impl cudnnStatus_t {
    pub fn from(value: i32) -> Self {
        match value {
            0 => cudnnStatus_t::CUDNN_STATUS_SUCCESS,
            1 => cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            2 => cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED,
            3 => cudnnStatus_t::CUDNN_STATUS_BAD_PARAM,
            4 => cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR,
            5 => cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE,
            6 => cudnnStatus_t::CUDNN_STATUS_ARCH_MISMATCH,
            7 => cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR,
            8 => cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED,
            9 => cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            10 => cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR,
            11 => cudnnStatus_t::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING,
            12 => cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS,
            13 => cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW,
            14 => cudnnStatus_t::CUDNN_STATUS_VERSION_MISMATCH,
            _ => panic!(format!("unsupported cudnnStatus_t {}", value))
        }
    }
}