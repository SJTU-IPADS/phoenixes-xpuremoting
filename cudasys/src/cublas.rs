use super::*;
pub use crate::types::cublas::*;
include!("bindings/funcs/cublas.rs");

impl cublasStatus_t {
    pub fn from(value: i32) -> Self {
        match value {
            0 => cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            1 => cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED,
            3 => cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED,
            7 => cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
            8 => cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH,
            11 => cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR,
            13 => cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED,
            14 => cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR,
            15 => cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED,
            16 => cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR,
            _ => panic!(format!("unsupported cublasStatus_t {}", value))
        }
    }
}