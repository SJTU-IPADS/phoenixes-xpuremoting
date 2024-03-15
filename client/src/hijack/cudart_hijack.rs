#![allow(non_snake_case)]
use super::*;

gen_hijack!(
    0,
    "cudaGetDevice",
    "cudaError_t",
    "*mut ::std::os::raw::c_int"
);
gen_hijack!(1, "cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_hijack!(
    2,
    "cudaGetDeviceCount",
    "cudaError_t",
    "*mut ::std::os::raw::c_int"
);
gen_hijack!(3, "cudaGetLastError", "cudaError_t");
gen_hijack!(4, "cudaPeekAtLastError", "cudaError_t");
gen_hijack!(5, "cudaStreamSynchronize", "cudaError_t", "cudaStream_t");
gen_hijack!(6, "cudaMalloc", "cudaError_t", "*mut MemPtr", "usize");
// gen_hijack!(
//     7,
//     "cudaMemcpy",
//     "cudaError_t",
//     "MemPtr",
//     "MemPtr",
//     "usize",
//     "cudaMemcpyKind"
// );
gen_hijack!(8, "cudaFree", "cudaError_t", "MemPtr");
gen_hijack!(
    9,
    "cudaStreamIsCapturing",
    "cudaError_t",
    "cudaStream_t",
    "*mut cudaStreamCaptureStatus"
);
gen_hijack!(
    10,
    "cudaGetDeviceProperties",
    "cudaError_t",
    "*mut cudaDeviceProp",
    "::std::os::raw::c_int"
);
// gen_hijack!(
//     100,
//     "__cudaRegisterFatBinary",
//     "MemPtr",
//     "*const ::std::os::raw::c_void"
// );
// gen_hijack!(
//     101,
//     "__cudaUnregisterFatBinary",
//     "null",
//     "MemPtr"
// );
// gen_hijack!(
//     102,
//     "__cudaRegisterFunction",
//     "null",
//     "MemPtr",
//     "MemPtr",
//     "*mut ::std::os::raw::c_char",
//     "*const ::std::os::raw::c_char",
//     "::std::os::raw::c_int",
//     "MemPtr",
//     "MemPtr",
//     "MemPtr",
//     "MemPtr",
//     "MemPtr"
// );
// gen_hijack!(
//     103,
//     "__cudaRegisterVar",
//     "null",
//     "MemPtr",
//     "MemPtr",
//     "*const ::std::os::raw::c_char",
//     "*const ::std::os::raw::c_char",
//     "::std::os::raw::c_int",
//     "usize",
//     "::std::os::raw::c_int",
//     "::std::os::raw::c_int"
// );
// gen_hijack!(
//     200,
//     "cudaLaunchKernel",
//     "cudaError_t",
//     "*const ::std::os::raw::c_void",
//     "dim3",
//     "dim3",
//     "*mut *mut ::std::os::raw::c_void",
//     "usize",
//     "cudaStream_t"
// );