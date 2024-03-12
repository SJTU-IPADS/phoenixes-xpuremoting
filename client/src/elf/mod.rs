pub mod interfaces;
use self::interfaces::{fat_header, kernel_info_t, elf2_init, kernel_infos_free};

pub struct ElfController;

impl ElfController {
    pub fn new() -> Self {
        let result = unsafe { elf2_init() };
        if result != 0 {
            panic!("Failed to initialize elf.");
        }
        ElfController
    }

    // wrapper of interfaces
    pub fn add_kernel_host_func(func: *const ::std::os::raw::c_void, info: *mut kernel_info_t) {
        unsafe { interfaces::add_kernel_host_func(func, info) };
    }
    pub fn add_kernel_name(name: *const ::std::os::raw::c_char, info: *mut kernel_info_t) {
        unsafe { interfaces::add_kernel_name(name, info) };
    }
    pub fn elf2_get_fatbin_info(
        fatbin: *const fat_header,
        fatbin_mem: *mut *mut u8,
        fatbin_size: *mut usize,
    ) -> ::std::os::raw::c_int {
        unsafe { interfaces::elf2_get_fatbin_info(fatbin, fatbin_mem, fatbin_size) }
    }
    pub fn elf2_parameter_info(
        memory: *mut ::std::os::raw::c_void,
        memsize: usize,
    ) -> ::std::os::raw::c_int {
        unsafe { interfaces::elf2_parameter_info(memory, memsize) }
    }
    pub fn elf2_symbol_address(
        symbol: *const ::std::os::raw::c_char,
    ) -> *mut ::std::os::raw::c_void {
        unsafe { interfaces::elf2_symbol_address(symbol) }
    }
    pub fn find_kernel_host_func(func: *const ::std::os::raw::c_void) -> *mut kernel_info_t {
        unsafe { interfaces::find_kernel_host_func(func) }
    }
    pub fn find_kernel_name(name: *const ::std::os::raw::c_char) -> *mut kernel_info_t {
        unsafe { interfaces::find_kernel_name(name) }
    }
    pub fn utils_parameter_info(path: *mut ::std::os::raw::c_char) -> ::std::os::raw::c_int {
        unsafe { interfaces::utils_parameter_info(path) }
    }
    pub fn utils_search_info(kernelname: *const ::std::os::raw::c_char) -> *mut kernel_info_t {
        unsafe { interfaces::utils_search_info(kernelname) }
    }
}

impl Drop for ElfController {
    fn drop(&mut self) {
        unsafe { kernel_infos_free() };
    }
}
