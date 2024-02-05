use super::ChannelBufferManager;
use log::error;

use std::ffi::CString;
use std::io::{self, Result};
use std::os::unix::io::RawFd;

/// A shared memory channel buffer manager
pub struct SHMChannelBufferManager {
    shm_name: String,
    shm_len: usize,
    shm_ptr: *mut libc::c_void,
}

impl SHMChannelBufferManager {
    /// Create a new shared memory channel buffer manager for the server
    pub fn new_server(shm_name: &str, shm_len: usize) -> Result<Self> {
        Self::new_inner(
            shm_name,
            shm_len,
            libc::O_CREAT,
            (libc::S_IRUSR | libc::S_IWUSR) as i32,
        )
    }

    fn new_inner(shm_name: &str, shm_len: usize, oflag: i32, sflag: i32) -> Result<Self> {
        let fd: RawFd = unsafe { libc::shm_open(shm_name.as_ptr() as _, oflag, sflag) };

        if fd == -1 {
            error!("Error on shm_open for new_host");
            return Err(io::Error::from_raw_os_error(
                unsafe { *libc::__error() } as i32
            ));
        }

        if unsafe { libc::ftruncate(fd, shm_len as libc::off_t) } == -1 {
            error!("Error on ftruncate");
            unsafe { libc::shm_unlink(shm_name.as_ptr() as _) };
            return Err(io::Error::from_raw_os_error(
                unsafe { *libc::__error() } as i32
            ));
        }

        // map the shared memory to the process's address space
        let shm_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                shm_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if shm_ptr == libc::MAP_FAILED {
            error!("Error on mmap the SHM pointer");
            unsafe { libc::shm_unlink(shm_name.as_ptr() as _) };
            return Err(io::Error::from_raw_os_error(
                unsafe { *libc::__error() } as i32
            ));
        }

        Ok(Self {
            shm_name: String::from(shm_name),
            shm_len,
            shm_ptr: shm_ptr,
        })
    }
}

impl Drop for SHMChannelBufferManager {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.shm_ptr, self.shm_len);
            let shm_name_ = CString::new(self.shm_name.clone()).unwrap();
            libc::shm_unlink(shm_name_.as_ptr());
        }
    }
}

impl ChannelBufferManager for SHMChannelBufferManager {
    fn get_managed_memory(&self) -> (*mut u8, usize) {
        (self.shm_ptr as *mut u8, self.shm_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shm_channel_buffer_manager() {
        let shm_name = "t";
        let shm_len = 64;
        let manager = SHMChannelBufferManager::new_server(shm_name, shm_len).unwrap();
        assert_eq!(manager.shm_len, shm_len);
    }
}
