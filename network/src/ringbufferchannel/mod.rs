pub mod channel;
use std::ptr::NonNull;

pub use channel::RingBuffer;
pub mod test;

// Only implemented in Linux now
#[cfg(target_os = "linux")]
pub mod shm;

pub mod utils;

/// A ring buffer can use arbitrary memory for its channel 
/// 
/// It will manage the following:
/// - The buffer memory allocation and 
/// - The buffer memory deallocation
pub trait ChannelBufferManager { 
    fn get_managed_memory(&self) -> (*mut u8, usize);
}

/// A simple local channel buffer manager 
pub struct LocalChannelBufferManager {
    buffer: NonNull<u8>, 
    size : usize
}

unsafe impl Send for LocalChannelBufferManager {}

impl LocalChannelBufferManager {
    pub fn new(size: usize) -> LocalChannelBufferManager {
        LocalChannelBufferManager {
            buffer: utils::allocate_cache_line_aligned(size, 64),
            size : size
        }
    }
}

impl ChannelBufferManager for LocalChannelBufferManager {
    fn get_managed_memory(&self) -> (*mut u8, usize) {
        (self.buffer.as_ptr(), self.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_channel_buffer_manager() {
        let size = 64;
        let manager = LocalChannelBufferManager::new(size);
        let (ptr, len) = manager.get_managed_memory();        
        assert!(utils::is_cache_line_aligned(ptr));

        assert_eq!(len, size);
    }
}


