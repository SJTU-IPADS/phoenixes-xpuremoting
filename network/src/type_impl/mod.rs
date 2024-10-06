use crate::{CommChannel, CommChannelError, RawMemory, RawMemoryMut, Transportable, MemorySize};

macro_rules! impl_transportable {
    ($($t:ty),*) => {
        $(
            impl Transportable for $t {
                fn emulate_send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
                    let memory = RawMemory::new(self, std::mem::size_of::<Self>());
                    match channel.put_bytes(&memory)? == std::mem::size_of::<Self>() {
                        true => {
                            Ok(())},
                        false => Err(CommChannelError::IoError),
                    }
                }

                fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
                    let memory = RawMemory::new(self, std::mem::size_of::<Self>());
                    match channel.put_bytes(&memory)? == std::mem::size_of::<Self>() {
                        true => {
                            Ok(())},
                        false => Err(CommChannelError::IoError),
                    }
                }

                fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError> {
                    let mut memory = RawMemoryMut::new(self, std::mem::size_of::<Self>());
                    match channel.get_bytes(&mut memory)? == std::mem::size_of::<Self>() {
                        true => {Ok(())},
                        false => Err(CommChannelError::IoError),
                    }
                }
            }

            impl MemorySize for $t {
                fn mem_size(&self) -> usize {
                    std::mem::size_of::<Self>()
                }
            }
        )*
    };
}

impl_transportable!(
    u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64, bool, char
);

impl Transportable for () {
    fn emulate_send<T: CommChannel>(&self, _channel: &mut T) -> Result<(), CommChannelError> {
        Ok(())
    }

    fn send<T: CommChannel>(&self, _channel: &mut T) -> Result<(), CommChannelError> {
        Ok(())
    }

    fn recv<T: CommChannel>(&mut self, _channel: &mut T) -> Result<(), CommChannelError> {
        Ok(())
    }
}

/// a pointer type, we just need to use usize to represent it
/// the raw type `*mut void` is hard to handle:(.
///
/// IMPORTANT on replacing `*mut *mut` like parameters in memory operations.
pub type MemPtr = usize;

impl<S> Transportable for [S] {
    fn emulate_send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
        let len = self.len() * std::mem::size_of::<S>();
        let memory = RawMemory::from_ptr(self.as_ptr() as *const u8, len);
        match channel.put_bytes(&memory)? == len {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }

    fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
        let len = self.len() * std::mem::size_of::<S>();
        let memory = RawMemory::from_ptr(self.as_ptr() as *const u8, len);
        match channel.put_bytes(&memory)? == len {
            true => {
                Ok(())
            }
            false => Err(CommChannelError::IoError),
        }
    }

    fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError> {
        let len = self.len() * std::mem::size_of::<S>();
        let mut memory = RawMemoryMut::from_ptr(self.as_mut_ptr() as *mut u8, len);
        match channel.get_bytes(&mut memory)? == len {
            true => Ok(()),
            false => Err(CommChannelError::IoError),
        }
    }

}

impl MemorySize for *mut std::ffi::c_void {
    fn mem_size(&self) -> usize {
        std::mem::size_of::<usize>()
    }
}
impl MemorySize for *const std::ffi::c_void {
    fn mem_size(&self) -> usize {
        std::mem::size_of::<usize>()
    }
}
impl MemorySize for *const f32 {
    fn mem_size(&self) -> usize {
        std::mem::size_of::<usize>()
    }
}
impl MemorySize for *const f64 {
    fn mem_size(&self) -> usize {
        std::mem::size_of::<usize>()
    }
}

impl<S> MemorySize for [S] {
    fn mem_size(&self) -> usize {
        self.len() * std::mem::size_of::<S>()        
    }
}

impl<S> Transportable for Vec<S>
where
    S: Default + Clone,
{
    fn emulate_send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
        let len: usize = self.len();
        len.emulate_send(channel)?;
        self.as_slice().emulate_send(channel)
    }

    fn send<T: CommChannel>(&self, channel: &mut T) -> Result<(), CommChannelError> {
        let len: usize = self.len();
        len.send(channel)?;
        self.as_slice().send(channel)
    }

    fn recv<T: CommChannel>(&mut self, channel: &mut T) -> Result<(), CommChannelError> {
        let mut len: usize = 0;
        len.recv(channel)?;
        self.resize(len, S::default());
        self.as_mut_slice().recv(channel)
    }

}

impl<S> MemorySize for Vec<S>
where S: Default + Clone
{
    fn mem_size(&self) -> usize {
        self.len() * std::mem::size_of::<S>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ringbufferchannel::{LocalChannel, META_AREA},
        Channel,
    };

    /// Test bool Transportable impl
    #[test]
    fn test_bool_io() {
        let mut channel = Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = true;
        let mut b = false;
        a.send(&mut channel).unwrap();
        b.recv(&mut channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test i32 Transportable impl
    #[test]
    fn test_i32_io() {
        let mut channel = Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = 123;
        let mut b = 0;
        a.send(&mut channel).unwrap();
        b.recv(&mut channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test [u8] Transportable impl
    #[test]
    fn test_u8_array_io() {
        let mut channel = Channel::new(Box::new(LocalChannel::new(10 + META_AREA)));
        let a = [1u8, 2, 3, 4, 5];
        let mut b = [0u8; 5];
        a.send(&mut channel).unwrap();
        b.recv(&mut channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test [i32] Transportable impl
    #[test]
    fn test_i32_array_io() {
        let mut channel = Channel::new(Box::new(LocalChannel::new(50 + META_AREA)));
        let a = [1i32, 2, 3, 4, 5];
        let mut b = [0i32; 5];
        a.send(&mut channel).unwrap();
        b.recv(&mut channel).unwrap();
        assert_eq!(a, b);
    }

    /// Test Vec<i32> Transportable impl
    #[test]
    fn test_vec_io() {
        let mut channel = Channel::new(Box::new(LocalChannel::new(50 + META_AREA)));
        let a = vec![1, 2, 3, 4, 5];
        let mut b = vec![0; 5];
        a.send(&mut channel).unwrap();
        b.recv(&mut channel).unwrap();
        assert_eq!(a, b);
    }
}
