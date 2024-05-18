extern crate network;

use network::{
    ringbufferchannel::{ChannelBufferManager, LocalChannelBufferManager, RingBuffer},
    CommChannel, RawMemory, RawMemoryMut,
};

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use std::boxed::Box;

pub struct ConsumerManager {
    buf: *mut u8,
    capacity: usize,
}

impl ChannelBufferManager for ConsumerManager {
    fn get_managed_memory(&self) -> (*mut u8, usize) {
        (self.buf, self.capacity)
    }

    fn read_at(&self, offset: usize, dst: *mut u8, count: usize) -> usize {
        unsafe {
            std::ptr::copy_nonoverlapping(self.buf.add(offset) as _, dst, count);
        }
        count
    }

    fn write_at(&self, offset: usize, src: *const u8, count: usize) -> usize {
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.buf.add(offset) as _, count);
        }
        count
    }
}

impl ConsumerManager {
    pub fn new(producer: &LocalChannelBufferManager) -> Self {
        let (buf, capacity) = producer.get_managed_memory();
        ConsumerManager { buf, capacity }
    }
}

unsafe impl Send for ConsumerManager {}

#[test]
fn test_ring_buffer_producer_consumer() {
    let c_shared_buffer =
        LocalChannelBufferManager::new(1024 + network::ringbufferchannel::channel::META_AREA);
    let p_shared_buffer = ConsumerManager::new(&c_shared_buffer);

    let barrier = Arc::new(Barrier::new(2)); // Set up a barrier for 2 threads
    let producer_barrier = barrier.clone();
    let consumer_barrier = barrier.clone();

    let test_iters = 1000;

    // Producer thread
    let producer = thread::spawn(move || {
        let mut producer_ring_buffer = RingBuffer::new(Box::new(p_shared_buffer));
        producer_barrier.wait(); // Wait for both threads to be ready

        for i in 0..test_iters {
            let data = [(i % 256) as u8; 10]; // Simplified data to send
            let send_memory = RawMemory::new(&data, data.len());
            producer_ring_buffer.put_bytes(&send_memory).unwrap();
        }

        println!("Producer done");
    });

    // Consumer thread
    let consumer = thread::spawn(move || {
        let mut consumer_ring_buffer = RingBuffer::new(Box::new(c_shared_buffer));
        consumer_barrier.wait(); // Wait for both threads to be ready

        let mut received = 0;
        let mut buffer = [0u8; 10];

        while received < test_iters {
            let len = buffer.len();
            let mut recv_memory = RawMemoryMut::new(&mut buffer, len);
            match consumer_ring_buffer.get_bytes(&mut recv_memory) {
                Ok(size) => {
                    for i in 0..size {
                        assert_eq!(buffer[i], (received % 256) as u8);
                    }

                    received += 1;
                }
                Err(_) => thread::sleep(Duration::from_millis(10)), // Wait if buffer is empty
            }
        }
    });

    // Note: producer must be joined later, since the consumer will reuse the buffer
    consumer.join().unwrap();
    producer.join().unwrap();
}
