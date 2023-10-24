#ifndef DEVICE_BUFFER_H
#define DEVICE_BUFFER_H

#define BUFFER_IO_TIMEOUT 30 // seconds

enum BufferPrivilege {
    BufferHost = 0,
    BufferGuest
};

// An io buffer abstraction for heterogenous device
class DeviceBuffer {
public:
    DeviceBuffer(BufferPrivilege privilege)
        : privilege_(privilege) {};
    ~DeviceBuffer() {};

    // By default put `length` bytes from `src` to internal buffer,
    // and when buffer is full will call flush_out.
    virtual int putBytes(const char* src, int length) = 0;

    // By default get `length` bytes from internal buffer to `dst`,
    // and when buffer is empty will call fill_in.
    virtual int getBytes(char* dst, int length) = 0;

    // Send out all bytes in buffer.
    // User can call it to enforcing flush at once.
    virtual int FlushOut() = 0;

    // Pull as much bytes (no larger than buffer size) into buffer.
    virtual int FillIn() = 0;

protected:
    BufferPrivilege privilege_;
};

#endif // DEVICE_BUFFER_H
