#ifndef SHM_BUFFER_H
#define SHM_BUFFER_H

#include "device_buffer.h"

#define SHM_BUFFER_SIZE 10485752
#define SHM_NAME_STOC "/stoc"
#define SHM_NAME_CTOS "/ctos"

// Shared Memory Buffer, should act like a ring buffer.
class ShmBuffer final : public DeviceBuffer {
public:
    ShmBuffer(BufferPrivilege privilege, const char* shm_name, int buf_size);
    ~ShmBuffer();
    const char* getShmName();
    int getShmLen();

    // public DeviceBuffer methods
    int putBytes(const char* src, int length) override;
    int getBytes(char* dst, int length) override;
    int FlushOut() override;
    int FillIn() override;

private:
    // share memory related
    char* shm_name_;
    int shm_len_;
    void* shm_ptr_;

    // ring buffer
    char* buf_;
    int buf_size_;
    int *buf_head_, *buf_tail_;

    void HostInit();
    void GuestInit();
    int WriteCapacity(int read_head);
    int ReadCapacity(int read_tail);
};

#endif // SHM_BUFFER_H
