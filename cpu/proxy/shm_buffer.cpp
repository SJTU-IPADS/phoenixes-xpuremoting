#include "shm_buffer.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>

ShmBuffer::ShmBuffer(BufferPrivilege privilege, const char* shm_name, int buf_size)
    : DeviceBuffer(privilege)
{
    shm_name_ = new char(strlen(shm_name) + 1);
    if (shm_name_ == NULL) {
        std::cerr << "Error on new" << std::endl;
        exit(1);
    }
    strcpy(shm_name_, shm_name);
    buf_size_ = buf_size;

    if (privilege_ == BufferHost) {
        HostInit();
    } else {
        GuestInit();
    }
    // std::cout << "size: " << buf_size_ << std::endl;
}

ShmBuffer::~ShmBuffer()
{
    munmap(shm_ptr_, shm_len_);
    shm_unlink(shm_name_);
    delete[] shm_name_;
}

const char* ShmBuffer::getShmName()
{
    return shm_name_;
}

int ShmBuffer::getShmLen()
{
    return shm_len_;
}

// the capacity to write once
int ShmBuffer::WriteCapacity(int read_head)
{
    if (read_head == 0)
        read_head = buf_size_;
    if (*buf_tail_ >= read_head)
        return buf_size_ - *buf_tail_;
    else
        return read_head - *buf_tail_ - 1;
}

int ShmBuffer::putBytes(const char* src, int length)
{
    int current, len = length, read_head;
    // std::cout << "putBytes, length:" << length << ", head: " << *buf_head_ << ", tail: " << *buf_tail_ << std::endl;
    while (len > 0) {
        read_head = *buf_head_;
        if ((*buf_tail_ + 1) % buf_size_ == read_head) {
            int ret = FlushOut();
            if (ret < 0) {
                return -1;
            }
        }
        current = std::min(WriteCapacity(read_head), len);
        memcpy(&buf_[*buf_tail_], src, current);
        // std::cout << ", next_tail: " << next_tail << std::endl;
        *buf_tail_ = (*buf_tail_ + current) % buf_size_;
        src += current;
        len -= current;
    }
    return length - len;
}

int ShmBuffer::FlushOut()
{
    int count = 0;
    auto start = std::chrono::system_clock::now();
    while ((*buf_tail_ + 1) % buf_size_ == *buf_head_) {
        count++;
        if (count == 1000000) {
            count = 0;
            auto end = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
            if (duration.count() > BUFFER_IO_TIMEOUT) {
                std::cerr << "timeout in " << __func__ << " in " << __FILE__ << ":" << __LINE__ << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

// the capacity to read once
int ShmBuffer::ReadCapacity(int read_tail)
{
    if (read_tail >= *buf_head_)
        return read_tail - *buf_head_;
    else
        return buf_size_ - *buf_head_;
}

int ShmBuffer::getBytes(char* dst, int length)
{
    int current, len = length, read_tail;
    // std::cout << "getBytes, head: " << *buf_head_ << ", tail: " << *buf_tail_ << std::endl;
    while (len > 0) {
        read_tail = *buf_tail_;
        if (*buf_head_ == read_tail) {
            int ret = FillIn();
            if (ret < 0) {
                return -1;
            }
        }
        current = std::min(ReadCapacity(read_tail), len);
        memcpy(dst, &buf_[*buf_head_], current);
        // std::cout << ", next_head: " << next_head << std::endl;
        *buf_head_ = (*buf_head_ + current) % buf_size_;
        dst += current;
        len -= current;
    }
    return length - len;
}

int ShmBuffer::FillIn()
{
    int count = 0;
    auto start = std::chrono::system_clock::now();
    while (*buf_head_ == *buf_tail_) {
        count++;
        if (count == 1000000) {
            count = 0;
            auto end = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
            if (duration.count() > BUFFER_IO_TIMEOUT) {
                std::cerr << "timeout in " << __func__ << " in " << __FILE__ << ":" << __LINE__ << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

void ShmBuffer::HostInit()
{
    int fd;

    fd = shm_open(shm_name_, O_CREAT | O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        std::cerr << "Error on shm_open" << std::endl;
        delete[] shm_name_;
        exit(1);
    }

    // two int for head and tail
    shm_len_ = buf_size_ + sizeof(int) * 2;
    if (ftruncate(fd, shm_len_) == -1) {
        std::cerr << "Error on ftruncate" << std::endl;
        shm_unlink(shm_name_);
        delete[] shm_name_;
        exit(1);
    }

    shm_ptr_ = mmap(NULL, shm_len_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << "Error on mmap" << std::endl;
        shm_unlink(shm_name_);
        delete[] shm_name_;
        exit(1);
    }

    buf_ = (char*)shm_ptr_;
    buf_head_ = (int*)(buf_ + buf_size_), buf_tail_ = buf_head_ + 1;
    *buf_head_ = *buf_tail_ = 0;
}

void ShmBuffer::GuestInit()
{
    int fd;

    fd = shm_open(shm_name_, O_RDWR, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        std::cerr << "Error on shm_open" << std::endl;
        delete[] shm_name_;
        exit(1);
    }

    // two int for head and tail
    shm_len_ = buf_size_ + sizeof(int) * 2;

    shm_ptr_ = mmap(NULL, shm_len_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << "Error on mmap" << std::endl;
        shm_unlink(shm_name_);
        delete[] shm_name_;
        exit(1);
    }

    buf_ = (char*)shm_ptr_;
    buf_head_ = (int*)(buf_ + buf_size_), buf_tail_ = buf_head_ + 1;
}
