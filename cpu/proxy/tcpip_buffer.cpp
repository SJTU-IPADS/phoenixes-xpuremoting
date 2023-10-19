#include "tcpip_buffer.h"
#include <chrono>
#include <cstring>
#include <iostream>

TcpipBuffer::TcpipBuffer(BufferPrivilege privilege, struct sockaddr* addr, socklen_t* addr_len, int buf_size)
    : DeviceBuffer(privilege)
{
    buf_size_ = buf_size;
    buf_ = new char[buf_size_];
    if (buf_ == NULL) {
        std::cerr << "Error on buf creation" << std::endl;
        exit(1);
    }
    buf_head_ = buf_tail_ = 0;

    if (privilege_ == BufferHost) {
        HostInit(addr, addr_len);
    } else {
        GuestInit(addr, addr_len);
    }
}

TcpipBuffer::~TcpipBuffer()
{
    delete[] buf_;
    if (privilege_ == BufferHost) {
        HostDestroy();
    } else {
        GuestDestroy();
    }
}

int TcpipBuffer::putBytes(const char* src, int length)
{
    int current, len = length;
    // std::cout << "putBytes, buf_head_: " << buf_head_ << ", buf_tail_: " << buf_tail_ << std::endl;
    while (len > 0) {
        if (buf_tail_ == buf_size_) {
            int ret = FlushOut();
            if (ret < 0) {
                return -1;
            }
        }
        current = std::min(len, buf_size_ - buf_tail_);
        memcpy(&buf_[buf_tail_], src, current);
        buf_tail_ = buf_tail_ + current;
        src += current;
        len -= current;
    }
    // std::cout << "putBytes, buf_head_: " << buf_head_ << ", buf_tail_: " << buf_tail_ << std::endl;
    return length - len;
}

int TcpipBuffer::FlushOut()
{
    auto start = std::chrono::system_clock::now();
    while (buf_head_ < buf_tail_) {
        int ret = write(fd_, &buf_[buf_head_], buf_tail_ - buf_head_);
        if (ret < 0) {
            std::cerr << "send failed" << std::endl;
            return -1;
        }
        buf_head_ += ret;
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        if (duration.count() > BUFFER_IO_TIMEOUT) {
            std::cerr << "timeout in " << __func__ << " in " << __FILE__ << ":" << __LINE__ << std::endl;
            return -1;
        }
    }
    buf_head_ = buf_tail_ = 0;
    // std::cout << "FlushOut, buf_head_: " << buf_head_ << ", buf_tail_: " << buf_tail_ << std::endl;
    return 0;
}

int TcpipBuffer::getBytes(char* dst, int length)
{
    // std::cout << "getBytes, buf_head_: " << buf_head_ << ", buf_tail_: " << buf_tail_ << std::endl;
    int current, len = length;
    while (len > 0) {
        if (buf_head_ == buf_tail_) {
            int ret = FillIn();
            if (ret < 0) {
                return -1;
            }
        }
        current = std::min(len, buf_tail_ - buf_head_);
        memcpy(dst, &buf_[buf_head_], current);
        buf_head_ += current;
        dst += current;
        len -= current;
    }
    // std::cout << "getBytes, buf_head_: " << buf_head_ << ", buf_tail_: " << buf_tail_ << std::endl;
    return length - len;
}

int TcpipBuffer::FillIn()
{
    auto start = std::chrono::system_clock::now();
    while (buf_head_ == buf_tail_) {
        int ret = read(fd_, buf_, buf_size_);
        if (ret < 0) {
            std::cerr << "recv failed" << std::endl;
            return -1;
        }
        buf_head_ = 0, buf_tail_ = ret;
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        if (duration.count() > BUFFER_IO_TIMEOUT) {
            std::cerr << "timeout in " << __func__ << " in " << __FILE__ << ":" << __LINE__ << std::endl;
            return -1;
        }
    }
    // std::cout << "FillIn, buf_head_: " << buf_head_ << ", buf_tail_: " << buf_tail_ << std::endl;
    return 0;
}

void TcpipBuffer::HostInit(struct sockaddr* addr, socklen_t* addr_len)
{
    int opt = 1;

    if ((sock_fd_ = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "socket creation failed" << std::endl;
        exit(1);
    }

    if (setsockopt(sock_fd_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        std::cerr << "setsockopt failed" << std::endl;
        exit(1);
    }

    struct timeval time_out;
    time_out.tv_sec = BUFFER_IO_TIMEOUT;
    time_out.tv_usec = 0;
    if (setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &time_out, sizeof(time_out))) {
        std::cerr << "setsockopt failed" << std::endl;
        exit(1);
    }
    if (setsockopt(sock_fd_, SOL_SOCKET, SO_SNDTIMEO, &time_out, sizeof(time_out))) {
        std::cerr << "setsockopt failed" << std::endl;
        exit(1);
    }

    if (bind(sock_fd_, addr, *addr_len) < 0) {
        std::cerr << "bind failed" << std::endl;
        exit(1);
    }

    if (listen(sock_fd_, 1) < 0) {
        std::cerr << "listen failed" << std::endl;
        exit(1);
    }

    if ((fd_ = accept(sock_fd_, addr, addr_len)) < 0) {
        std::cerr << "accept failed" << std::endl;
        exit(1);
    }
}

void TcpipBuffer::HostDestroy()
{
    close(fd_);
    shutdown(sock_fd_, SHUT_RDWR);
}

void TcpipBuffer::GuestInit(struct sockaddr* addr, socklen_t* addr_len)
{
    if ((sock_fd_ = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "socket creation failed" << std::endl;
        exit(1);
    }

    struct timeval time_out;
    time_out.tv_sec = BUFFER_IO_TIMEOUT;
    time_out.tv_usec = 0;
    if (setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &time_out, sizeof(time_out))) {
        std::cerr << "setsockopt failed" << std::endl;
        exit(1);
    }
    if (setsockopt(sock_fd_, SOL_SOCKET, SO_SNDTIMEO, &time_out, sizeof(time_out))) {
        std::cerr << "setsockopt failed" << std::endl;
        exit(1);
    }

    auto start = std::chrono::system_clock::now();
    while (connect(sock_fd_, addr, *addr_len) < 0) {
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        if (duration.count() > BUFFER_IO_TIMEOUT) {
            std::cerr << "timeout in " << __func__ << " in " << __FILE__ << ":" << __LINE__ << std::endl;
            exit(1);
        }
    }

    fd_ = sock_fd_;
}

void TcpipBuffer::GuestDestroy()
{
    close(fd_);
}
