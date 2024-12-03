#ifndef TCPIP_BUFFER_H
#define TCPIP_BUFFER_H

#include "device_buffer.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#define TCPIP_BUFFER_SIZE 10485752
#define TCPIP_PORT_STOC 8080
#define TCPIP_PORT_CTOS 8180

// TCP/IP Buffer, based on socket
class TcpipBuffer final : public DeviceBuffer {
public:
    TcpipBuffer(BufferPrivilege privilege, struct sockaddr* addr, socklen_t* addr_len, int buf_size);
    ~TcpipBuffer();

    // public DeviceBuffer methods
    int putBytes(const char* src, int length) override;
    int getBytes(char* dst, int length) override;
    int FlushOut() override;
    int FillIn() override;

private:
    // socket related
    int sock_fd_;

    // memory buuffer and io
    int fd_;
    char* buf_;
    int buf_size_;
    int buf_head_, buf_tail_;

    void HostInit(struct sockaddr* addr, socklen_t* addr_len);
    void HostDestroy();
    void GuestInit(struct sockaddr* addr, socklen_t* addr_len);
    void GuestDestroy();
};

#endif // TCPIP_BUFFER_H
