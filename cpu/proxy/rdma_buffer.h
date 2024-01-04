#ifndef RDMA_BUFFER_H
#define RDMA_BUFFER_H

#include "device_buffer.h"
#include "rib/core/lib.hh"
#include <memory>

using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;

#define RDMA_BUFFER_SIZE (512*1024*1024)
#define RDMA_CONNECTION_PORT_STOC 8080
#define RDMA_NIC_IDX_STOC 0
#define RDMA_NIC_NAME_STOC 60
#define RDMA_MEM_NAME_STOC 60
#define RDMA_CONNECTION_PORT_CTOS 8180
#define RDMA_NIC_IDX_CTOS 0
#define RDMA_NIC_NAME_CTOS 80
#define RDMA_MEM_NAME_CTOS 80
#define RDMA_MAX_PENDING_NUM 16
#define RDMA_UNSIGNALED_BUFFER_NUM 2

// RDMA Buffer, based on rlibv2
class RDMABuffer final : public DeviceBuffer
{
public:
    RDMABuffer(BufferPrivilege privilege, usize port, std::string addr,
               uint64_t nic_idx, uint64_t nic_name, uint64_t mem_name,
               std::string qp_name, int buf_size);
    ~RDMABuffer();

    // public DeviceBuffer methods
    int putBytes(const char *src, int length) override;
    int getBytes(char *dst, int length) override;
    int FlushOut() override;
    int FillIn() override;

private:
    // rdma related
    Arc<class rdmaio::RNic> nic_;
    Arc<RMem> mem_;
    std::unique_ptr<RCtrl> ctrl_;        // host
    std::unique_ptr<ConnectManager> cm_; // guest
    Arc<RegHandler> mr_;
    Arc<rdmaio::qp::RC> qp_;
    std::string qp_name_;
    unsigned long qp_key_;
    int pending_num_;
    int unsignaled_buf_idx_;
    int unsignaled_buf_len_;

    // local ring buffer
    char *buf_;
    int buf_size_;
    int *buf_head_, *buf_tail_;
    int last_tail_; // guest

    void HostInit(usize port, uint64_t nic_idx, uint64_t nic_name,
                  uint64_t mem_name);
    void HostDestroy();
    void GuestInit(std::string addr, uint64_t nic_idx, uint64_t nic_name,
                   uint64_t mem_name, std::string qp_name);
    void GuestDestroy();
    void UpdateBufPtr();
    int RemoteRead(void *addr, u64 remote_addr, int length);
    int PollComp();
    int RemoteWrite(void *addr, u64 remote_addr, int length, int call_end);
    int WriteCapacity(int read_head);
    int ReadCapacity(int read_tail);
};

#endif // RDMA_BUFFER_H
