#include "rdma_buffer.h"
#include "rib/core/rctrl.hh"
#include "rib/core/utils/logging.hh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>

// #define PRINT_LOG

RDMABuffer::RDMABuffer(BufferPrivilege privilege, usize port, std::string addr,
                       uint64_t nic_idx, uint64_t nic_name, uint64_t mem_name,
                       std::string qp_name, int buf_size)
    : DeviceBuffer(privilege)
{
    buf_size_ = buf_size;

    if (privilege_ == BufferHost) {
        HostInit(port, nic_idx, nic_name, mem_name);
    } else {
        GuestInit(addr, nic_idx, nic_name, mem_name, qp_name);
    }
    // std::cout << "size: " << buf_size_ << std::endl;
}

RDMABuffer::~RDMABuffer()
{
    if (privilege_ == BufferHost) {
        HostDestroy();
    } else {
        GuestDestroy();
    }
}

// the capacity to write once
int RDMABuffer::WriteCapacity(int read_head)
{
    if (read_head == 0)
        read_head = buf_size_;
    if (*buf_tail_ >= read_head)
        return buf_size_ - *buf_tail_;
    else
        return read_head - *buf_tail_ - 1;
}

int RDMABuffer::putBytes(const char *src, int length)
{
#ifdef PRINT_LOG
    printf("%s, head: %d, tail: %d, len: %d\n", __func__, *buf_head_,
           *buf_tail_, length);
    fflush(stdout);
#endif
    int current, len = length, read_head;
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
        *buf_tail_ = (*buf_tail_ + current) % buf_size_;
        src += current;
        len -= current;
    }
#ifdef PRINT_LOG
    printf("%s, head: %d, tail: %d\n", __func__, *buf_head_, *buf_tail_);
    fflush(stdout);
#endif
    return length - len;
}

int RDMABuffer::RemoteRead(void *addr, u64 remote_addr, int length)
{
    auto res_s = qp_->send_normal({ .op = IBV_WR_RDMA_READ,
                                    .flags = IBV_SEND_SIGNALED,
                                    .len = (rdmaio::u32)(length),
                                    .wr_id = 0 },
                                  { .local_addr =
                                        reinterpret_cast<RMem::raw_ptr_t>(addr),
                                    .remote_addr = remote_addr,
                                    .imm_data = 0 });
    RDMA_ASSERT(res_s == IOCode::Ok);
    auto res_p = qp_->wait_one_comp(BUFFER_IO_TIMEOUT * 1000000);
    RDMA_ASSERT(res_p == IOCode::Ok);
    if (res_p != IOCode::Ok) {
        if (res_p == IOCode::Timeout)
            std::cerr << "timeout in " << __func__ << " in " << __FILE__ << ":"
                      << __LINE__ << std::endl;
        return -1;
    }
    return 0;
}

int RDMABuffer::RemoteWrite(void *addr, u64 remote_addr, int length)
{
    auto res_s = qp_->send_normal({ .op = IBV_WR_RDMA_WRITE,
                                    .flags = IBV_SEND_SIGNALED,
                                    .len = (rdmaio::u32)(length),
                                    .wr_id = 0 },
                                  { .local_addr =
                                        reinterpret_cast<RMem::raw_ptr_t>(addr),
                                    .remote_addr = remote_addr,
                                    .imm_data = 0 });
    RDMA_ASSERT(res_s == IOCode::Ok);
    auto res_p = qp_->wait_one_comp(BUFFER_IO_TIMEOUT * 1000000);
    RDMA_ASSERT(res_p == IOCode::Ok);
    if (res_p != IOCode::Ok) {
        if (res_p == IOCode::Timeout)
            std::cerr << "timeout in " << __func__ << " in " << __FILE__ << ":"
                      << __LINE__ << std::endl;
        return -1;
    }
    return 0;
}

int RDMABuffer::FlushOut()
{
#ifdef PRINT_LOG
    printf("%s, head: %d, tail: %d\n", __func__, *buf_head_, *buf_tail_);
    fflush(stdout);
#endif
    auto start = std::chrono::system_clock::now();

    if (last_tail_ > *buf_tail_) {
        if (RemoteWrite(buf_ + last_tail_, last_tail_, buf_size_ - last_tail_) <
            0)
            return -1;
        last_tail_ = 0;
    }
    if (last_tail_ < *buf_tail_) {
        if (RemoteWrite(buf_ + last_tail_, last_tail_,
                        *buf_tail_ - last_tail_) < 0)
            return -1;
        last_tail_ = *buf_tail_;
    }
    if (RemoteWrite(buf_tail_, buf_size_ + sizeof(int), sizeof(int)) < 0)
        return -1;

    while ((*buf_tail_ + 1) % buf_size_ == *buf_head_) {
        auto end = std::chrono::system_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::seconds>(end - start);
        if (duration.count() > BUFFER_IO_TIMEOUT) {
            std::cerr << "timeout in " << __func__ << " in " << __FILE__ << ":"
                      << __LINE__ << std::endl;
            return -1;
        }

        if (RemoteRead(buf_head_, buf_size_, sizeof(int)) < 0)
            return -1;
    }
#ifdef PRINT_LOG
    printf("%s, head: %d, tail: %d\n", __func__, *buf_head_, *buf_tail_);
    fflush(stdout);
#endif
    return 0;
}

// the capacity to read once
int RDMABuffer::ReadCapacity(int read_tail)
{
    if (read_tail >= *buf_head_)
        return read_tail - *buf_head_;
    else
        return buf_size_ - *buf_head_;
}

int RDMABuffer::getBytes(char *dst, int length)
{
#ifdef PRINT_LOG
    printf("%s, head: %d, tail: %d, len: %d\n", __func__, *buf_head_,
           *buf_tail_, length);
    fflush(stdout);
#endif
    int current, len = length, read_tail;
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
        *buf_head_ = (*buf_head_ + current) % buf_size_;
        dst += current;
        len -= current;
    }
#ifdef PRINT_LOG
    printf("%s, head: %d, tail: %d\n", __func__, *buf_head_, *buf_tail_);
    fflush(stdout);
#endif
    return length - len;
}

int RDMABuffer::FillIn()
{
#ifdef PRINT_LOG
    printf("%s, head: %d, tail: %d\n", __func__, *buf_head_, *buf_tail_);
    fflush(stdout);
#endif
    int count = 0;
    auto start = std::chrono::system_clock::now();
    while (*buf_head_ == *buf_tail_) {
        count++;
        if (count == 1000000) {
            count = 0;
            auto end = std::chrono::system_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::seconds>(end - start);
            if (duration.count() > BUFFER_IO_TIMEOUT) {
                std::cerr << "timeout in " << __func__ << " in " << __FILE__
                          << ":" << __LINE__ << std::endl;
                return -1;
            }
        }
    }
#ifdef PRINT_LOG
    printf("%s, head: %d, tail: %d\n", __func__, *buf_head_, *buf_tail_);
    fflush(stdout);
#endif
    return 0;
}

void RDMABuffer::HostInit(usize port, uint64_t nic_idx, uint64_t nic_name,
                          uint64_t mem_name)
{
    ctrl_ = std::make_unique<RCtrl>(port);

    nic_ = RNic::create(RNicInfo::query_dev_names().at(nic_idx)).value();
    RDMA_ASSERT(ctrl_->opened_nics.reg(nic_name, nic_));

    // two int for head and tail
    mem_ = Arc<RMem>(new RMem(buf_size_ + sizeof(int) * 2));
    RDMA_ASSERT(ctrl_->registered_mrs.create_then_reg(
        mem_name, mem_, ctrl_->opened_nics.query(nic_name).value()));

    buf_ = (char *)(ctrl_->registered_mrs.query(mem_name)
                        .value()
                        ->get_reg_attr()
                        .value()
                        .buf);
    buf_head_ = (int *)(buf_ + buf_size_), buf_tail_ = buf_head_ + 1;
    *buf_head_ = *buf_tail_ = last_tail_ = 0;

    ctrl_->start_daemon();
}

void RDMABuffer::HostDestroy() {}

void RDMABuffer::GuestInit(std::string addr, uint64_t nic_idx,
                           uint64_t nic_name, uint64_t mem_name,
                           std::string qp_name)
{
    qp_name_ = qp_name;

    nic_ = RNic::create(RNicInfo::query_dev_names().at(nic_idx)).value();
    qp_ = RC::create(nic_, QPConfig()).value();

    cm_ = std::make_unique<ConnectManager>(addr);
    if (cm_->wait_ready(1000000, 10) == IOCode::Timeout)
        RDMA_ASSERT(false) << "cm connect to server timeout";

    auto qp_res = cm_->cc_rc(qp_name_, qp_, nic_name, QPConfig());
    RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
    qp_key_ = std::get<1>(qp_res.desc);
    RDMA_LOG(4) << "client fetch QP authentical key: " << qp_key_;

    mem_ = Arc<RMem>(new RMem(buf_size_ + sizeof(int) * 2));
    mr_ = RegHandler::create(mem_, nic_).value();

    auto fetch_res = cm_->fetch_remote_mr(mem_name);
    RDMA_ASSERT(fetch_res == IOCode::Ok) << std::get<0>(fetch_res.desc);
    rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);

    qp_->bind_remote_mr(remote_attr);
    qp_->bind_local_mr(mr_->get_reg_attr().value());

    buf_ = (char *)(mem_->raw_ptr);
    buf_head_ = (int *)(buf_ + buf_size_), buf_tail_ = buf_head_ + 1;
    *buf_head_ = *buf_tail_ = last_tail_ = 0;
}

void RDMABuffer::GuestDestroy()
{
    auto del_res = cm_->delete_remote_rc(qp_name_, qp_key_);
    RDMA_ASSERT(del_res == IOCode::Ok)
        << "delete remote QP error: " << del_res.desc;
}
