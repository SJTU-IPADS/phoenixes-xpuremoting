#include "clnt.h"
#include "async_batch_caller.h"
#include "measurement.h"
#include "proxy_header.h"
#include "../log.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <rpc/xdr.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>

static enum clnt_stat clnt_device_call(CLIENT *h, rpcproc_t proc,
                                       xdrproc_t xargs, void *argsp,
                                       xdrproc_t xresults, void *resultsp,
                                       struct timeval timeout);

static const struct clnt_ops clnt_device_ops = {
    clnt_device_call, NULL, NULL, NULL, NULL, NULL
};

BufferPool *buffers;
AsyncBatch *batchs;
// std::mutex *muts;

static void createBuffer()
{
    for (int i = 0; i < BUFFER_POOL_CAPACITY; i++) {
#ifdef WITH_TCPIP
        struct sockaddr_in address_receiver;
        int addrlen = sizeof(address_receiver);
        address_receiver.sin_family = AF_INET;
        address_receiver.sin_port = htons(TCPIP_PORT_STOC + i);
        inet_pton(AF_INET, "127.0.0.1", &address_receiver.sin_addr);
        buffers[i].receiver =
            new TcpipBuffer(BufferGuest, (struct sockaddr *)&address_receiver,
                            (socklen_t *)&addrlen, TCPIP_BUFFER_SIZE);
        struct sockaddr_in address_sender;
        addrlen = sizeof(address_sender);
        address_sender.sin_family = AF_INET;
        address_sender.sin_port = htons(TCPIP_PORT_CTOS + i);
        inet_pton(AF_INET, "127.0.0.1", &address_sender.sin_addr);
        buffers[i].sender =
            new TcpipBuffer(BufferGuest, (struct sockaddr *)&address_sender,
                            (socklen_t *)&addrlen, TCPIP_BUFFER_SIZE);
#endif // WITH_TCPIP
#ifdef WITH_SHARED_MEMORY
        std::stringstream ss;
        ss << SHM_NAME_STOC << i;
        buffers[i].receiver =
            new ShmBuffer(BufferGuest, ss.str().c_str(), SHM_BUFFER_SIZE);
        ss.str("");
        ss << SHM_NAME_CTOS << i;
        buffers[i].sender =
            new ShmBuffer(BufferGuest, ss.str().c_str(), SHM_BUFFER_SIZE);
#endif // WITH_SHARED_MEMORY
#ifdef WITH_RDMA
        buffers[i].sender = new RDMABuffer(
            BufferGuest, 0,
            "localhost:" + std::to_string(RDMA_CONNECTION_PORT_CTOS + i),
            RDMA_NIC_IDX_CTOS, RDMA_NIC_NAME_CTOS + i,
            RDMA_MEM_NAME_CTOS + i, "client-qp", RDMA_BUFFER_SIZE);
        buffers[i].receiver =
            new RDMABuffer(BufferHost, RDMA_CONNECTION_PORT_STOC + i, "",
                           RDMA_NIC_IDX_STOC, RDMA_NIC_NAME_STOC + i,
                           RDMA_MEM_NAME_STOC + i, "", RDMA_BUFFER_SIZE);
#endif // WITH_RDMA
        buffers[i].xdrs_arg = new_xdrmemory(XDR_ENCODE);
        buffers[i].xdrs_res = new_xdrmemory(XDR_DECODE);
    }
}

static void destroyBuffer()
{
    for (int i = 0; i < BUFFER_POOL_CAPACITY; i++) {
        destroy_xdrmemory(&buffers[i].xdrs_res);
        destroy_xdrmemory(&buffers[i].xdrs_arg);
        buffers[i].xdrs_arg = buffers[i].xdrs_res = nullptr;
#ifdef WITH_TCPIP
        TcpipBuffer *tcpip_sender =
                        dynamic_cast<TcpipBuffer *>(buffers[i].sender),
                    *tcpip_receiver =
                        dynamic_cast<TcpipBuffer *>(buffers[i].receiver);
        delete tcpip_sender;
        delete tcpip_receiver;
#endif // WITH_TCPIP
#ifdef WITH_SHARED_MEMORY
        ShmBuffer *shm_sender = dynamic_cast<ShmBuffer *>(buffers[i].sender),
                  *shm_receiver =
                      dynamic_cast<ShmBuffer *>(buffers[i].receiver);
        delete shm_sender;
        delete shm_receiver;
#endif // WITH_SHARED_MEMORY
#ifdef WITH_RDMA
        RDMABuffer *rdma_sender = dynamic_cast<RDMABuffer *>(buffers[i].sender),
                   *rdma_receiver =
                       dynamic_cast<RDMABuffer *>(buffers[i].receiver);
        delete rdma_sender;
        delete rdma_receiver;
#endif // WITH_RDMA
        buffers[i].sender = buffers[i].receiver = nullptr;
    }
}

int is_client = 0;

CLIENT *clnt_device_create()
{
    if (access("client_exist.txt", F_OK) == -1) {
        is_client = 1;
        std::ofstream outfile("client_exist.txt");
        outfile << "1";
        outfile.close();
        buffers = new BufferPool[BUFFER_POOL_CAPACITY];
        createBuffer();
        batchs = new AsyncBatch[BUFFER_POOL_CAPACITY];
        // muts = new std::mutex[BUFFER_POOL_CAPACITY];
    } else {
        is_client = 0;
    }
    CLIENT *client = (CLIENT *)malloc(sizeof(CLIENT));
    client->cl_ops = &clnt_device_ops;
    client->cl_auth = NULL;
    client->cl_private = NULL;
    return client;
}

detailed_info clnt_apis[API_COUNT];
// std::vector<uint64_t> thread_ids;

void clnt_device_destroy(CLIENT *clnt)
{
    if (is_client) {
        is_client = 0;
        free(clnt);
        remove("client_exist.txt");
        destroyBuffer();
        delete[] buffers;
        delete[] batchs;
        // delete[] muts;
        print_detailed_info(clnt_apis, API_COUNT, "client");
        // sort(thread_ids.begin(), thread_ids.end());
        // thread_ids.erase(unique(thread_ids.begin(), thread_ids.end()),
        //                  thread_ids.end());
        // std::fstream log_file("log.txt", std::ios::out);
        // for (auto id : thread_ids) {
        //     log_file << "thread id: " << id << std::endl;
        // }
        // log_file.close();
    }
}

thread_local int local_device = -1;

std::mutex buffer_num_mut;
int buffer_num = 0;


static enum clnt_stat clnt_device_call(CLIENT *h, rpcproc_t proc,
                                       xdrproc_t xargs, void *argsp,
                                       xdrproc_t xresults, void *resultsp,
                                       struct timeval timeout)
{
    // std::stringstream ss;
    // ss << std::this_thread::get_id();
    // uint64_t id = std::stoull(ss.str()) / 1000;
    // thread_ids.push_back(id);
    // int buffer_idx = id % BUFFER_POOL_CAPACITY;
    thread_local static int buffer_idx = -1;
    if (buffer_idx == -1) {
        std::lock_guard<std::mutex> lk(buffer_num_mut);
        buffer_idx = buffer_num++;
    }
    time_start(clnt_apis, proc, TOTAL_TIME);
    int payload = 0;

    if (!is_client) {
        return RPC_SUCCESS;
    }

    // std::lock_guard<std::mutex> lk(muts[buffer_idx]);

    batchs[buffer_idx].Call(proc, xargs, argsp, xresults, resultsp, timeout,
                            payload, buffers[buffer_idx].xdrs_arg,
                            buffers[buffer_idx].xdrs_res, clnt_apis,
                            local_device, buffers[buffer_idx].sender,
                            buffers[buffer_idx].receiver);

    time_end(clnt_apis, proc, TOTAL_TIME);
    add_cnt(clnt_apis, proc);
    add_payload_size(clnt_apis, proc, payload);
    return RPC_SUCCESS;
}
