#include "clnt.h"
#include "measurement.h"
#include "proxy_header.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>
#include <unordered_set>
#include <cassert>
#include "async_batch_caller.h"
#include <rpc/xdr.h>
#include <unistd.h>

static enum clnt_stat clnt_device_call(CLIENT *h, rpcproc_t proc, xdrproc_t xargs,
                                    void *argsp, xdrproc_t xresults,
                                    void *resultsp, struct timeval timeout);

static const struct clnt_ops clnt_device_ops = { clnt_device_call, NULL, NULL,
                                              NULL,          NULL, NULL };

DeviceBuffer *sender = NULL, *receiver = NULL;

static void createDeviceBuffer()
{
#ifdef WITH_TCPIP
    struct sockaddr_in address_receiver;
    int addrlen = sizeof(address_receiver);
    address_receiver.sin_family = AF_INET;
    address_receiver.sin_port = htons(TCPIP_PORT_STOC);
    inet_pton(AF_INET, "127.0.0.1", &address_receiver.sin_addr);
    receiver =
        new TcpipBuffer(BufferGuest, (struct sockaddr *)&address_receiver,
                        (socklen_t *)&addrlen, TCPIP_BUFFER_SIZE);
    struct sockaddr_in address_sender;
    addrlen = sizeof(address_sender);
    address_sender.sin_family = AF_INET;
    address_sender.sin_port = htons(TCPIP_PORT_CTOS);
    inet_pton(AF_INET, "127.0.0.1", &address_sender.sin_addr);
    sender = new TcpipBuffer(BufferGuest, (struct sockaddr *)&address_sender,
                             (socklen_t *)&addrlen, TCPIP_BUFFER_SIZE);
    // std::cout << "create tcpip buffer" << std::endl;
#endif // WITH_TCPIP
#ifdef WITH_SHARED_MEMORY
    receiver = new ShmBuffer(BufferGuest, SHM_NAME_STOC, SHM_BUFFER_SIZE);
    sender = new ShmBuffer(BufferGuest, SHM_NAME_CTOS, SHM_BUFFER_SIZE);
    // std::cout << "create shm buffer" << std::endl;
#endif // WITH_SHARED_MEMORY
#ifdef WITH_RDMA
    sender =
        new RDMABuffer(BufferGuest, 0,
                       "localhost:" + std::to_string(RDMA_CONNECTION_PORT_CTOS),
                       RDMA_NIC_IDX_CTOS, RDMA_NIC_NAME_CTOS,
                       RDMA_MEM_NAME_CTOS, "client-qp", RDMA_BUFFER_SIZE);
    receiver = new RDMABuffer(BufferHost, RDMA_CONNECTION_PORT_STOC, "",
                              RDMA_NIC_IDX_STOC, RDMA_NIC_NAME_STOC,
                              RDMA_MEM_NAME_STOC, "", RDMA_BUFFER_SIZE);
    // std::cout << "create rdma buffer" << std::endl;
#endif // WITH_RDMA
}

static void destroyDeviceBuffer()
{
#ifdef WITH_TCPIP
    TcpipBuffer *tcpip_sender = dynamic_cast<TcpipBuffer *>(sender),
                *tcpip_receiver = dynamic_cast<TcpipBuffer *>(receiver);
    delete tcpip_sender;
    delete tcpip_receiver;
#endif // WITH_TCPIP
#ifdef WITH_SHARED_MEMORY
    ShmBuffer *shm_sender = dynamic_cast<ShmBuffer *>(sender),
              *shm_receiver = dynamic_cast<ShmBuffer *>(receiver);
    delete shm_sender;
    delete shm_receiver;
#endif // WITH_SHARED_MEMORY
#ifdef WITH_RDMA
    RDMABuffer *rdma_sender = dynamic_cast<RDMABuffer *>(sender),
               *rdma_receiver = dynamic_cast<RDMABuffer *>(receiver);
    delete rdma_sender;
    delete rdma_receiver;
#endif // WITH_RDMA
    sender = receiver = NULL;
}

int is_client = 0;
XDR *xdrs_arg, *xdrs_res;

CLIENT *clnt_device_create()
{
    if (access("client_exist.txt", F_OK) == -1) {
        is_client = 1;
        std::ofstream outfile("client_exist.txt");
        outfile << "1";
        outfile.close();
        createDeviceBuffer();
        xdrs_arg = new_xdrmemory(XDR_ENCODE);
        xdrs_res = new_xdrmemory(XDR_DECODE);
    } else {
        is_client = 0;
    }
    CLIENT *client = (CLIENT *)malloc(sizeof(CLIENT));
    client->cl_ops = &clnt_device_ops;
    client->cl_auth = NULL;
    client->cl_private = NULL;
    return client;
}

// static int max_len = 0;

detailed_info clnt_apis[API_COUNT];

void clnt_device_destroy(CLIENT *clnt)
{
    free(clnt);
    if (is_client) {
        destroy_xdrmemory(&xdrs_res);
        destroy_xdrmemory(&xdrs_arg);
        destroyDeviceBuffer();
        remove("client_exist.txt");
        print_detailed_info(clnt_apis, API_COUNT, "client");
    }
    // std::cout << "max_len = " << max_len << std::endl;
}

std::mutex mut;
extern int local_device;

static enum clnt_stat clnt_device_call(CLIENT *h, rpcproc_t proc, xdrproc_t xargs,
                                    void *argsp, xdrproc_t xresults,
                                    void *resultsp, struct timeval timeout)
{
    static AsyncBatch batch;
    time_start(clnt_apis, proc, TOTAL_TIME);
    int payload = 0;

    if (!is_client) {
        return RPC_SUCCESS;
    }

    std::lock_guard<std::mutex> lk(mut);

    batch.Call(proc, xargs,argsp, xresults, resultsp, timeout,
        payload, xdrs_arg, xdrs_res, clnt_apis, local_device, sender, receiver);

    time_end(clnt_apis, proc, TOTAL_TIME);
    add_cnt(clnt_apis, proc);
    add_payload_size(clnt_apis, proc, payload);
    return RPC_SUCCESS;
}
