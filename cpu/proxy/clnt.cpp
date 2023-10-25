#include "clnt.h"
#include "measurement.h"
#include "proxy_header.h"
#include "shm_buffer.h"
#include "tcpip_buffer.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>
#include <unordered_set>
#include <cassert>
#include "async_batch_caller.h"
#include <rpc/xdr.h>
#include <unistd.h>

static enum clnt_stat clnt_shm_call(CLIENT *h, rpcproc_t proc, xdrproc_t xargs,
                                    void *argsp, xdrproc_t xresults,
                                    void *resultsp, struct timeval timeout);

static const struct clnt_ops clnt_shm_ops = { clnt_shm_call, NULL, NULL,
                                              NULL,          NULL, NULL };

DeviceBuffer *sender, *receiver;

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
#endif // WITH_TCPIP
#ifdef WITH_SHARED_MEMORY
    receiver = new ShmBuffer(BufferGuest, SHM_NAME_STOC, SHM_BUFFER_SIZE);
    sender = new ShmBuffer(BufferGuest, SHM_NAME_CTOS, SHM_BUFFER_SIZE);
#endif // WITH_SHARED_MEMORY
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
    sender = receiver = NULL;
}

int is_client = 0;
XDR *xdrs_arg, *xdrs_res;

CLIENT *clnt_shm_create()
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
    printf("clnt_shm_create\n");
    CLIENT *client = (CLIENT *)malloc(sizeof(CLIENT));
    client->cl_ops = &clnt_shm_ops;
    client->cl_auth = NULL;
    client->cl_private = NULL;
    return client;
}

// static int max_len = 0;

detailed_info clnt_apis[API_COUNT];

void clnt_shm_destroy(CLIENT *clnt)
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

bool is_async_api(int proc_id) {
    static std::unordered_set<int> async_apis{475, 317, 402, 5008, /*5041,*/ 5044, 5048, /*5301,*/ 5309, 5304, 5310, 5302, /*5010,*/ 5014, 5018, 5308, 5319, 3008, 3010, 3004};
    return (async_apis.find(proc_id) != async_apis.end());
}

void call_async_api(rpcproc_t proc, xdrproc_t xargs,
                    void *argsp, xdrproc_t xresults,
                    void *resultsp, struct timeval timeout, AsyncBatch& batch) {
    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    XDRMemory *xdrmemory = reinterpret_cast<XDRMemory *>(xdrs_arg->x_private);
    xdrmemory->Clear();
    (*xargs)(xdrs_arg, argsp);
    AsyncCall call(proc, local_device, xdrmemory->GetBuffer());
    batch.Push(call);
    time_end(clnt_apis, proc, SERIALIZATION_TIME);
    *(int*) resultsp = 0; // as CUDA_SUCCESS
    return;
}

void send_request(int proc_id, int device_num, char* payload, int len) {
    ProxyHeader header(proc_id, device_num);
    time_start(clnt_apis, proc_id, NETWORK_TIME);
    auto ret = sender->putBytes((char *)&header, sizeof(ProxyHeader));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc_id);
        exit(-1);
    }
    ret = sender->putBytes((char *)&len, sizeof(int));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc_id);
        exit(-1);
    }
    ret = sender->putBytes((char *)payload, len);
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc_id);
        exit(-1);
    }
    sender->FlushOut();
    time_end(clnt_apis, proc_id, NETWORK_TIME);
}

int receive_response(int proc_id) {
    int len;
    time_start(clnt_apis, proc_id, NETWORK_TIME);
    auto ret = receiver->getBytes((char *)&len, sizeof(int));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc_id);
        exit(-1);
    }
    auto xdrmemory = reinterpret_cast<XDRMemory *>(xdrs_res->x_private);
    xdrmemory->Resize(len);
    ret = receiver->getBytes(xdrmemory->Data(), len);
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu, len: %d\n", __func__,
               __FILE__, __LINE__, proc_id, len);
        exit(-1);
    }
    time_end(clnt_apis, proc_id, NETWORK_TIME);
    return len;
}

static enum clnt_stat clnt_shm_call(CLIENT *h, rpcproc_t proc, xdrproc_t xargs,
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

    if (is_async_api(proc)) {
        call_async_api(proc, xargs, argsp, xresults, resultsp, timeout, batch);
        return RPC_SUCCESS;
    }

    // write request count first
    int request_count = batch.Size() + 1;
    assert(request_count >= 1);
    time_start(clnt_apis, proc, NETWORK_TIME);
    auto ret = sender->putBytes((char*)&request_count, sizeof(request_count));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc);
        exit(-1);
    }
    payload += sizeof(request_count);
    // printf("write request count %d\n", request_count);
    time_end(clnt_apis, proc, NETWORK_TIME);

    // execute the former async apis
    for (int i = 0; i < request_count-1; i++) {
        auto& call = batch.Front();
        send_request(call.proc_id, call.device_num, (char*) call.payload.data(), call.payload.size());
        payload += call.payload.size() + sizeof(int) + sizeof(ProxyHeader);
        batch.Pop();
    }

    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    XDRMemory *xdrmemory = reinterpret_cast<XDRMemory *>(xdrs_arg->x_private);
    xdrmemory->Clear();
    (*xargs)(xdrs_arg, argsp);
    time_end(clnt_apis, proc, SERIALIZATION_TIME);

    send_request(proc, local_device, xdrmemory->Data(), xdrmemory->Size());
    payload += xdrmemory->Size() + sizeof(int) + sizeof(ProxyHeader);

    auto len = receive_response(proc);
    payload += len + sizeof(int);

    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    (*xresults)(xdrs_res, resultsp);
    time_end(clnt_apis, proc, SERIALIZATION_TIME);

    time_end(clnt_apis, proc, TOTAL_TIME);
    add_cnt(clnt_apis, proc);
    add_payload_size(clnt_apis, proc, payload);
    return RPC_SUCCESS;
}
