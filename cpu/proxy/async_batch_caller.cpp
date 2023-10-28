#include "async_batch_caller.h"
#include "measurement.h"
#include "proxy_header.h"
#include "xdr_memory.h"
#include "device_buffer.h"
#include <cassert>
#include <iostream>
#include <unordered_set>

static bool is_async_api(int proc_id) {
    static std::unordered_set<int> async_apis{475, 317, 402, 5008, /*5041,*/ 5044, 5048, /*5301,*/ 5309, 5304, 5310, 5302, /*5010,*/ 5014, 5018, 5308, 5319, 3008, 3010, 3004};
    return (async_apis.find(proc_id) != async_apis.end());
}

void send_request(int proc_id, int device_num, char* payload, int len, DeviceBuffer *sender, detailed_info* clnt_apis) {
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
    // sender->FlushOut();
    time_end(clnt_apis, proc_id, NETWORK_TIME);
}

int receive_response(int proc_id, DeviceBuffer *receiver, detailed_info* clnt_apis, XDR* xdrs_res) {
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

int AsyncBatch::Size() {
    return queue.size();
}

AsyncCall& AsyncBatch::Front() {
    return queue.front();
}

void AsyncBatch::Pop() {
    queue.pop_front();
}

void AsyncBatch::Clear() {
    queue.clear();
}

void AsyncBatch::Push(AsyncCall& call) {
    queue.push_back(call);
}

void AsyncBatch::Call(rpcproc_t proc, xdrproc_t xargs,
    void *argsp, xdrproc_t xresults,
    void *resultsp, struct timeval timeout, int& payload_size,
    XDR *xdrs_arg, XDR *xdrs_res, detailed_info* clnt_apis, int local_device,
    DeviceBuffer *sender, DeviceBuffer *receiver) {

    // if this is an async api and we can fit it in a batch, just call it
    if (is_async_api(proc) && this->Size() < this->max_batch_size) {
        this->call_async_api(proc, xargs, argsp, xresults, resultsp, timeout, xdrs_arg, clnt_apis, local_device);
        return;
    }

    // we need to write the request count first as an int
    // write request count first
    int request_count = this->Size() + 1;
    assert(request_count >= 1);
    time_start(clnt_apis, proc, NETWORK_TIME);
    auto ret = sender->putBytes((char*)&request_count, sizeof(request_count));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc);
        exit(-1);
    }
    payload_size += sizeof(request_count);
    time_end(clnt_apis, proc, NETWORK_TIME);

    // execute the former async apis
    for (int i = 0; i < request_count-1; i++) {
        auto& call = this->Front();
        send_request(call.proc_id, call.device_num, (char*) call.payload.data(), call.payload.size(), sender, clnt_apis);
        payload_size += call.payload.size() + sizeof(int) + sizeof(ProxyHeader);
        this->Pop();
    }

    // execute the last request
    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    XDRMemory *xdrmemory = reinterpret_cast<XDRMemory *>(xdrs_arg->x_private);
    xdrmemory->Clear();
    (*xargs)(xdrs_arg, argsp);
    time_end(clnt_apis, proc, SERIALIZATION_TIME);

    send_request(proc, local_device, xdrmemory->Data(), xdrmemory->Size(), sender, clnt_apis);
    payload_size += xdrmemory->Size() + sizeof(int) + sizeof(ProxyHeader);

    sender->FlushOut();

    auto len = receive_response(proc, receiver, clnt_apis, xdrs_res);
    payload_size += len + sizeof(int);

    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    (*xresults)(xdrs_res, resultsp);
    time_end(clnt_apis, proc, SERIALIZATION_TIME);
}

void AsyncBatch::call_async_api(rpcproc_t proc, xdrproc_t xargs,
    void *argsp, xdrproc_t xresults,
    void *resultsp, struct timeval timeout, XDR *xdrs_arg, detailed_info* clnt_apis, int local_device) {
    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    XDRMemory *xdrmemory = reinterpret_cast<XDRMemory *>(xdrs_arg->x_private);
    xdrmemory->Clear();
    (*xargs)(xdrs_arg, argsp);
    AsyncCall call(proc, local_device, xdrmemory->GetBuffer());
    this->Push(call);
    time_end(clnt_apis, proc, SERIALIZATION_TIME);
    *(int*) resultsp = 0; // as CUDA_SUCCESS
    return;
}
