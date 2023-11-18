#ifndef ASYNC_BATCH_CALLER_H
#define ASYNC_BATCH_CALLER_H

#include <vector>
#include <deque>
#include <rpc/xdr.h>
#include "measurement.h"
#include "device_buffer.h"
#include <rpc/types.h>
#include <unordered_set>

class AsyncCall {
    public:
        int device_num;
        int proc_id;
        std::vector<char> payload;
        AsyncCall(int proc_id, int device_num, std::vector<char>& payload) : proc_id(proc_id), device_num(device_num), payload(payload) {}
        AsyncCall(int proc_id, int device_num, std::vector<char>&& payload) : proc_id(proc_id), device_num(device_num), payload(payload) {}
};

class AsyncBatch {
    public:
        void Call(rpcproc_t proc, xdrproc_t xargs, void *argsp, xdrproc_t xresults, void *resultsp, struct timeval timeout, int& payload_size, XDR *xdrs_arg, XDR *xdrs_res, detailed_info* clnt_apis, int local_device, DeviceBuffer *sender, DeviceBuffer *receiver);
        AsyncBatch() {}
        AsyncBatch(int max_batch_size) : max_batch_size(max_batch_size) {}
        static bool is_async_api(int proc_id);
    private:
        std::deque<AsyncCall> queue = {};
        AsyncCall& Front();
        void Pop();
        void Clear();
        void Push(AsyncCall& call);
        int Size();
        int max_batch_size = 16;
        void call_async_api(rpcproc_t proc, xdrproc_t xargs,
            void *argsp, xdrproc_t xresults,
            void *resultsp, struct timeval timeout, XDR *xdrs_arg, detailed_info* clnt_apis, int local_device);
};

#endif
