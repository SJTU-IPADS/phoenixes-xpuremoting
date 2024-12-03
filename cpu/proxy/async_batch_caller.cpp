#include "async_batch_caller.h"
#include "measurement.h"
#include "proxy_header.h"
#include <cassert>
#include <iostream>
#include <unordered_set>

static bool _is_async_api(int proc_id)
{
#ifdef NO_ASYNC_OPTIMIZATION
    static std::unordered_set<int> async_apis;
#else
    static std::unordered_set<int> async_apis{ 475,  317,  402,  5008, 5044,
                                               5048, 5309, 5304, 5310, 5302,
                                               5014, 5018, 5308, 5319, 3008,
                                               3010, 3004 };
#endif
#ifdef NO_HANDLER_OPTIMIZATION
    static std::unordered_set<int> handler_apis;
#else
    static std::unordered_set<int> handler_apis{ 5010, 5041, 5301 };
#endif
    return (async_apis.find(proc_id) != async_apis.end() ||
            handler_apis.find(proc_id) != handler_apis.end());
}

bool AsyncBatch::is_async_api(int proc_id) { return _is_async_api(proc_id); }

void AsyncBatch::Call(rpcproc_t proc, xdrproc_t xargs, void *argsp,
                      xdrproc_t xresults, void *resultsp,
                      struct timeval timeout, int &payload_size, XDR *xdrs_arg,
                      XDR *xdrs_res, detailed_info *clnt_apis, int local_device,
                      int retrieve_error, DeviceBuffer *sender,
                      DeviceBuffer *receiver)
{
    int async = AsyncBatch::is_async_api(proc);
    if (async == 1) {
        issuing_mode = ASYNC_ISSUING;
    } else {
        issuing_mode = SYNC_ISSUING;
    }

    time_start(clnt_apis, proc, NETWORK_SEND_TIME);
    ProxyHeader header(proc, local_device, retrieve_error);
    retrieve_error = 0;
    auto ret = sender->putBytes((char *)&header, sizeof(ProxyHeader));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc);
        exit(-1);
    }
    time_end(clnt_apis, proc, NETWORK_SEND_TIME);

    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    auto xdrdevice_arg = reinterpret_cast<XDRDevice *>(xdrs_arg->x_private);
    xdrdevice_arg->Setpos(0);
    (*xargs)(xdrs_arg, argsp);
    payload_size += sizeof(ProxyHeader) + xdrdevice_arg->Getpos();
    time_end(clnt_apis, proc, SERIALIZATION_TIME);

    time_start(clnt_apis, proc, NETWORK_SEND_TIME);
    sender->FlushOut();
    time_end(clnt_apis, proc, NETWORK_SEND_TIME);

    if (async == 1) {
        *(int *)resultsp = 0; // as CUDA_SUCCESS
        return;
    }

    time_start(clnt_apis, proc, NETWORK_RECEIVE_TIME);
    receiver->FillIn();
    time_end(clnt_apis, proc, NETWORK_RECEIVE_TIME);

    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    auto xdrdevice_res = reinterpret_cast<XDRDevice *>(xdrs_res->x_private);
    xdrdevice_res->Setpos(0);
    (*xresults)(xdrs_res, resultsp);
    payload_size += xdrdevice_res->Getpos();
    time_end(clnt_apis, proc, SERIALIZATION_TIME);
}
