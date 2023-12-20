#ifndef ASYNC_BATCH_CALLER_H
#define ASYNC_BATCH_CALLER_H

#include "device_buffer.h"
#include "measurement.h"
#include "xdr_device.h"
#include <rpc/types.h>
#include <rpc/xdr.h>
#include <unordered_set>

class AsyncBatch
{
public:
    void Call(rpcproc_t proc, xdrproc_t xargs, void *argsp, xdrproc_t xresults,
              void *resultsp, struct timeval timeout, int &payload_size,
              XDR *xdrs_arg, XDR *xdrs_res, detailed_info *clnt_apis,
              int local_device, int retrieve_error, DeviceBuffer *sender,
              DeviceBuffer *receiver);
    AsyncBatch() {}
    static bool is_async_api(int proc_id);
};

#endif
