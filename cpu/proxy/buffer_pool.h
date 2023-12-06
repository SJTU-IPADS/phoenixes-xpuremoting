#ifndef BUFFER_POOL_H
#define BUFFER_POOL_H

#include "device_buffer.h"
#include "xdr_memory.h"

#define BUFFER_POOL_CAPACITY 1

// just for Integrity
struct BufferPool {
    DeviceBuffer *sender, *receiver;
    XDR *xdrs_arg, *xdrs_res;

    BufferPool() {
        sender = nullptr;
        receiver = nullptr;
        xdrs_arg = nullptr;
        xdrs_res = nullptr;
    }
};

#endif // BUFFER_POOL_H
