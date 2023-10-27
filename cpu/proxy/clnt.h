#ifndef CLNT_H
#define CLNT_H

#include "shm_buffer.h"
#include "tcpip_buffer.h"
#include "rdma_buffer.h"
#include "xdr_memory.h"
#include <rpc/rpc.h>

extern "C" {
CLIENT *clnt_device_create();
void clnt_device_destroy(CLIENT *clnt);
}

#endif /* CLNT_H */