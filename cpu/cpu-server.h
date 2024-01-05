#ifndef _CPU_SERVER_H_
#define _CPU_SERVER_H_

#include <stddef.h>

#ifdef POS_ENABLE
    #include "pos/include/common.h"
    #include "pos/include/transport.h"
    #include "pos/cuda_impl/workspace.h"

    extern POSWorkspace_CUDA<POSTransport_SHM> *pos_cuda_ws;
#endif

void cricket_main(size_t prog_version, size_t vers_num);

#endif //_CPU_SERVER_H_
