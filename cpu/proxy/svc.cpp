#include "svc.h"
#include "async_batch_caller.h"
#include "measurement.h"
#include "proxy_header.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thread>

#ifdef POS_ENABLE
    #include "../cpu-server.h"
    #include "pos/include/common.h"
    #include "pos/include/log.h"
    #include "pos/cuda_impl/api_context.h"
#endif

extern int dispatch(int proc_id, XDR *xdrs_arg, XDR *xdrs_res);

BufferPool buffers[BUFFER_POOL_CAPACITY];

int remoting_shutdown = 0;

static void createBuffer()
{
    const char *ENV_REMOTE_ADDRESS = std::getenv("XPU_REMOTE_ADDRESS");
    if (ENV_REMOTE_ADDRESS == nullptr || strcmp(ENV_REMOTE_ADDRESS, "") == 0) {
        ENV_REMOTE_ADDRESS = "localhost";
    }
    printf("xpu remote address: %s\n", ENV_REMOTE_ADDRESS);
    for (int i = 0; i < BUFFER_POOL_CAPACITY; i++) {
#ifdef WITH_TCPIP
        struct sockaddr_in address_sender;
        int addrlen = sizeof(address_sender);
        address_sender.sin_family = AF_INET;
        address_sender.sin_addr.s_addr = INADDR_ANY;
        address_sender.sin_port = htons(TCPIP_PORT_STOC + i);
        buffers[i].sender =
            new TcpipBuffer(BufferHost, (struct sockaddr *)&address_sender,
                            (socklen_t *)&addrlen, TCPIP_BUFFER_SIZE);
        struct sockaddr_in address_receiver;
        addrlen = sizeof(address_receiver);
        address_receiver.sin_family = AF_INET;
        address_receiver.sin_addr.s_addr = INADDR_ANY;
        address_receiver.sin_port = htons(TCPIP_PORT_CTOS + i);
        buffers[i].receiver =
            new TcpipBuffer(BufferHost, (struct sockaddr *)&address_receiver,
                            (socklen_t *)&addrlen, TCPIP_BUFFER_SIZE);
        std::cout << "create tcpip buffer" << std::endl;
#endif // WITH_TCPIP
#ifdef WITH_SHARED_MEMORY
        std::stringstream ss;
        ss << SHM_NAME_STOC << i;
        buffers[i].sender =
            new ShmBuffer(BufferHost, ss.str().c_str(), SHM_BUFFER_SIZE);
        ss.str("");
        ss << SHM_NAME_CTOS << i;
        buffers[i].receiver =
            new ShmBuffer(BufferHost, ss.str().c_str(), SHM_BUFFER_SIZE);
        std::cout << "create shm buffer" << std::endl;
#endif // WITH_SHARED_MEMORY
#ifdef WITH_RDMA
        buffers[i].receiver =
            new RDMABuffer(BufferHost, RDMA_CONNECTION_PORT_CTOS + i, "",
                           RDMA_NIC_IDX_CTOS, RDMA_NIC_NAME_CTOS + i,
                           RDMA_MEM_NAME_CTOS + i, "", RDMA_BUFFER_SIZE);
        buffers[i].sender =
            new RDMABuffer(BufferGuest, 0,
                           ENV_REMOTE_ADDRESS + std::string(":") +
                               std::to_string(RDMA_CONNECTION_PORT_STOC + i),
                           RDMA_NIC_IDX_STOC, RDMA_NIC_NAME_STOC + i,
                           RDMA_MEM_NAME_STOC + i, "client-qp",
                           RDMA_BUFFER_SIZE);
        std::cout << "create rdma buffer" << std::endl;
#endif // WITH_RDMA
        buffers[i].xdrs_arg = new_xdrdevice(XDR_DECODE);
        auto xdrdevice =
            reinterpret_cast<XDRDevice *>(buffers[i].xdrs_arg->x_private);
        xdrdevice->SetBuffer(buffers[i].receiver);
        buffers[i].xdrs_res = new_xdrdevice(XDR_ENCODE);
        xdrdevice =
            reinterpret_cast<XDRDevice *>(buffers[i].xdrs_res->x_private);
        xdrdevice->SetBuffer(buffers[i].sender);
    }
}

static void destroyBuffer()
{
    for (int i = 0; i < BUFFER_POOL_CAPACITY; i++) {
        destroy_xdrdevice(&buffers[i].xdrs_res);
        destroy_xdrdevice(&buffers[i].xdrs_arg);
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

detailed_info svc_apis[API_COUNT];
thread_local int current_device = -1;

int process_header(ProxyHeader &header)
{
#define cudaSetDevice_API 126
#define cudaGetLastError_API 142
#ifndef NO_CACHE_OPTIMIZATION
    int local_device = header.get_device_id();
    // POS_LOG("current_device: %d, local_device: %d", current_device, local_device);
    if (current_device != local_device && local_device >= 0) {
        #ifdef POS_ENABLE
            current_device = local_device;
            int result = pos_cuda_ws->pos_process( 
                /* api_id */ CUDA_SET_DEVICE,
                /* uuid */ 0,
                /* param_desps */ {{ .value = &current_device, .size = sizeof(int) }}
            );
            POS_DEBUG("client %lu switch device to %d", 0, current_device);
        #else // POS_ENABLE
            add_cnt(svc_apis, cudaSetDevice_API);
            time_start(svc_apis, cudaSetDevice_API, TOTAL_TIME);
            current_device = local_device;
            int result = cudaSetDevice(current_device);
            time_end(svc_apis, cudaSetDevice_API, TOTAL_TIME);
        #endif
        if (result != cudaSuccess) {
            printf("cudaSetDevice failed, error code: %d, current device: "
                "%d, "
                "proc id: %d\n",
                result, current_device, header.get_proc_id());
            return -1;
        }
    }

    int retrieve_flag = header.get_retrieve_flag();
    if (retrieve_flag == 1) {
        add_cnt(svc_apis, cudaGetLastError_API);
        time_start(svc_apis, cudaGetLastError_API, TOTAL_TIME);
        int result = cudaGetLastError();
        if (result != cudaSuccess) {
            printf("cudaGetLastError failed, error code: %d, "
                   "proc id: %d\n",
                   result, header.get_proc_id());
            return -1;
        }
        time_end(svc_apis, cudaGetLastError_API, TOTAL_TIME);
    }
#endif // NO_CACHE_OPTIMIZATION
    return 0;
}

int receive_request(int buffer_idx, bool& has_new_request)
{
    ProxyHeader header;
    auto ret = buffers[buffer_idx].receiver->getBytes((char *)&header, sizeof(ProxyHeader));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %d\n", __func__, __FILE__, __LINE__, header.get_proc_id());
        return -1;
    } else if (ret == 0){
        has_new_request = false;
        return 0;
    }

    has_new_request = true;
    int proc_id = header.get_proc_id();
    if (process_header(header) < 0) {
        return -1;
    }

    return proc_id;
}

void print_config()
{
// cache optimization
#ifdef NO_CACHE_OPTIMIZATION
    printf("Cache Optimization: Disabled!\n");
#else
    printf("Cache Optimization: Enabled!\n");
#endif
// async optimization
#ifdef NO_ASYNC_OPTIMIZATION
    printf("Async Optimization: Disabled!\n");
#else
    printf("Async Optimization: Enabled!\n");
#endif
// handler optimization
#ifdef NO_HANDLER_OPTIMIZATION
    printf("Handler Optimization: Disabled!\n");
#else
    printf("Handler Optimization: Enabled!\n");
#endif
}

void svc_run()
{
    print_config();
    createBuffer();

    auto thread_serving = [&](int buffer_idx) {
        current_device = 0;
        // Make sure runtime API is initialized
        // If we don't do this and use the driver API, it might be unintialized
        // cudaError_t cres;
        // if ((cres = cudaSetDevice(current_device)) != cudaSuccess) {
        //     printf("cudaSetDevice failed, error code: %d\n", cres);
        //     goto end;
        // }

        #ifdef POS_ENABLE
            /*!
             *  \note   POS should init CUDA context in the worker thread
             */
        #else
            // seems only need to call cudaDeviceSynchronize once
            cudaDeviceSynchronize();
        #endif

        while (!remoting_shutdown) {
            uint64_t start_0 = rdtscp();
            int payload = 0;
            bool has_new_request = false;

            // try lock client
            // TODO: we assume the client id is always 0
            if(unlikely(pos_cuda_ws->try_lock_client(0) == 0)){
                // TODO: in future, we need to c/r the shm and stop the thread
                //      rpc thread should be re-spawned while restoring
                continue;
            }

            int proc_id = receive_request(buffer_idx, has_new_request);
            if (proc_id < 0) {
                goto end;
            }
            
            if(has_new_request == false){ continue; }

            int async = AsyncBatch::is_async_api(proc_id);

            #if defined(POS_ENABLE) && POS_ENABLE_HIJACK_API_CHECK
                if(unlikely(pos_is_hijacked(proc_id) == false)){
                    POS_ERROR_DETAIL("POS hasn't hijack api %lu", proc_id)
                }
            #endif // POS_ENABLE && POS_ENABLE_HIJACK_API_CHECK

            set_start(svc_apis, proc_id, NETWORK_RECEIVE_TIME, start_0);
            payload += sizeof(ProxyHeader);
            time_end(svc_apis, proc_id, NETWORK_RECEIVE_TIME);

            auto xdrdevice_arg = reinterpret_cast<XDRDevice *>(
                     buffers[buffer_idx].xdrs_arg->x_private),
                 xdrdevice_res = reinterpret_cast<XDRDevice *>(
                     buffers[buffer_idx].xdrs_res->x_private);
            xdrdevice_arg->Setpos(0);
            xdrdevice_res->Setpos(0);
            if (async == 1) {
                xdrdevice_res->SetMask(1);
            }
            if (dispatch(proc_id, buffers[buffer_idx].xdrs_arg,
                         buffers[buffer_idx].xdrs_res) < 0) {
                goto end;
            }
            payload += xdrdevice_arg->Getpos() + xdrdevice_res->Getpos();

            if (async == 1) {
                xdrdevice_res->SetMask(0);
                // do not need to send reply for an async api
                goto loop_end;
            }

            time_start(svc_apis, proc_id, NETWORK_SEND_TIME);
            buffers[buffer_idx].sender->FlushOut();
            time_end(svc_apis, proc_id, NETWORK_SEND_TIME);

        loop_end:
            set_start(svc_apis, proc_id, TOTAL_TIME, start_0);
            time_end(svc_apis, proc_id, TOTAL_TIME);
            add_cnt(svc_apis, proc_id);
            add_payload_size(svc_apis, proc_id, payload);
        }

    end:
        print_detailed_info(svc_apis, API_COUNT, "server");
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < BUFFER_POOL_CAPACITY; i++) {
        threads.emplace_back(thread_serving, i);
    }

    for (auto &t : threads) {
        t.join();
    }

    destroyBuffer();
}
