#include "svc.h"
#include "async_batch_caller.h"
#include "measurement.h"
#include "proxy_header.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thread>

extern int dispatch(int proc_id, XDR *xdrs_arg, XDR *xdrs_res);

BufferPool buffers[BUFFER_POOL_CAPACITY];

static void createBuffer()
{
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
        buffers[i].sender = new RDMABuffer(
            BufferGuest, 0,
            "localhost:" + std::to_string(RDMA_CONNECTION_PORT_STOC + i),
            RDMA_NIC_IDX_STOC, RDMA_NIC_NAME_STOC + i, RDMA_MEM_NAME_STOC + i,
            "client-qp", RDMA_BUFFER_SIZE);
        std::cout << "create rdma buffer" << std::endl;
#endif // WITH_RDMA
        buffers[i].xdrs_arg = new_xdrmemory(XDR_DECODE);
        buffers[i].xdrs_res = new_xdrmemory(XDR_ENCODE);
    }
}

static void destroyBuffer()
{
    for (int i = 0; i < BUFFER_POOL_CAPACITY; i++) {
        destroy_xdrmemory(&buffers[i].xdrs_res);
        destroy_xdrmemory(&buffers[i].xdrs_arg);
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
    int local_device = header.get_device_id();
    if (current_device != local_device && local_device >= 0) {
        add_cnt(svc_apis, cudaSetDevice_API);
        time_start(svc_apis, cudaSetDevice_API, TOTAL_TIME);
        current_device = local_device;
        int result = cudaSetDevice(current_device);
        if (result != cudaSuccess) {
            printf("cudaSetDevice failed, error code: %d, current device: "
                   "%d, "
                   "proc id: %d\n",
                   result, current_device, header.get_proc_id());
            return -1;
        }
        time_end(svc_apis, cudaSetDevice_API, TOTAL_TIME);
    }
    return 0;
}

std::pair<int, int> receive_request(int buffer_idx)
{
    ProxyHeader header;
    auto ret = buffers[buffer_idx].receiver->getBytes((char *)&header,
                                                      sizeof(ProxyHeader));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %d\n", __func__, __FILE__,
               __LINE__, header.get_proc_id());
        return { -1, -1 };
    }
    int proc_id = header.get_proc_id();
    if (process_header(header) < 0) {
        return { -1, -1 };
    }

    int len = 0;
    ret = buffers[buffer_idx].receiver->getBytes((char *)&len, sizeof(int));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %d\n", __func__, __FILE__,
               __LINE__, proc_id);
        return { -1, -1 };
    }
    XDRMemory *xdrmemory =
        reinterpret_cast<XDRMemory *>(buffers[buffer_idx].xdrs_arg->x_private);
    xdrmemory->Resize(len);
    ret = buffers[buffer_idx].receiver->getBytes(xdrmemory->Data(), len);
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n", __func__,
               __FILE__, __LINE__, proc_id, len);
        return { -1, -1 };
    }
    return { proc_id, len };
}

int send_response(XDRMemory *xdrmemory, int proc_id, int buffer_idx)
{
    auto len = xdrmemory->Size();
    auto ret = buffers[buffer_idx].sender->putBytes((char *)&len, sizeof(int));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n", __func__,
               __FILE__, __LINE__, proc_id, len);
        return -1;
    }
    ret = buffers[buffer_idx].sender->putBytes(xdrmemory->Data(), len);
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n", __func__,
               __FILE__, __LINE__, proc_id, len);
        return -1;
    }
    buffers[buffer_idx].sender->FlushOut();
    return 0;
}

void svc_run()
{
    createBuffer();

    auto thread_serving = [&](int buffer_idx) {
        current_device = 0;
        // Make sure runtime API is initialized
        // If we don't do this and use the driver API, it might be unintialized
        cudaError_t cres;
        if ((cres = cudaSetDevice(current_device)) != cudaSuccess) {
            printf("cudaSetDevice failed, error code: %d\n", cres);
            goto end;
        }
        cudaDeviceSynchronize();

        while (1) {
            struct timeval start_0;
            gettimeofday(&start_0, NULL);
            int payload = 0, ret = 0;
            XDRMemory *xdrmemory = nullptr;

            auto [proc_id, len] = receive_request(buffer_idx);
            if (proc_id < 0) {
                goto end;
            }

            set_start(svc_apis, proc_id, NETWORK_TIME, &start_0);
            time_end(svc_apis, proc_id, NETWORK_TIME);
            payload += len + sizeof(int) + sizeof(ProxyHeader);

            if (dispatch(proc_id, buffers[buffer_idx].xdrs_arg,
                         buffers[buffer_idx].xdrs_res) < 0) {
                goto end;
            }

            if (AsyncBatch::is_async_api(proc_id)) {
                // do not need to send reply for an async api
                goto loop_end;
            }

            time_start(svc_apis, proc_id, NETWORK_TIME);
            xdrmemory = reinterpret_cast<XDRMemory *>(
                buffers[buffer_idx].xdrs_res->x_private);
            len = xdrmemory->Size();
            ret = send_response(xdrmemory, proc_id, buffer_idx);
            if (ret < 0) {
                printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n",
                       __func__, __FILE__, __LINE__, proc_id, len);
                goto end;
            }

            time_end(svc_apis, proc_id, NETWORK_TIME);
            payload += len + sizeof(int);

        loop_end:
            set_start(svc_apis, proc_id, TOTAL_TIME, &start_0);
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
