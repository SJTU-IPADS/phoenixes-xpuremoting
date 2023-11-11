#include "svc.h"
#include "measurement.h"
#include "proxy_header.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

extern int dispatch(int proc_id, XDR *xdrs_arg, XDR *xdrs_res);

DeviceBuffer *sender = NULL, *receiver = NULL;

static void createDeviceBuffer()
{
#ifdef WITH_TCPIP
    struct sockaddr_in address_sender;
    int addrlen = sizeof(address_sender);
    address_sender.sin_family = AF_INET;
    address_sender.sin_addr.s_addr = INADDR_ANY;
    address_sender.sin_port = htons(TCPIP_PORT_STOC);
    sender = new TcpipBuffer(BufferHost, (struct sockaddr *)&address_sender,
                             (socklen_t *)&addrlen, TCPIP_BUFFER_SIZE);
    struct sockaddr_in address_receiver;
    addrlen = sizeof(address_receiver);
    address_receiver.sin_family = AF_INET;
    address_receiver.sin_addr.s_addr = INADDR_ANY;
    address_receiver.sin_port = htons(TCPIP_PORT_CTOS);
    receiver = new TcpipBuffer(BufferHost, (struct sockaddr *)&address_receiver,
                               (socklen_t *)&addrlen, TCPIP_BUFFER_SIZE);
    std::cout << "create tcpip buffer" << std::endl;
#endif // WITH_TCPIP
#ifdef WITH_SHARED_MEMORY
    sender = new ShmBuffer(BufferHost, SHM_NAME_STOC, SHM_BUFFER_SIZE);
    receiver = new ShmBuffer(BufferHost, SHM_NAME_CTOS, SHM_BUFFER_SIZE);
    std::cout << "create shm buffer" << std::endl;
#endif // WITH_SHARED_MEMORY
#ifdef WITH_RDMA
    receiver = new RDMABuffer(BufferHost, RDMA_CONNECTION_PORT_CTOS, "",
                              RDMA_NIC_IDX_CTOS, RDMA_NIC_NAME_CTOS,
                              RDMA_MEM_NAME_CTOS, "", RDMA_BUFFER_SIZE);
    sender =
        new RDMABuffer(BufferGuest, 0,
                       "localhost:" + std::to_string(RDMA_CONNECTION_PORT_STOC),
                       RDMA_NIC_IDX_STOC, RDMA_NIC_NAME_STOC,
                       RDMA_MEM_NAME_STOC, "client-qp", RDMA_BUFFER_SIZE);
    std::cout << "create rdma buffer" << std::endl;
#endif // WITH_RDMA
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
#ifdef WITH_RDMA
    RDMABuffer *rdma_sender = dynamic_cast<RDMABuffer *>(sender),
               *rdma_receiver = dynamic_cast<RDMABuffer *>(receiver);
    delete rdma_sender;
    delete rdma_receiver;
#endif // WITH_RDMA
    sender = receiver = NULL;
}

detailed_info svc_apis[API_COUNT];
extern int current_device;

int process_header(ProxyHeader &header)
{
#define cudaSetDevice_API 126
    int local_device = header.get_device_id();
    if (current_device != local_device) {
        add_cnt(svc_apis, cudaSetDevice_API);
        time_start(svc_apis, cudaSetDevice_API, TOTAL_TIME);
        current_device = local_device;
        int result = cudaSetDevice(current_device);
        if (result != cudaSuccess) {
            printf("cudaSetDevice failed, error code: %d\n", result);
            return -1;
        }
        time_end(svc_apis, cudaSetDevice_API, TOTAL_TIME);
    }
    return 0;
}

XDR *xdrs_arg, *xdrs_res;

void svc_run()
{
    createDeviceBuffer();
    xdrs_arg = new_xdrmemory(XDR_DECODE);
    xdrs_res = new_xdrmemory(XDR_ENCODE);

    while (1) {
        struct timeval start_0;
        gettimeofday(&start_0, NULL);
        int payload = 0;

        ProxyHeader header;
        auto ret = receiver->getBytes((char *)&header, sizeof(ProxyHeader));
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d\n", __func__, __FILE__,
                   __LINE__, header.get_proc_id());
            goto end;
        }
        // printf("proc_id = %d\n", proc_id);
        int proc_id = header.get_proc_id();
        if (process_header(header) < 0) {
            goto end;
        }

        int len = 0;
        ret = receiver->getBytes((char *)&len, sizeof(int));
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d\n", __func__, __FILE__,
                   __LINE__, proc_id);
            goto end;
        }
        XDRMemory *xdrmemory =
            reinterpret_cast<XDRMemory *>(xdrs_arg->x_private);
        xdrmemory->Resize(len);
        ret = receiver->getBytes(xdrmemory->Data(), len);
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n", __func__,
                   __FILE__, __LINE__, proc_id, len);
            goto end;
        }
        set_start(svc_apis, proc_id, NETWORK_TIME, &start_0);
        time_end(svc_apis, proc_id, NETWORK_TIME);
        payload += len + sizeof(int) + sizeof(ProxyHeader);

        if (dispatch(proc_id, xdrs_arg, xdrs_res) < 0) {
            goto end;
        }

        time_start(svc_apis, proc_id, NETWORK_TIME);
        xdrmemory = reinterpret_cast<XDRMemory *>(xdrs_res->x_private);
        len = xdrmemory->Size();
        ret = sender->putBytes((char *)&len, sizeof(int));
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n", __func__,
                   __FILE__, __LINE__, proc_id, len);
            goto end;
        }
        ret = sender->putBytes(xdrmemory->Data(), len);
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n", __func__,
                   __FILE__, __LINE__, proc_id, len);
            goto end;
        }
        sender->FlushOut();
        time_end(svc_apis, proc_id, NETWORK_TIME);
        payload += len + sizeof(int);

        set_start(svc_apis, proc_id, TOTAL_TIME, &start_0);
        time_end(svc_apis, proc_id, TOTAL_TIME);
        add_cnt(svc_apis, proc_id);
        add_payload_size(svc_apis, proc_id, payload);
    }

end:
    destroy_xdrmemory(&xdrs_res);
    destroy_xdrmemory(&xdrs_arg);
    destroyDeviceBuffer();
}
