#include "svc.h"
#include "measurement.h"
#include "proxy_header.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

extern XDR *dispatch(int proc_id, XDR **xdrs_arg);

detailed_info svc_apis[API_COUNT];
extern int current_device;

void process_header(ProxyHeader &header)
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
            exit(-1);
        }
        time_end(svc_apis, cudaSetDevice_API, TOTAL_TIME);
    }
}

void svc_run()
{
    // ShmBuffer sender("/stoc", 62914552), receiver("/ctos", 62914552);
    ShmBuffer sender("/stoc", 10485752), receiver("/ctos", 10485752);
    printf("Shm_svc_run\n");

    while (1) {
        struct timeval start_0;
        gettimeofday(&start_0, NULL);
        int payload = 0;

        ProxyHeader header;
        auto ret = receiver.getBytes((char *)&header, sizeof(ProxyHeader));
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d\n", __func__, __FILE__,
                   __LINE__, header.get_proc_id());
            exit(-1);
        }
        // printf("proc_id = %d\n", proc_id);
        int proc_id = header.get_proc_id();
        process_header(header);

        int len = 0;
        ret = receiver.getBytes((char *)&len, sizeof(int));
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d\n", __func__, __FILE__,
                   __LINE__, proc_id);
            exit(-1);
        }
        XDR *xdrs_arg = new_xdrmemory(XDR_DECODE);
        XDRMemory *xdrmemory =
            reinterpret_cast<XDRMemory *>(xdrs_arg->x_private);
        xdrmemory->Resize(len);
        ret = receiver.getBytes(xdrmemory->Data(), len);
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n", __func__,
                   __FILE__, __LINE__, proc_id, len);
            exit(-1);
        }
        set_start(svc_apis, proc_id, NETWORK_TIME, &start_0);
        time_end(svc_apis, proc_id, NETWORK_TIME);
        payload += len + sizeof(int) + sizeof(ProxyHeader);

        XDR *xdrs_res = dispatch(proc_id, &xdrs_arg);

        time_start(svc_apis, proc_id, NETWORK_TIME);
        xdrmemory = reinterpret_cast<XDRMemory *>(xdrs_res->x_private);
        len = xdrmemory->Size();
        ret = sender.putBytes((char *)&len, sizeof(int));
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n", __func__,
                   __FILE__, __LINE__, proc_id, len);
            exit(-1);
        }
        ret = sender.putBytes(xdrmemory->Data(), len);
        if (ret < 0) {
            printf("timeout in %s in %s:%d, proc_id: %d, len: %d\n", __func__,
                   __FILE__, __LINE__, proc_id, len);
            exit(-1);
        }
        destroy_xdrmemory(&xdrs_res);
        time_end(svc_apis, proc_id, NETWORK_TIME);
        payload += len + sizeof(int);

        set_start(svc_apis, proc_id, TOTAL_TIME, &start_0);
        time_end(svc_apis, proc_id, TOTAL_TIME);
        add_cnt(svc_apis, proc_id);
        add_payload_size(svc_apis, proc_id, payload);
    }
}
