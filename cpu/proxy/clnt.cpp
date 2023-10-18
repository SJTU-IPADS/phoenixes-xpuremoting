#include "clnt.h"
#include "measurement.h"
#include "proxy_header.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <rpc/xdr.h>
#include <unistd.h>

ShmBuffer *sender, *receiver;

static enum clnt_stat clnt_shm_call(CLIENT *h, rpcproc_t proc, xdrproc_t xargs,
                                    void *argsp, xdrproc_t xresults,
                                    void *resultsp, struct timeval timeout);

static const struct clnt_ops clnt_shm_ops = { clnt_shm_call, NULL, NULL,
                                              NULL,          NULL, NULL };

int is_client = 0;

CLIENT *clnt_shm_create()
{
    if (access("client_exist.txt", F_OK) == -1) {
        is_client = 1;
        std::ofstream outfile("client_exist.txt");
        outfile << "1";
        outfile.close();
        // sender = new ShmBuffer("/ctos", 62914552);
        // receiver = new ShmBuffer("/stoc", 62914552);
        sender = new ShmBuffer("/ctos", 10485752);
        receiver = new ShmBuffer("/stoc", 10485752);
    } else {
        is_client = 0;
    }
    printf("clnt_shm_create\n");
    CLIENT *client = (CLIENT *)malloc(sizeof(CLIENT));
    client->cl_ops = &clnt_shm_ops;
    client->cl_auth = NULL;
    client->cl_private = NULL;
    return client;
}

// static int max_len = 0;

detailed_info clnt_apis[API_COUNT];

void clnt_shm_destroy(CLIENT *clnt)
{
    free(clnt);
    if (is_client) {
        delete sender;
        delete receiver;
        remove("client_exist.txt");
        print_detailed_info(clnt_apis, API_COUNT, "client");
    }
    // std::cout << "max_len = " << max_len << std::endl;
}

std::mutex mut;
extern int local_device;

static enum clnt_stat clnt_shm_call(CLIENT *h, rpcproc_t proc, xdrproc_t xargs,
                                    void *argsp, xdrproc_t xresults,
                                    void *resultsp, struct timeval timeout)
{
    time_start(clnt_apis, proc, TOTAL_TIME);
    int payload = 0;

    if (!is_client) {
        return RPC_SUCCESS;
    }

    std::lock_guard<std::mutex> lk(mut);

    time_start(clnt_apis, proc, NETWORK_TIME);
    ProxyHeader header(proc, local_device);
    auto ret = sender->putBytes((char *)&header, sizeof(ProxyHeader));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc);
        exit(-1);
    }
    time_end(clnt_apis, proc, NETWORK_TIME);
    payload += sizeof(ProxyHeader);

    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    XDR *xdrs_arg = new_xdrmemory(XDR_ENCODE);
    (*xargs)(xdrs_arg, argsp);
    time_end(clnt_apis, proc, SERIALIZATION_TIME);

    time_start(clnt_apis, proc, NETWORK_TIME);
    XDRMemory *xdrmemory = reinterpret_cast<XDRMemory *>(xdrs_arg->x_private);
    int len = xdrmemory->Size();
    // max_len = std::max(max_len, len);
    ret = sender->putBytes((char *)&len, sizeof(int));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc);
        exit(-1);
    }
    ret = sender->putBytes(xdrmemory->Data(), len);
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu, len: %d\n", __func__,
               __FILE__, __LINE__, proc, len);
        exit(-1);
    }
    destroy_xdrmemory(&xdrs_arg);
    time_end(clnt_apis, proc, NETWORK_TIME);
    payload += len + sizeof(int);

    time_start(clnt_apis, proc, NETWORK_TIME);
    ret = receiver->getBytes((char *)&len, sizeof(int));
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu\n", __func__, __FILE__,
               __LINE__, proc);
        exit(-1);
    }
    XDR *xdrs_res = new_xdrmemory(XDR_DECODE);
    xdrmemory = reinterpret_cast<XDRMemory *>(xdrs_res->x_private);
    xdrmemory->Resize(len);
    ret = receiver->getBytes(xdrmemory->Data(), len);
    if (ret < 0) {
        printf("timeout in %s in %s:%d, proc_id: %lu, len: %d\n", __func__,
               __FILE__, __LINE__, proc, len);
        exit(-1);
    }
    time_end(clnt_apis, proc, NETWORK_TIME);
    payload += len + sizeof(int);

    time_start(clnt_apis, proc, SERIALIZATION_TIME);
    (*xresults)(xdrs_res, resultsp);
    time_end(clnt_apis, proc, SERIALIZATION_TIME);

    time_end(clnt_apis, proc, TOTAL_TIME);
    add_cnt(clnt_apis, proc);
    add_payload_size(clnt_apis, proc, payload);
    return RPC_SUCCESS;
}
