#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h> //unlink()
#include <signal.h> //sigaction
#include <sys/types.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <link.h>

#include "cpu-server.h"
#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "cpu-server-runtime.h"
#include "cpu-server-driver.h"
#include "rpc/xdr.h"
#include "cr.h"
#include "cpu-elf2.h"
#ifdef WITH_IB
#include "cpu-ib.h"
#endif //WITH_IB
#define WITH_RECORDER
#include "api-recorder.h"
#include "gsched.h"
#include "cpu-server-nvml.h"
#include "cpu-server-cudnn.h"

#include "cpu-measurement.h"

#ifndef NO_OPTIMIZATION
    #include "proxy/measurement.h"
    extern int remoting_shutdown;
#endif //WITH_OPTIMIZATION

#ifndef NO_HANDLER_OPTIMIZATION
    #include "proxy/handler_mapper.h"
#endif

extern cpu_measurement_info vanillas[CPU_API_COUNT];
#ifndef NO_OPTIMIZATION
extern detailed_info svc_apis[API_COUNT];
#endif //WITH_OPTIMIZATION
#ifndef NO_HANDLER_OPTIMIZATION
HandlerMapper handler_mapper;
#endif

INIT_SOCKTYPE

int connection_is_local = 0;
int shm_enabled = 1;

#ifdef WITH_IB
    int ib_device = 0;
#endif //WITH_IB

extern gsched_t sched_none;
gsched_t *sched;

//Runtime API RMs
resource_mg rm_streams;
resource_mg rm_events;
resource_mg rm_arrays;
resource_mg rm_memory;
resource_mg rm_kernels;

//Driver API RMs
resource_mg rm_modules;
resource_mg rm_functions;
resource_mg rm_globals;

//Other RMs
resource_mg rm_cusolver;
resource_mg rm_cublas;

//CUDNN RMs
resource_mg rm_cudnn;
resource_mg rm_cudnn_tensors;
resource_mg rm_cudnn_filters;
resource_mg rm_cudnn_tensortransform;
resource_mg rm_cudnn_poolings;
resource_mg rm_cudnn_activations;
resource_mg rm_cudnn_lrns;
resource_mg rm_cudnn_convs;
resource_mg rm_cudnn_backendds;

unsigned long prog=0, vers=0;

#if defined(POS_ENABLE)
    #include "pos/include/common.h"
    #include "pos/include/transport.h"
    #include "pos/cuda_impl/workspace.h"

    POSWorkspace_CUDA *pos_cuda_ws;
#endif

extern void rpc_cd_prog_1(struct svc_req *rqstp, register SVCXPRT *transp);

void int_handler(int signal) {
    if (socktype == UNIX) {
        unlink(CD_SOCKET_PATH);
    }
    LOG(LOG_INFO, "have a nice day!\n");

    #ifndef NO_OPTIMIZATION
        remoting_shutdown = 1;
    #else
        svc_exit();
    #endif
}

bool_t rpc_printmessage_1_svc(char *argp, int *result, struct svc_req *rqstp)
{
    int proc = 2;
    cpu_time_start(vanillas, proc);
    LOG(LOG_INFO, "string: \"%s\"\n", argp);
    *result = 42;
    cpu_time_end(vanillas, proc);
    return 1;
}

bool_t rpc_deinit_1_svc(int *result, struct svc_req *rqstp)
{
    cpu_print_measurement_info("server_vanilla_", vanillas, CPU_API_COUNT);
#ifndef NO_OPTIMIZATION
    print_detailed_info(svc_apis, API_COUNT, "server");
#endif //WITH_OPTIMIZATION
    LOG(LOG_INFO, "RPC deinit requested.");
    // svc_exit();
    return 1;
}

int cricket_server_checkpoint(int dump_memory)
{
    int ret;
    struct stat path_stat = { 0 };
    const char *ckp_path = "ckp";
    LOG(LOG_INFO, "rpc_checkpoint requested.");

    if (!ckp_path) {
        LOGE(LOG_ERROR, "ckp_path is NULL");
        goto error;
    }
    if (stat(ckp_path, &path_stat) != 0) {
        LOG(LOG_DEBUG, "directory \"%s\" does not exist. Let's create it.", ckp_path);
        if (mkdir(ckp_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            LOGE(LOG_ERROR, "failed to create directory \"%s\"", ckp_path);
            goto error;
        }
    } else if (!S_ISDIR(path_stat.st_mode)) {
        LOG(LOG_ERROR, "file \"%s\" is not a directory", ckp_path);
        goto error;
    }
    
    if ((ret = server_runtime_checkpoint(ckp_path, dump_memory, prog, vers)) != 0) {
        LOGE(LOG_ERROR, "server_runtime_checkpoint returned %d", ret);
        goto error;
    }

    LOG(LOG_INFO, "checkpoint successfully created.");
    return 0;
 error:
    LOG(LOG_INFO, "checkpoint creation failed.");
    return 1;
}

static void signal_checkpoint(int signo)
{
    if (cricket_server_checkpoint(1) != 0) {
        LOGE(LOG_ERROR, "failed to create checkpoint");
    }
}

bool_t rpc_checkpoint_1_svc(int *result, struct svc_req *rqstp)
{
    int ret;
    if ((ret = cricket_server_checkpoint(1)) != 0) {
        LOGE(LOG_ERROR, "failed to create checkpoint");
    }
    return ret == 0;
}

/* Call CUDA initialization function (usually called by __libc_init_main())
* Address of "_ZL24__sti____cudaRegisterAllv" in static symbol table is e.g. 0x4016c8
*/
void cricket_so_register(void* dlhandle, char *path)
{
    // struct link_map *map;
    // dlinfo(dlhandle, RTLD_DI_LINKMAP, &map);

    // // add load location of library to offset in symbol table
    // void (*cudaRegisterAllv)(void) = 
    //     (void(*)(void)) elf_symbol_address(path, "_ZL24__sti____cudaRegisterAllv");
    
    // LOG(LOG_INFO, "found CUDA initialization function at %p + %p = %p", 
    //     map->l_addr, cudaRegisterAllv, map->l_addr + cudaRegisterAllv);

    // cudaRegisterAllv += map->l_addr;
    
    // if (cudaRegisterAllv == NULL) {
    //     LOGE(LOG_WARNING, "could not find cudaRegisterAllv initialization function in cubin. Kernels cannot be launched without it!");
    // } else {
    //     cudaRegisterAllv();
    // }
}

bool_t rpc_dlopen_1_svc(char *path, int *result, struct svc_req *rqstp)
{
    void *dlhandle;

    if (path == NULL) {
        LOGE(LOG_ERROR, "path is NULL");
        *result = 1;
        return 1;
    }
    if ((dlhandle = dlopen(path, RTLD_LAZY)) == NULL) {
        LOGE(LOG_ERROR, "error opening \"%s\": %s. Make sure libraries are present.", path, dlerror());
        *result = 1;
        return 1;
    } else {
        LOG(LOG_INFO, "dlopened \"%s\"", path);

       //cricket_so_register(dlhandle, path);

    }
    *result = 0;
    return 1;
}

void cricket_main(size_t prog_num, size_t vers_num, int argc, char *argv[])
{
    int ret = 1;
    register SVCXPRT *transp;

    int protocol = 0;
    int restore = 0;
    struct sigaction act;
    char *command = NULL;
    act.sa_handler = int_handler;
    sigaction(SIGINT, &act, NULL);
    LOG(LOG_DBG(1), "log level is %d", LOG_LEVEL);
    init_log(LOG_LEVEL, __FILE__);

    #ifdef WITH_IB
    char client[256];
    char envvar[] = "CLIENT_ADDRESS";

    if(!getenv(envvar)) {
        LOG(LOG_ERROR, "Environment variable %s does not exist. For memory transports using InfiniBand it must contain the address of the client.", envvar);
        exit(1);
    }
    if(strncpy(client, getenv(envvar), 256) == NULL) {
        LOGE(LOG_ERROR, "strncpy failed.");
        exit(1);
    }
    LOG(LOG_INFO, "connection to client \"%s\"", client);

    if(getenv("IB_DEVICE_ID")) {
        ib_device = atoi(getenv("IB_DEVICE_ID"));
    }
    LOG(LOG_INFO, "Using IB device: %d.", ib_device);

    #endif //WITH_IB


    if (getenv("CRICKET_DISABLE_RPC")) {
        LOG(LOG_INFO, "RPC server was disable by setting CRICKET_DISABLE_RPC");
        return;
    }
    if (getenv("CRICKET_RESTORE")) {
        LOG(LOG_INFO, "restoring previous state was enabled by setting CRICKET_RESTORE");
            restore = 1;
    }

    if (restore == 1) {
        if (cr_restore_rpc_id("ckp", &prog, &vers) != 0) {
            LOGE(LOG_ERROR, "error while restoring rpc id");
        }
    } else {
        prog = prog_num;
        vers = vers_num;
    }

    LOGE(LOG_DEBUG, "using prog=%d, vers=%d", prog, vers);

#ifndef NO_OPTIMIZATION
#else
    switch (socktype) {
    case UNIX:
        LOG(LOG_INFO, "using UNIX...");
        transp = svcunix_create(RPC_ANYSOCK, 0, 0, CD_SOCKET_PATH);
        if (transp == NULL) {
            LOGE(LOG_ERROR, "cannot create service.");
            exit(1);
        }
        connection_is_local = 1;
        break;
    case TCP:
        LOG(LOG_INFO, "using TCP...");
        transp = svctcp_create(RPC_ANYSOCK, 0, 0);
        if (transp == NULL) {
            LOGE(LOG_ERROR, "cannot create service.");
            exit(1);
        }
        pmap_unset(prog, vers);
        LOG(LOG_INFO, "listening on port %d", transp->xp_port);
        protocol = IPPROTO_TCP;
        break;
    case UDP:
        /* From RPCGEN documentation:
         * Warning: since UDP-based RPC messages can only hold up to 8 Kbytes
         * of encoded data, this transport cannot be used for procedures that
         * take large arguments or return huge results.
         * -> Sounds like UDP does not make sense for CUDA, because we need to
         *    be able to copy large memory chunks
         **/
        LOG(LOG_INFO, "UDP is not supported...");
        break;
    }

    if (!svc_register(transp, prog, vers, rpc_cd_prog_1, protocol)) {
        LOGE(LOG_ERROR, "unable to register (RPC_PROG_PROG, RPC_PROG_VERS).");
        exit(1);
    }
#endif // NO_OPTIMIZATION

    /* Call CUDA initialization function (usually called by __libc_init_main())
     * Address of "_ZL24__sti____cudaRegisterAllv" in static symbol table is e.g. 0x4016c8
     */
    // void (*cudaRegisterAllv)(void) =
    //     (void(*)(void)) elf_symbol_address(NULL, "_ZL24__sti____cudaRegisterAllv");
    // LOG(LOG_INFO, "found CUDA initialization function at %p", cudaRegisterAllv);
    // if (cudaRegisterAllv == NULL) {
    //     LOGE(LOG_WARNING, "could not find cudaRegisterAllv initialization function in cubin. Kernels cannot be launched without it!");
    // } else {
    //     cudaRegisterAllv();
    // }

    sched = &sched_none;
    if (sched->init() != 0) {
        LOGE(LOG_ERROR, "initializing scheduler failed.");
        goto cleanup4;
    }

    if (list_init(&api_records, sizeof(api_record_t)) != 0) {
        LOGE(LOG_ERROR, "initializing api recorder failed.");
        goto cleanup4;
    }

    if (server_runtime_init(restore) != 0) {
        LOGE(LOG_ERROR, "initializing server_runtime failed.");
        goto cleanup3;
    }

    if (server_driver_init(restore) != 0) {
        LOGE(LOG_ERROR, "initializing server_runtime failed.");
        goto cleanup2;        
    }
    
    if (server_nvml_init(restore) != 0) {
        LOGE(LOG_ERROR, "initializing server_nvml failed.");
        goto cleanup1;
    }

    if (server_cudnn_init(restore) != 0) {
        LOGE(LOG_ERROR, "initializing server_nvml failed.");
        goto cleanup0;
    }

#ifdef WITH_IB

    if (ib_init(ib_device, client) != 0) {
        LOG(LOG_ERROR, "initilization of infiniband verbs failed.");
        goto cleanup1;
    }
    
#endif // WITH_IB


    if (signal(SIGUSR1, signal_checkpoint) == SIG_ERR) {
        LOGE(LOG_ERROR, "An error occurred while setting a signal handler.");
        goto cleanup00;
    }

    #if defined(POS_ENABLE)
        pos_cuda_ws = new POSWorkspace_CUDA(argc, argv);
        POS_CHECK_POINTER(pos_cuda_ws);
        pos_cuda_ws->init();
    #endif

    LOG(LOG_INFO, "waiting for RPC requests...");

    svc_run();

    LOG(LOG_DEBUG, "svc_run returned. Cleaning up.");
    ret = 0;
    //api_records_print();
 cleanup00:
    server_cudnn_deinit();
 cleanup0:
    server_driver_deinit();
 cleanup1:
    server_nvml_deinit();
 cleanup2:
    server_runtime_deinit();
 cleanup3:
    api_records_free();
 cleanup4:
 
#if defined(POS_ENABLE)
    if(likely(pos_cuda_ws != nullptr)){
        delete pos_cuda_ws;
    }  
#endif

#ifndef NO_OPTIMIZATION
#else
    pmap_unset(prog, vers);
    svc_destroy(transp);
    unlink(CD_SOCKET_PATH);
#endif

    LOG(LOG_DEBUG, "have a nice day!");
    exit(ret);
}

int rpc_cd_prog_1_freeresult (SVCXPRT * a, xdrproc_t b , caddr_t c)
{
    if (b == (xdrproc_t) xdr_str_result) {
        str_result *res = (str_result*)c;
        if (res->err == 0) {
            free( res->str_result_u.str);
        }
    }
    else if (b == (xdrproc_t) xdr_mem_result) {
        mem_result *res = (mem_result*)c;
        if (res->err == 0) {
            free( (void*)res->mem_result_u.data.mem_data_val);
        }
    }
    return 1;
}

