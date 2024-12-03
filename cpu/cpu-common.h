#ifndef _CD_COMMON_H_
#define _CD_COMMON_H_

#include <rpc/rpc.h>
#include <unordered_map>
#include <iostream>
#include "list.h"

#define CD_SOCKET_PATH "/tmp/cricketd_sock"
#ifndef LOG_LEVEL
    #define LOG_LEVEL LOG_ERROR
#endif //LOG_LEVEL

#define CHECK_CU(cmd) do {                      \
  CUresult ret = (cmd); \
  if(ret != 0) { \
    LOGE(LOG_ERROR, "error in %s: %d", #cmd, ret); \
    return 0; \
  } \
} while(0)

typedef struct kernel_info {
    char *name;
    size_t param_size;
    size_t param_num;
    uint16_t *param_offsets;
    uint16_t *param_sizes;
    void *host_fun;
} kernel_info_t;

extern enum socktype_t {UNIX, TCP, UDP} socktype;
#define INIT_SOCKTYPE enum socktype_t socktype = TCP;

extern int connection_is_local;
extern int shm_enabled;
//#define INIT_SOCKTYPE enum socktype_t socktype = UNIX;
#define WITH_API_CNT
//#define WITH_IB


extern CLIENT *clnt;
extern std::unordered_map<std::string, kernel_info_t*> name_to_kernel_infos;
extern std::unordered_map<void*, kernel_info_t*> func_ptr_to_kernel_infos;
static kernel_info_t* find_kernel(const void* func) {
    auto it = func_ptr_to_kernel_infos.find(func);
    if (it == func_ptr_to_kernel_infos.end()) {
        return NULL;
    } else {
        return it->second;
    }
}
static kernel_info_t* find_kernel(const std::string& name) {
    auto it = name_to_kernel_infos.find(name);
    if (it == name_to_kernel_infos.end()) {
        return NULL;
    } else {
        return it->second;
    }
}
static void add_kernel(const void* func, kernel_info_t* info) {
    func_ptr_to_kernel_infos[(void*) func] = info;
}
static void add_kernel(const std::string& name, kernel_info_t* info) {
    name_to_kernel_infos[name] = info;
}

#endif //_CD_COMMON_H_

