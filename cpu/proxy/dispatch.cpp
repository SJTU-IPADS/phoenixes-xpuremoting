#include "../cpu_rpc_prot.h"
#include "measurement.h"
#include "xdr_memory.h"
#include <iostream>
#include <memory.h>
#include <netinet/in.h>
#include <rpc/pmap_clnt.h>
#include <rpc/svc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#ifndef SIG_PF
#define SIG_PF void (*)(int)
#endif

static int _rpc_checkpoint_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_checkpoint_1_svc(result, rqstp));
}

static int _rpc_deinit_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_deinit_1_svc(result, rqstp));
}

static int _rpc_printmessage_1(char **argp, void *result, struct svc_req *rqstp)
{
    return (rpc_printmessage_1_svc(*argp, result, rqstp));
}

static int _rpc_dlopen_1(char **argp, void *result, struct svc_req *rqstp)
{
    return (rpc_dlopen_1_svc(*argp, result, rqstp));
}

static int _rpc_register_function_1(rpc_register_function_1_argument *argp,
                                    void *result, struct svc_req *rqstp)
{
    return (rpc_register_function_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                        argp->arg4, argp->arg5, result, rqstp));
}

static int _rpc_elf_load_1(rpc_elf_load_1_argument *argp, void *result,
                           struct svc_req *rqstp)
{
    return (rpc_elf_load_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _rpc_elf_unload_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_elf_unload_1_svc(*argp, result, rqstp));
}

static int _rpc_register_var_1(rpc_register_var_1_argument *argp, void *result,
                               struct svc_req *rqstp)
{
    return (rpc_register_var_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                   argp->arg4, argp->arg5, argp->arg6,
                                   argp->arg7, argp->arg8, result, rqstp));
}

static int _cuda_choose_device_1(mem_data *argp, void *result,
                                 struct svc_req *rqstp)
{
    return (cuda_choose_device_1_svc(*argp, result, rqstp));
}

static int
_cuda_device_get_attribute_1(cuda_device_get_attribute_1_argument *argp,
                             void *result, struct svc_req *rqstp)
{
    return (
        cuda_device_get_attribute_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _cuda_device_get_by_pci_bus_id_1(char **argp, void *result,
                                            struct svc_req *rqstp)
{
    return (cuda_device_get_by_pci_bus_id_1_svc(*argp, result, rqstp));
}

static int _cuda_device_get_cache_config_1(void *argp, void *result,
                                           struct svc_req *rqstp)
{
    return (cuda_device_get_cache_config_1_svc(result, rqstp));
}

static int _cuda_device_get_limit_1(int *argp, void *result,
                                    struct svc_req *rqstp)
{
    return (cuda_device_get_limit_1_svc(*argp, result, rqstp));
}

static int
_cuda_device_get_p2p_attribute_1(cuda_device_get_p2p_attribute_1_argument *argp,
                                 void *result, struct svc_req *rqstp)
{
    return (cuda_device_get_p2p_attribute_1_svc(argp->arg1, argp->arg2,
                                                argp->arg3, result, rqstp));
}

static int
_cuda_device_get_pci_bus_id_1(cuda_device_get_pci_bus_id_1_argument *argp,
                              void *result, struct svc_req *rqstp)
{
    return (cuda_device_get_pci_bus_id_1_svc(argp->arg1, argp->arg2, result,
                                             rqstp));
}

static int _cuda_device_get_shared_mem_config_1(void *argp, void *result,
                                                struct svc_req *rqstp)
{
    return (cuda_device_get_shared_mem_config_1_svc(result, rqstp));
}

static int _cuda_device_get_stream_priority_range_1(void *argp, void *result,
                                                    struct svc_req *rqstp)
{
    return (cuda_device_get_stream_priority_range_1_svc(result, rqstp));
}

static int
_cuda_device_get_texture_lmw_1(cuda_device_get_texture_lmw_1_argument *argp,
                               void *result, struct svc_req *rqstp)
{
    return (cuda_device_get_texture_lmw_1_svc(argp->arg1, argp->arg2, result,
                                              rqstp));
}

static int _cuda_device_reset_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_device_reset_1_svc(result, rqstp));
}

static int _cuda_device_set_cache_config_1(int *argp, void *result,
                                           struct svc_req *rqstp)
{
    return (cuda_device_set_cache_config_1_svc(*argp, result, rqstp));
}

static int _cuda_device_set_limit_1(cuda_device_set_limit_1_argument *argp,
                                    void *result, struct svc_req *rqstp)
{
    return (cuda_device_set_limit_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _cuda_device_set_shared_mem_config_1(int *argp, void *result,
                                                struct svc_req *rqstp)
{
    return (cuda_device_set_shared_mem_config_1_svc(*argp, result, rqstp));
}

static int _cuda_device_synchronize_1(void *argp, void *result,
                                      struct svc_req *rqstp)
{
    return (cuda_device_synchronize_1_svc(result, rqstp));
}

static int _cuda_get_device_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_get_device_1_svc(result, rqstp));
}

static int _cuda_get_device_count_1(void *argp, void *result,
                                    struct svc_req *rqstp)
{
    return (cuda_get_device_count_1_svc(result, rqstp));
}

static int _cuda_get_device_flags_1(void *argp, void *result,
                                    struct svc_req *rqstp)
{
    return (cuda_get_device_flags_1_svc(result, rqstp));
}

static int _cuda_get_device_properties_1(int *argp, void *result,
                                         struct svc_req *rqstp)
{
    return (cuda_get_device_properties_1_svc(*argp, result, rqstp));
}

static int _cuda_set_device_1(int *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_set_device_1_svc(*argp, result, rqstp));
}

static int _cuda_set_device_flags_1(int *argp, void *result,
                                    struct svc_req *rqstp)
{
    return (cuda_set_device_flags_1_svc(*argp, result, rqstp));
}

static int _cuda_set_valid_devices_1(cuda_set_valid_devices_1_argument *argp,
                                     void *result, struct svc_req *rqstp)
{
    return (
        cuda_set_valid_devices_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _cuda_get_error_name_1(int *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (cuda_get_error_name_1_svc(*argp, result, rqstp));
}

static int _cuda_get_error_string_1(int *argp, void *result,
                                    struct svc_req *rqstp)
{
    return (cuda_get_error_string_1_svc(*argp, result, rqstp));
}

static int _cuda_get_last_error_1(void *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (cuda_get_last_error_1_svc(result, rqstp));
}

static int _cuda_peek_at_last_error_1(void *argp, void *result,
                                      struct svc_req *rqstp)
{
    return (cuda_peek_at_last_error_1_svc(result, rqstp));
}

static int _cuda_ctx_reset_persisting_l2cache_1(void *argp, void *result,
                                                struct svc_req *rqstp)
{
    return (cuda_ctx_reset_persisting_l2cache_1_svc(result, rqstp));
}

static int
_cuda_stream_copy_attributes_1(cuda_stream_copy_attributes_1_argument *argp,
                               void *result, struct svc_req *rqstp)
{
    return (cuda_stream_copy_attributes_1_svc(argp->arg1, argp->arg2, result,
                                              rqstp));
}

static int _cuda_stream_create_1(void *argp, void *result,
                                 struct svc_req *rqstp)
{
    return (cuda_stream_create_1_svc(result, rqstp));
}

static int _cuda_stream_create_with_flags_1(int *argp, void *result,
                                            struct svc_req *rqstp)
{
    return (cuda_stream_create_with_flags_1_svc(*argp, result, rqstp));
}

static int _cuda_stream_create_with_priority_1(
    cuda_stream_create_with_priority_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (cuda_stream_create_with_priority_1_svc(argp->arg1, argp->arg2,
                                                   result, rqstp));
}

static int _cuda_stream_destroy_1(ptr *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (cuda_stream_destroy_1_svc(*argp, result, rqstp));
}

static int _cuda_stream_get_flags_1(ptr *argp, void *result,
                                    struct svc_req *rqstp)
{
    return (cuda_stream_get_flags_1_svc(*argp, result, rqstp));
}

static int _cuda_stream_get_priority_1(ptr *argp, void *result,
                                       struct svc_req *rqstp)
{
    return (cuda_stream_get_priority_1_svc(*argp, result, rqstp));
}

static int _cuda_stream_is_capturing_1(ptr *argp, void *result,
                                       struct svc_req *rqstp)
{
    return (cuda_stream_is_capturing_1_svc(*argp, result, rqstp));
}

static int _cuda_stream_query_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_stream_query_1_svc(*argp, result, rqstp));
}

static int _cuda_stream_synchronize_1(ptr *argp, void *result,
                                      struct svc_req *rqstp)
{
    return (cuda_stream_synchronize_1_svc(*argp, result, rqstp));
}

static int _cuda_stream_wait_event_1(cuda_stream_wait_event_1_argument *argp,
                                     void *result, struct svc_req *rqstp)
{
    return (cuda_stream_wait_event_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                         result, rqstp));
}

static int _cuda_thread_exchange_stream_capture_mode_1(int *argp, void *result,
                                                       struct svc_req *rqstp)
{
    return (
        cuda_thread_exchange_stream_capture_mode_1_svc(*argp, result, rqstp));
}

static int _cuda_event_create_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_event_create_1_svc(result, rqstp));
}

static int _cuda_event_create_with_flags_1(int *argp, void *result,
                                           struct svc_req *rqstp)
{
    return (cuda_event_create_with_flags_1_svc(*argp, result, rqstp));
}

static int _cuda_event_destroy_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_event_destroy_1_svc(*argp, result, rqstp));
}

static int _cuda_event_elapsed_time_1(cuda_event_elapsed_time_1_argument *argp,
                                      void *result, struct svc_req *rqstp)
{
    return (
        cuda_event_elapsed_time_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _cuda_event_query_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_event_query_1_svc(*argp, result, rqstp));
}

static int _cuda_event_record_1(cuda_event_record_1_argument *argp,
                                void *result, struct svc_req *rqstp)
{
    return (cuda_event_record_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int
_cuda_event_record_with_flags_1(cuda_event_record_with_flags_1_argument *argp,
                                void *result, struct svc_req *rqstp)
{
    return (cuda_event_record_with_flags_1_svc(argp->arg1, argp->arg2,
                                               argp->arg3, result, rqstp));
}

static int _cuda_event_synchronize_1(ptr *argp, void *result,
                                     struct svc_req *rqstp)
{
    return (cuda_event_synchronize_1_svc(*argp, result, rqstp));
}

static int _cuda_func_get_attributes_1(ptr *argp, void *result,
                                       struct svc_req *rqstp)
{
    return (cuda_func_get_attributes_1_svc(*argp, result, rqstp));
}

static int
_cuda_func_set_attributes_1(cuda_func_set_attributes_1_argument *argp,
                            void *result, struct svc_req *rqstp)
{
    return (cuda_func_set_attributes_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                           result, rqstp));
}

static int
_cuda_func_set_cache_config_1(cuda_func_set_cache_config_1_argument *argp,
                              void *result, struct svc_req *rqstp)
{
    return (cuda_func_set_cache_config_1_svc(argp->arg1, argp->arg2, result,
                                             rqstp));
}

static int _cuda_func_set_shared_mem_config_1(
    cuda_func_set_shared_mem_config_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (cuda_func_set_shared_mem_config_1_svc(argp->arg1, argp->arg2,
                                                  result, rqstp));
}

static int _cuda_launch_cooperative_kernel_1(
    cuda_launch_cooperative_kernel_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (cuda_launch_cooperative_kernel_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        result, rqstp));
}

static int _cuda_launch_kernel_1(cuda_launch_kernel_1_argument *argp,
                                 void *result, struct svc_req *rqstp)
{
    return (cuda_launch_kernel_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                     argp->arg4, argp->arg5, argp->arg6, result,
                                     rqstp));
}

static int _cuda_occupancy_available_dsmpb_1(
    cuda_occupancy_available_dsmpb_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (cuda_occupancy_available_dsmpb_1_svc(argp->arg1, argp->arg2,
                                                 argp->arg3, result, rqstp));
}

static int
_cuda_occupancy_max_active_bpm_1(cuda_occupancy_max_active_bpm_1_argument *argp,
                                 void *result, struct svc_req *rqstp)
{
    return (cuda_occupancy_max_active_bpm_1_svc(argp->arg1, argp->arg2,
                                                argp->arg3, result, rqstp));
}

static int _cuda_occupancy_max_active_bpm_with_flags_1(
    cuda_occupancy_max_active_bpm_with_flags_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (cuda_occupancy_max_active_bpm_with_flags_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, result, rqstp));
}

static int _cuda_array_get_info_1(ptr *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (cuda_array_get_info_1_svc(*argp, result, rqstp));
}

static int _cuda_array_get_sparse_properties_1(ptr *argp, void *result,
                                               struct svc_req *rqstp)
{
    return (cuda_array_get_sparse_properties_1_svc(*argp, result, rqstp));
}

static int _cuda_free_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_free_1_svc(*argp, result, rqstp));
}

static int _cuda_free_array_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_free_array_1_svc(*argp, result, rqstp));
}

static int _cuda_free_host_1(int *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_free_host_1_svc(*argp, result, rqstp));
}

static int _cuda_get_symbol_address_1(ptr *argp, void *result,
                                      struct svc_req *rqstp)
{
    return (cuda_get_symbol_address_1_svc(*argp, result, rqstp));
}

static int _cuda_get_symbol_size_1(ptr *argp, void *result,
                                   struct svc_req *rqstp)
{
    return (cuda_get_symbol_size_1_svc(*argp, result, rqstp));
}

static int _cuda_host_alloc_1(cuda_host_alloc_1_argument *argp, void *result,
                              struct svc_req *rqstp)
{
    return (cuda_host_alloc_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _cuda_host_alloc_regshm_1(cuda_host_alloc_regshm_1_argument *argp,
                                     void *result, struct svc_req *rqstp)
{
    return (
        cuda_host_alloc_regshm_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int
_cuda_host_get_device_pointer_1(cuda_host_get_device_pointer_1_argument *argp,
                                void *result, struct svc_req *rqstp)
{
    return (cuda_host_get_device_pointer_1_svc(argp->arg1, argp->arg2, result,
                                               rqstp));
}

static int _cuda_host_get_flags_1(ptr *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (cuda_host_get_flags_1_svc(*argp, result, rqstp));
}

static int _cuda_malloc_1(size_t *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_malloc_1_svc(*argp, result, rqstp));
}

static int _cuda_malloc_3d_1(cuda_malloc_3d_1_argument *argp, void *result,
                             struct svc_req *rqstp)
{
    return (cuda_malloc_3d_1_svc(argp->arg1, argp->arg2, argp->arg3, result,
                                 rqstp));
}

static int _cuda_malloc_3d_array_1(cuda_malloc_3d_array_1_argument *argp,
                                   void *result, struct svc_req *rqstp)
{
    return (cuda_malloc_3d_array_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                       argp->arg4, argp->arg5, result, rqstp));
}

static int _cuda_malloc_array_1(cuda_malloc_array_1_argument *argp,
                                void *result, struct svc_req *rqstp)
{
    return (cuda_malloc_array_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                    argp->arg4, result, rqstp));
}

static int _cuda_malloc_pitch_1(cuda_malloc_pitch_1_argument *argp,
                                void *result, struct svc_req *rqstp)
{
    return (cuda_malloc_pitch_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _cuda_mem_advise_1(cuda_mem_advise_1_argument *argp, void *result,
                              struct svc_req *rqstp)
{
    return (cuda_mem_advise_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                  argp->arg4, result, rqstp));
}

static int _cuda_mem_get_info_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (cuda_mem_get_info_1_svc(result, rqstp));
}

static int _cuda_mem_prefetch_async_1(cuda_mem_prefetch_async_1_argument *argp,
                                      void *result, struct svc_req *rqstp)
{
    return (cuda_mem_prefetch_async_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                          argp->arg4, result, rqstp));
}

static int _cuda_memcpy_htod_1(cuda_memcpy_htod_1_argument *argp, void *result,
                               struct svc_req *rqstp)
{
    return (cuda_memcpy_htod_1_svc(argp->arg1, argp->arg2, argp->arg3, result,
                                   rqstp));
}

static int _cuda_memcpy_dtoh_1(cuda_memcpy_dtoh_1_argument *argp, void *result,
                               struct svc_req *rqstp)
{
    return (cuda_memcpy_dtoh_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _cuda_memcpy_shm_1(cuda_memcpy_shm_1_argument *argp, void *result,
                              struct svc_req *rqstp)
{
    return (cuda_memcpy_shm_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                  argp->arg4, result, rqstp));
}

static int _cuda_memcpy_dtod_1(cuda_memcpy_dtod_1_argument *argp, void *result,
                               struct svc_req *rqstp)
{
    return (cuda_memcpy_dtod_1_svc(argp->arg1, argp->arg2, argp->arg3, result,
                                   rqstp));
}

static int _cuda_memcpy_to_symbol_1(cuda_memcpy_to_symbol_1_argument *argp,
                                    void *result, struct svc_req *rqstp)
{
    return (cuda_memcpy_to_symbol_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                        argp->arg4, result, rqstp));
}

static int
_cuda_memcpy_to_symbol_shm_1(cuda_memcpy_to_symbol_shm_1_argument *argp,
                             void *result, struct svc_req *rqstp)
{
    return (cuda_memcpy_to_symbol_shm_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                            argp->arg4, argp->arg5, result,
                                            rqstp));
}

static int _cuda_memcpy_ib_1(cuda_memcpy_ib_1_argument *argp, void *result,
                             struct svc_req *rqstp)
{
    return (cuda_memcpy_ib_1_svc(argp->arg1, argp->arg2, argp->arg3, argp->arg4,
                                 result, rqstp));
}

static int _cuda_memcpy_mt_htod_1(cuda_memcpy_mt_htod_1_argument *argp,
                                  void *result, struct svc_req *rqstp)
{
    return (cuda_memcpy_mt_htod_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                      result, rqstp));
}

static int _cuda_memcpy_mt_dtoh_1(cuda_memcpy_mt_dtoh_1_argument *argp,
                                  void *result, struct svc_req *rqstp)
{
    return (cuda_memcpy_mt_dtoh_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                      result, rqstp));
}

static int _cuda_memcpy_mt_sync_1(int *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (cuda_memcpy_mt_sync_1_svc(*argp, result, rqstp));
}

static int _cuda_memset_1(cuda_memset_1_argument *argp, void *result,
                          struct svc_req *rqstp)
{
    return (
        cuda_memset_1_svc(argp->arg1, argp->arg2, argp->arg3, result, rqstp));
}

static int _cuda_memset_2d_1(cuda_memset_2d_1_argument *argp, void *result,
                             struct svc_req *rqstp)
{
    return (cuda_memset_2d_1_svc(argp->arg1, argp->arg2, argp->arg3, argp->arg4,
                                 argp->arg5, result, rqstp));
}

static int _cuda_memset_2d_async_1(cuda_memset_2d_async_1_argument *argp,
                                   void *result, struct svc_req *rqstp)
{
    return (cuda_memset_2d_async_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                       argp->arg4, argp->arg5, argp->arg6,
                                       result, rqstp));
}

static int _cuda_memset_3d_1(cuda_memset_3d_1_argument *argp, void *result,
                             struct svc_req *rqstp)
{
    return (cuda_memset_3d_1_svc(argp->arg1, argp->arg2, argp->arg3, argp->arg4,
                                 argp->arg5, argp->arg6, argp->arg7, argp->arg8,
                                 result, rqstp));
}

static int _cuda_memset_3d_async_1(cuda_memset_3d_async_1_argument *argp,
                                   void *result, struct svc_req *rqstp)
{
    return (cuda_memset_3d_async_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, result, rqstp));
}

static int _cuda_memset_async_1(cuda_memset_async_1_argument *argp,
                                void *result, struct svc_req *rqstp)
{
    return (cuda_memset_async_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                    argp->arg4, result, rqstp));
}

static int
_cuda_device_can_access_peer_1(cuda_device_can_access_peer_1_argument *argp,
                               void *result, struct svc_req *rqstp)
{
    return (cuda_device_can_access_peer_1_svc(argp->arg1, argp->arg2, result,
                                              rqstp));
}

static int _cuda_device_disable_peer_access_1(int *argp, void *result,
                                              struct svc_req *rqstp)
{
    return (cuda_device_disable_peer_access_1_svc(*argp, result, rqstp));
}

static int _cuda_device_enable_peer_access_1(
    cuda_device_enable_peer_access_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (cuda_device_enable_peer_access_1_svc(argp->arg1, argp->arg2, result,
                                                 rqstp));
}

static int _cuda_driver_get_version_1(void *argp, void *result,
                                      struct svc_req *rqstp)
{
    return (cuda_driver_get_version_1_svc(result, rqstp));
}

static int _cuda_runtime_get_version_1(void *argp, void *result,
                                       struct svc_req *rqstp)
{
    return (cuda_runtime_get_version_1_svc(result, rqstp));
}

static int _cuda_profiler_start_1(void *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (cuda_profiler_start_1_svc(result, rqstp));
}

static int _cuda_profiler_stop_1(void *argp, void *result,
                                 struct svc_req *rqstp)
{
    return (cuda_profiler_stop_1_svc(result, rqstp));
}

static int _rpc_cudevicegetcount_1(void *argp, void *result,
                                   struct svc_req *rqstp)
{
    return (rpc_cudevicegetcount_1_svc(result, rqstp));
}

static int _rpc_cuinit_1(int *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cuinit_1_svc(*argp, result, rqstp));
}

static int _rpc_cudrivergetversion_1(void *argp, void *result,
                                     struct svc_req *rqstp)
{
    return (rpc_cudrivergetversion_1_svc(result, rqstp));
}

static int _rpc_cudeviceget_1(int *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cudeviceget_1_svc(*argp, result, rqstp));
}

static int _rpc_cudevicegetname_1(int *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (rpc_cudevicegetname_1_svc(*argp, result, rqstp));
}

static int _rpc_cudevicetotalmem_1(int *argp, void *result,
                                   struct svc_req *rqstp)
{
    return (rpc_cudevicetotalmem_1_svc(*argp, result, rqstp));
}

static int
_rpc_cudevicegetattribute_1(rpc_cudevicegetattribute_1_argument *argp,
                            void *result, struct svc_req *rqstp)
{
    return (
        rpc_cudevicegetattribute_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _rpc_cudevicegetuuid_1(int *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (rpc_cudevicegetuuid_1_svc(*argp, result, rqstp));
}

static int _rpc_cuctxgetcurrent_1(void *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (rpc_cuctxgetcurrent_1_svc(result, rqstp));
}

static int _rpc_cuctxsetcurrent_1(ptr *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (rpc_cuctxsetcurrent_1_svc(*argp, result, rqstp));
}

static int _rpc_cudeviceprimaryctxretain_1(int *argp, void *result,
                                           struct svc_req *rqstp)
{
    return (rpc_cudeviceprimaryctxretain_1_svc(*argp, result, rqstp));
}

static int _rpc_cumodulegetfunction_1(rpc_cumodulegetfunction_1_argument *argp,
                                      void *result, struct svc_req *rqstp)
{
    return (
        rpc_cumodulegetfunction_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _rpc_cumemalloc_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cumemalloc_1_svc(*argp, result, rqstp));
}

static int _rpc_cuctxgetdevice_1(void *argp, void *result,
                                 struct svc_req *rqstp)
{
    return (rpc_cuctxgetdevice_1_svc(result, rqstp));
}

static int _rpc_cumemcpyhtod_1(rpc_cumemcpyhtod_1_argument *argp, void *result,
                               struct svc_req *rqstp)
{
    return (rpc_cumemcpyhtod_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _rpc_culaunchkernel_1(rpc_culaunchkernel_1_argument *argp,
                                 void *result, struct svc_req *rqstp)
{
    return (rpc_culaunchkernel_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, result, rqstp));
}

static int _rpc_cumoduleload_1(char **argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cumoduleload_1_svc(*argp, result, rqstp));
}

static int _rpc_cugeterrorstring_1(int *argp, void *result,
                                   struct svc_req *rqstp)
{
    return (rpc_cugeterrorstring_1_svc(*argp, result, rqstp));
}

static int _rpc_cumoduleunload_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cumoduleunload_1_svc(*argp, result, rqstp));
}

static int _rpc_cudeviceprimaryctxgetstate_1(int *argp, void *result,
                                             struct svc_req *rqstp)
{
    return (rpc_cudeviceprimaryctxgetstate_1_svc(*argp, result, rqstp));
}

static int _rpc_cudevicegetproperties_1(int *argp, void *result,
                                        struct svc_req *rqstp)
{
    return (rpc_cudevicegetproperties_1_svc(*argp, result, rqstp));
}

static int _rpc_cudevicecomputecapability_1(int *argp, void *result,
                                            struct svc_req *rqstp)
{
    return (rpc_cudevicecomputecapability_1_svc(*argp, result, rqstp));
}

static int
_rpc_cudevicegetp2pattribute_1(rpc_cudevicegetp2pattribute_1_argument *argp,
                               void *result, struct svc_req *rqstp)
{
    return (rpc_cudevicegetp2pattribute_1_svc(argp->arg1, argp->arg2,
                                              argp->arg3, result, rqstp));
}

static int _rpc_cusolverdncreate_1(void *argp, void *result,
                                   struct svc_req *rqstp)
{
    return (rpc_cusolverdncreate_1_svc(result, rqstp));
}

static int _rpc_cusolverdnsetstream_1(rpc_cusolverdnsetstream_1_argument *argp,
                                      void *result, struct svc_req *rqstp)
{
    return (
        rpc_cusolverdnsetstream_1_svc(argp->arg1, argp->arg2, result, rqstp));
}

static int _rpc_cusolverdndgetrf_buffersize_1(
    rpc_cusolverdndgetrf_buffersize_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cusolverdndgetrf_buffersize_1_svc(argp->arg1, argp->arg2,
                                                  argp->arg3, argp->arg4,
                                                  argp->arg5, result, rqstp));
}

static int _rpc_cusolverdndgetrf_1(rpc_cusolverdndgetrf_1_argument *argp,
                                   void *result, struct svc_req *rqstp)
{
    return (rpc_cusolverdndgetrf_1_svc(argp->arg1, argp->arg2, argp->arg3,
                                       argp->arg4, argp->arg5, argp->arg6,
                                       argp->arg7, argp->arg8, result, rqstp));
}

static int _rpc_cusolverdndgetrs_1(rpc_cusolverdndgetrs_1_argument *argp,
                                   void *result, struct svc_req *rqstp)
{
    return (rpc_cusolverdndgetrs_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, result, rqstp));
}

static int _rpc_cusolverdndestroy_1(ptr *argp, void *result,
                                    struct svc_req *rqstp)
{
    return (rpc_cusolverdndestroy_1_svc(*argp, result, rqstp));
}

static int _rpc_cublascreate_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cublascreate_1_svc(result, rqstp));
}

static int _rpc_cublasdgemm_1(rpc_cublasdgemm_1_argument *argp, void *result,
                              struct svc_req *rqstp)
{
    return (rpc_cublasdgemm_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, argp->arg11,
        argp->arg12, argp->arg13, argp->arg14, result, rqstp));
}

static int _rpc_cublasdestroy_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cublasdestroy_1_svc(*argp, result, rqstp));
}

static int _rpc_cublassgemm_1(rpc_cublassgemm_1_argument *argp, void *result,
                              struct svc_req *rqstp)
{
    return (rpc_cublassgemm_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, argp->arg11,
        argp->arg12, argp->arg13, argp->arg14, result, rqstp));
}

static int _rpc_cublassgemv_1(rpc_cublassgemv_1_argument *argp, void *result,
                              struct svc_req *rqstp)
{
    return (rpc_cublassgemv_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, argp->arg11,
        argp->arg12, result, rqstp));
}

static int _rpc_cublasdgemv_1(rpc_cublasdgemv_1_argument *argp, void *result,
                              struct svc_req *rqstp)
{
    return (rpc_cublasdgemv_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, argp->arg11,
        argp->arg12, result, rqstp));
}

static int _rpc_cublassgemmex_1(rpc_cublassgemmex_1_argument *argp,
                                void *result, struct svc_req *rqstp)
{
    return (rpc_cublassgemmex_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, argp->arg11,
        argp->arg12, argp->arg13, argp->arg14, argp->arg15, argp->arg16,
        argp->arg17, result, rqstp));
}

static int _rpc_cublassetstream_1(rpc_cublassetstream_1_argument *argp,
                                  void *result, struct svc_req *rqstp)
{
    return (
        rpc_cublassetstream_1_svc(argp->handle, argp->streamId, result, rqstp));
}

static int _rpc_cublassetworkspace_1(rpc_cublassetworkspace_1_argument *argp,
                                     void *result, struct svc_req *rqstp)
{
    return (rpc_cublassetworkspace_1_svc(argp->handle, argp->workspace,
                                         argp->workspaceSizeInBytes, result,
                                         rqstp));
}

static int _rpc_cublassetmathmode_1(rpc_cublassetmathmode_1_argument *argp,
                                    void *result, struct svc_req *rqstp)
{
    return (
        rpc_cublassetmathmode_1_svc(argp->handle, argp->mode, result, rqstp));
}

static int _rpc_nvmldevicegetcount_v2_1(void *argp, void *result,
                                        struct svc_req *rqstp)
{
    return (rpc_nvmldevicegetcount_v2_1_svc(result, rqstp));
}

static int _rpc_nvmlinitwithflags_1(int *argp, void *result,
                                    struct svc_req *rqstp)
{
    return (rpc_nvmlinitwithflags_1_svc(*argp, result, rqstp));
}

static int _rpc_nvmlinit_v2_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_nvmlinit_v2_1_svc(result, rqstp));
}

static int _rpc_nvmlshutdown_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_nvmlshutdown_1_svc(result, rqstp));
}

static int _rpc_cudnngetversion_1(void *argp, void *result,
                                  struct svc_req *rqstp)
{
    return (rpc_cudnngetversion_1_svc(result, rqstp));
}

static int _rpc_cudnngetmaxdeviceversion_1(void *argp, void *result,
                                           struct svc_req *rqstp)
{
    return (rpc_cudnngetmaxdeviceversion_1_svc(result, rqstp));
}

static int _rpc_cudnngetcudartversion_1(void *argp, void *result,
                                        struct svc_req *rqstp)
{
    return (rpc_cudnngetcudartversion_1_svc(result, rqstp));
}

static int _rpc_cudnngeterrorstring_1(int *argp, void *result,
                                      struct svc_req *rqstp)
{
    return (rpc_cudnngeterrorstring_1_svc(*argp, result, rqstp));
}

static int
_rpc_cudnnqueryruntimeerror_1(rpc_cudnnqueryruntimeerror_1_argument *argp,
                              void *result, struct svc_req *rqstp)
{
    return (rpc_cudnnqueryruntimeerror_1_svc(argp->handle, argp->mode, result,
                                             rqstp));
}

static int _rpc_cudnngetproperty_1(int *argp, void *result,
                                   struct svc_req *rqstp)
{
    return (rpc_cudnngetproperty_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnncreate_1(void *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cudnncreate_1_svc(result, rqstp));
}

static int _rpc_cudnndestroy_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cudnndestroy_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnnsetstream_1(rpc_cudnnsetstream_1_argument *argp,
                                 void *result, struct svc_req *rqstp)
{
    return (
        rpc_cudnnsetstream_1_svc(argp->handle, argp->streamId, result, rqstp));
}

static int _rpc_cudnngetstream_1(ptr *argp, void *result, struct svc_req *rqstp)
{
    return (rpc_cudnngetstream_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnncreatetensordescriptor_1(void *argp, void *result,
                                              struct svc_req *rqstp)
{
    return (rpc_cudnncreatetensordescriptor_1_svc(result, rqstp));
}

static int _rpc_cudnnsettensor4ddescriptor_1(
    rpc_cudnnsettensor4ddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsettensor4ddescriptor_1_svc(
        argp->tensorDesc, argp->format, argp->dataType, argp->n, argp->c,
        argp->h, argp->w, result, rqstp));
}

static int _rpc_cudnnsettensor4ddescriptorex_1(
    rpc_cudnnsettensor4ddescriptorex_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsettensor4ddescriptorex_1_svc(
        argp->tensorDesc, argp->dataType, argp->n, argp->c, argp->h, argp->w,
        argp->nStride, argp->cStride, argp->hStride, argp->wStride, result,
        rqstp));
}

static int _rpc_cudnngettensor4ddescriptor_1(ptr *argp, void *result,
                                             struct svc_req *rqstp)
{
    return (rpc_cudnngettensor4ddescriptor_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnnsettensornddescriptor_1(
    rpc_cudnnsettensornddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsettensornddescriptor_1_svc(
        argp->tensorDesc, argp->dataType, argp->nbDims, argp->dimA,
        argp->strideA, result, rqstp));
}

static int _rpc_cudnnsettensornddescriptorex_1(
    rpc_cudnnsettensornddescriptorex_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsettensornddescriptorex_1_svc(
        argp->tensorDesc, argp->format, argp->dataType, argp->nbDims,
        argp->dimA, result, rqstp));
}

static int _rpc_cudnngettensornddescriptor_1(
    rpc_cudnngettensornddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnngettensornddescriptor_1_svc(
        argp->tensorDesc, argp->nbDimsRequested, result, rqstp));
}

static int _rpc_cudnngettensorsizeinbytes_1(ptr *argp, void *result,
                                            struct svc_req *rqstp)
{
    return (rpc_cudnngettensorsizeinbytes_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnndestroytensordescriptor_1(ptr *argp, void *result,
                                               struct svc_req *rqstp)
{
    return (rpc_cudnndestroytensordescriptor_1_svc(*argp, result, rqstp));
}

static int
_rpc_cudnntransformtensor_1(rpc_cudnntransformtensor_1_argument *argp,
                            void *result, struct svc_req *rqstp)
{
    return (rpc_cudnntransformtensor_1_svc(
        argp->handle, argp->alpha, argp->xDesc, argp->x, argp->beta,
        argp->yDesc, argp->y, result, rqstp));
}

static int _rpc_cudnnaddtensor_1(rpc_cudnnaddtensor_1_argument *argp,
                                 void *result, struct svc_req *rqstp)
{
    return (rpc_cudnnaddtensor_1_svc(argp->handle, argp->alpha, argp->aDesc,
                                     argp->A, argp->beta, argp->cDesc, argp->C,
                                     result, rqstp));
}

static int _rpc_cudnncreatefilterdescriptor_1(void *argp, void *result,
                                              struct svc_req *rqstp)
{
    return (rpc_cudnncreatefilterdescriptor_1_svc(result, rqstp));
}

static int _rpc_cudnnsetfilter4ddescriptor_1(
    rpc_cudnnsetfilter4ddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsetfilter4ddescriptor_1_svc(
        argp->filterDesc, argp->dataType, argp->format, argp->k, argp->c,
        argp->h, argp->w, result, rqstp));
}

static int _rpc_cudnngetfilter4ddescriptor_1(ptr *argp, void *result,
                                             struct svc_req *rqstp)
{
    return (rpc_cudnngetfilter4ddescriptor_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnnsetfilternddescriptor_1(
    rpc_cudnnsetfilternddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsetfilternddescriptor_1_svc(
        argp->filterDesc, argp->dataType, argp->format, argp->nbDims,
        argp->filterDimA, result, rqstp));
}

static int _rpc_cudnngetfilternddescriptor_1(
    rpc_cudnngetfilternddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnngetfilternddescriptor_1_svc(
        argp->filterDesc, argp->nbDimsRequested, result, rqstp));
}

static int _rpc_cudnngetfiltersizeinbytes_1(ptr *argp, void *result,
                                            struct svc_req *rqstp)
{
    return (rpc_cudnngetfiltersizeinbytes_1_svc(*argp, result, rqstp));
}

static int
_rpc_cudnntransformfilter_1(rpc_cudnntransformfilter_1_argument *argp,
                            void *result, struct svc_req *rqstp)
{
    return (rpc_cudnntransformfilter_1_svc(
        argp->handle, argp->transDesc, argp->alpha, argp->srcDesc,
        argp->srcData, argp->beta, argp->destDesc, argp->destData, result,
        rqstp));
}

static int _rpc_cudnndestroyfilterdescriptor_1(ptr *argp, void *result,
                                               struct svc_req *rqstp)
{
    return (rpc_cudnndestroyfilterdescriptor_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnnsoftmaxforward_1(rpc_cudnnsoftmaxforward_1_argument *argp,
                                      void *result, struct svc_req *rqstp)
{
    return (rpc_cudnnsoftmaxforward_1_svc(
        argp->handle, argp->algo, argp->mode, argp->alpha, argp->xDesc, argp->x,
        argp->beta, argp->yDesc, argp->y, result, rqstp));
}

static int _rpc_cudnncreatepoolingdescriptor_1(void *argp, void *result,
                                               struct svc_req *rqstp)
{
    return (rpc_cudnncreatepoolingdescriptor_1_svc(result, rqstp));
}

static int _rpc_cudnnsetpooling2ddescriptor_1(
    rpc_cudnnsetpooling2ddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsetpooling2ddescriptor_1_svc(
        argp->poolingDesc, argp->mode, argp->maxpoolingNanOpt,
        argp->windowHeight, argp->windowWidth, argp->verticalPadding,
        argp->horizontalPadding, argp->verticalStride, argp->horizontalStride,
        result, rqstp));
}

static int _rpc_cudnngetpooling2ddescriptor_1(ptr *argp, void *result,
                                              struct svc_req *rqstp)
{
    return (rpc_cudnngetpooling2ddescriptor_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnnsetpoolingnddescriptor_1(
    rpc_cudnnsetpoolingnddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsetpoolingnddescriptor_1_svc(
        argp->poolingDesc, argp->mode, argp->maxpoolingNanOpt, argp->nbDims,
        argp->windowDimA, argp->paddingA, argp->strideA, result, rqstp));
}

static int _rpc_cudnngetpoolingnddescriptor_1(
    rpc_cudnngetpoolingnddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnngetpoolingnddescriptor_1_svc(
        argp->poolingDesc, argp->nbDimsRequested, result, rqstp));
}

static int _rpc_cudnngetpoolingndforwardoutputdim_1(
    rpc_cudnngetpoolingndforwardoutputdim_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnngetpoolingndforwardoutputdim_1_svc(
        argp->poolingDesc, argp->inputTensorDesc, argp->nbDims, result, rqstp));
}

static int _rpc_cudnngetpooling2dforwardoutputdim_1(
    rpc_cudnngetpooling2dforwardoutputdim_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnngetpooling2dforwardoutputdim_1_svc(
        argp->poolingDesc, argp->inputTensorDesc, result, rqstp));
}

static int _rpc_cudnndestroypoolingdescriptor_1(ptr *argp, void *result,
                                                struct svc_req *rqstp)
{
    return (rpc_cudnndestroypoolingdescriptor_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnnpoolingforward_1(rpc_cudnnpoolingforward_1_argument *argp,
                                      void *result, struct svc_req *rqstp)
{
    return (rpc_cudnnpoolingforward_1_svc(
        argp->handle, argp->poolingDesc, argp->alpha, argp->xDesc, argp->x,
        argp->beta, argp->yDesc, argp->y, result, rqstp));
}

static int _rpc_cudnncreateactivationdescriptor_1(void *argp, void *result,
                                                  struct svc_req *rqstp)
{
    return (rpc_cudnncreateactivationdescriptor_1_svc(result, rqstp));
}

static int _rpc_cudnnsetactivationdescriptor_1(
    rpc_cudnnsetactivationdescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsetactivationdescriptor_1_svc(argp->activationDesc,
                                                   argp->mode, argp->reluNanOpt,
                                                   argp->coef, result, rqstp));
}

static int _rpc_cudnngetactivationdescriptor_1(ptr *argp, void *result,
                                               struct svc_req *rqstp)
{
    return (rpc_cudnngetactivationdescriptor_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnnsetactivationdescriptorswishbeta_1(
    rpc_cudnnsetactivationdescriptorswishbeta_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsetactivationdescriptorswishbeta_1_svc(
        argp->activationDesc, argp->swish_beta, result, rqstp));
}

static int _rpc_cudnngetactivationdescriptorswishbeta_1(ptr *argp, void *result,
                                                        struct svc_req *rqstp)
{
    return (
        rpc_cudnngetactivationdescriptorswishbeta_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnndestroyactivationdescriptor_1(ptr *argp, void *result,
                                                   struct svc_req *rqstp)
{
    return (rpc_cudnndestroyactivationdescriptor_1_svc(*argp, result, rqstp));
}

static int
_rpc_cudnnactivationforward_1(rpc_cudnnactivationforward_1_argument *argp,
                              void *result, struct svc_req *rqstp)
{
    return (rpc_cudnnactivationforward_1_svc(
        argp->handle, argp->activationDesc, argp->alpha, argp->xDesc, argp->x,
        argp->beta, argp->yDesc, argp->y, result, rqstp));
}

static int _rpc_cudnncreatelrndescriptor_1(void *argp, void *result,
                                           struct svc_req *rqstp)
{
    return (rpc_cudnncreatelrndescriptor_1_svc(result, rqstp));
}

static int
_rpc_cudnnsetlrndescriptor_1(rpc_cudnnsetlrndescriptor_1_argument *argp,
                             void *result, struct svc_req *rqstp)
{
    return (rpc_cudnnsetlrndescriptor_1_svc(argp->normDesc, argp->lrnN,
                                            argp->lrnAlpha, argp->lrnBeta,
                                            argp->lrnK, result, rqstp));
}

static int _rpc_cudnngetlrndescriptor_1(ptr *argp, void *result,
                                        struct svc_req *rqstp)
{
    return (rpc_cudnngetlrndescriptor_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnndestroylrndescriptor_1(ptr *argp, void *result,
                                            struct svc_req *rqstp)
{
    return (rpc_cudnndestroylrndescriptor_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnnlrncrosschannelforward_1(
    rpc_cudnnlrncrosschannelforward_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnlrncrosschannelforward_1_svc(
        argp->handle, argp->normDesc, argp->lrnMode, argp->alpha, argp->xDesc,
        argp->x, argp->beta, argp->yDesc, argp->y, result, rqstp));
}

static int _rpc_cudnncreateconvolutiondescriptor_1(void *argp, void *result,
                                                   struct svc_req *rqstp)
{
    return (rpc_cudnncreateconvolutiondescriptor_1_svc(result, rqstp));
}

static int _rpc_cudnndestroyconvolutiondescriptor_1(ptr *argp, void *result,
                                                    struct svc_req *rqstp)
{
    return (rpc_cudnndestroyconvolutiondescriptor_1_svc(*argp, result, rqstp));
}

static int _rpc_cudnngetconvolutionndforwardoutputdim_1(
    rpc_cudnngetconvolutionndforwardoutputdim_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnngetconvolutionndforwardoutputdim_1_svc(
        argp->convDesc, argp->inputTensorDesc, argp->filterDesc, argp->nbDims,
        result, rqstp));
}

static int _rpc_cudnnsetconvolutionnddescriptor_1(
    rpc_cudnnsetconvolutionnddescriptor_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsetconvolutionnddescriptor_1_svc(
        argp->convDesc, argp->arrayLength, argp->padA, argp->filterStrideA,
        argp->dilationA, argp->mode, argp->computeType, result, rqstp));
}

static int _rpc_cudnngetconvolutionforwardalgorithm_v7_1(
    rpc_cudnngetconvolutionforwardalgorithm_v7_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnngetconvolutionforwardalgorithm_v7_1_svc(
        argp->handle, argp->srcDesc, argp->filterDesc, argp->convDesc,
        argp->destDesc, argp->requestedAlgoCount, result, rqstp));
}

static int _rpc_cudnnfindconvolutionforwardalgorithm_1(
    rpc_cudnnfindconvolutionforwardalgorithm_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnfindconvolutionforwardalgorithm_1_svc(
        argp->handle, argp->xDesc, argp->wDesc, argp->convDesc, argp->yDesc,
        argp->requestedAlgoCount, result, rqstp));
}

static int _rpc_cudnngetconvolutionforwardworkspacesize_1(
    rpc_cudnngetconvolutionforwardworkspacesize_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnngetconvolutionforwardworkspacesize_1_svc(
        argp->handle, argp->xDesc, argp->wDesc, argp->convDesc, argp->yDesc,
        argp->algo, result, rqstp));
}

static int
_rpc_cudnnconvolutionforward_1(rpc_cudnnconvolutionforward_1_argument *argp,
                               void *result, struct svc_req *rqstp)
{
    return (rpc_cudnnconvolutionforward_1_svc(
        argp->handle, argp->alpha, argp->xDesc, argp->x, argp->wDesc, argp->w,
        argp->convDesc, argp->algo, argp->workSpace, argp->workSpaceSizeInBytes,
        argp->beta, argp->yDesc, argp->y, result, rqstp));
}

static int _rpc_cudnnsetconvolutiongroupcount_1(
    rpc_cudnnsetconvolutiongroupcount_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsetconvolutiongroupcount_1_svc(
        argp->convDesc, argp->groupCnt, result, rqstp));
}

static int _rpc_cudnnsetconvolutionmathtype_1(
    rpc_cudnnsetconvolutionmathtype_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnsetconvolutionmathtype_1_svc(
        argp->convDesc, argp->mathType, result, rqstp));
}

static int _rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1(
    rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1_argument
        *argp,
    void *result, struct svc_req *rqstp)
{
    return (rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1_svc(
        argp->handle, argp->mode, argp->bnOps, argp->xDesc, argp->zDesc,
        argp->yDesc, argp->bnScaleBiasMeanVarDesc, argp->activationDesc, result,
        rqstp));
}

static int _rpc_cudnnbatchnormalizationforwardtrainingex_1(
    rpc_cudnnbatchnormalizationforwardtrainingex_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnbatchnormalizationforwardtrainingex_1_svc(
        argp->handle, argp->mode, argp->bnOps, argp->alpha, argp->beta,
        argp->xDesc, argp->x, argp->zDesc, argp->z, argp->yDesc, argp->y,
        argp->bnScaleBiasMeanVarDesc, argp->bnScaleData, argp->bnBiasData,
        argp->exponentialAverageFactor, argp->resultRunningMeanData,
        argp->resultRunningVarianceData, argp->epsilon, argp->saveMean,
        argp->saveInvVariance, argp->activationDesc, argp->workspace,
        argp->workSpaceSizeInBytes, argp->reserveSpace,
        argp->reserveSpaceSizeInBytes, result, rqstp));
}

static int _rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1(
    rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1_argument *argp,
    void *result, struct svc_req *rqstp)
{
    return (rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1_svc(
        argp->handle, argp->mode, argp->bnOps, argp->xDesc, argp->yDesc,
        argp->dyDesc, argp->dzDesc, argp->dxDesc, argp->dBnScaleBiasDesc,
        argp->activationDesc, result, rqstp));
}

static int _rpc_cudnnbatchnormalizationbackwardex_1(
    rpc_cudnnbatchnormalizationbackwardex_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnbatchnormalizationbackwardex_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, argp->arg11,
        argp->arg12, argp->arg13, argp->arg14, argp->arg15, argp->arg16,
        argp->arg17, argp->arg18, argp->arg19, argp->arg20, argp->arg21,
        argp->arg22, argp->arg23, argp->arg24, argp->arg25, argp->arg26,
        argp->arg27, argp->arg28, argp->arg29, argp->arg30, result, rqstp));
}

static int _rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1(
    rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1_argument *argp,
    void *result, struct svc_req *rqstp)
{
    return (rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        result, rqstp));
}

static int _rpc_cudnnconvolutionbackwarddata_1(
    rpc_cudnnconvolutionbackwarddata_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnconvolutionbackwarddata_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, argp->arg11,
        argp->arg12, argp->arg13, result, rqstp));
}

static int _rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1(
    rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1_argument *argp,
    void *result, struct svc_req *rqstp)
{
    return (rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        result, rqstp));
}

static int _rpc_cudnnconvolutionbackwardfilter_1(
    rpc_cudnnconvolutionbackwardfilter_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnconvolutionbackwardfilter_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, argp->arg11,
        argp->arg12, argp->arg13, result, rqstp));
}

static int _rpc_cudnnbatchnormalizationforwardinference_1(
    rpc_cudnnbatchnormalizationforwardinference_1_argument *argp, void *result,
    struct svc_req *rqstp)
{
    return (rpc_cudnnbatchnormalizationforwardinference_1_svc(
        argp->arg1, argp->arg2, argp->arg3, argp->arg4, argp->arg5, argp->arg6,
        argp->arg7, argp->arg8, argp->arg9, argp->arg10, argp->arg11,
        argp->arg12, argp->arg13, argp->arg14, result, rqstp));
}

extern detailed_info svc_apis[API_COUNT];

XDR *dispatch(int proc_id, XDR **xdrs_arg)
{
    union {
        char *rpc_printmessage_1_arg;
        char *rpc_dlopen_1_arg;
        rpc_register_function_1_argument rpc_register_function_1_arg;
        rpc_elf_load_1_argument rpc_elf_load_1_arg;
        ptr rpc_elf_unload_1_arg;
        rpc_register_var_1_argument rpc_register_var_1_arg;
        mem_data cuda_choose_device_1_arg;
        cuda_device_get_attribute_1_argument cuda_device_get_attribute_1_arg;
        char *cuda_device_get_by_pci_bus_id_1_arg;
        int cuda_device_get_limit_1_arg;
        cuda_device_get_p2p_attribute_1_argument
            cuda_device_get_p2p_attribute_1_arg;
        cuda_device_get_pci_bus_id_1_argument cuda_device_get_pci_bus_id_1_arg;
        cuda_device_get_texture_lmw_1_argument cuda_device_get_texture_lmw_1_arg;
        int cuda_device_set_cache_config_1_arg;
        cuda_device_set_limit_1_argument cuda_device_set_limit_1_arg;
        int cuda_device_set_shared_mem_config_1_arg;
        int cuda_get_device_properties_1_arg;
        int cuda_set_device_1_arg;
        int cuda_set_device_flags_1_arg;
        cuda_set_valid_devices_1_argument cuda_set_valid_devices_1_arg;
        int cuda_get_error_name_1_arg;
        int cuda_get_error_string_1_arg;
        cuda_stream_copy_attributes_1_argument cuda_stream_copy_attributes_1_arg;
        int cuda_stream_create_with_flags_1_arg;
        cuda_stream_create_with_priority_1_argument
            cuda_stream_create_with_priority_1_arg;
        ptr cuda_stream_destroy_1_arg;
        ptr cuda_stream_get_flags_1_arg;
        ptr cuda_stream_get_priority_1_arg;
        ptr cuda_stream_is_capturing_1_arg;
        ptr cuda_stream_query_1_arg;
        ptr cuda_stream_synchronize_1_arg;
        cuda_stream_wait_event_1_argument cuda_stream_wait_event_1_arg;
        int cuda_thread_exchange_stream_capture_mode_1_arg;
        int cuda_event_create_with_flags_1_arg;
        ptr cuda_event_destroy_1_arg;
        cuda_event_elapsed_time_1_argument cuda_event_elapsed_time_1_arg;
        ptr cuda_event_query_1_arg;
        cuda_event_record_1_argument cuda_event_record_1_arg;
        cuda_event_record_with_flags_1_argument
            cuda_event_record_with_flags_1_arg;
        ptr cuda_event_synchronize_1_arg;
        ptr cuda_func_get_attributes_1_arg;
        cuda_func_set_attributes_1_argument cuda_func_set_attributes_1_arg;
        cuda_func_set_cache_config_1_argument cuda_func_set_cache_config_1_arg;
        cuda_func_set_shared_mem_config_1_argument
            cuda_func_set_shared_mem_config_1_arg;
        cuda_launch_cooperative_kernel_1_argument
            cuda_launch_cooperative_kernel_1_arg;
        cuda_launch_kernel_1_argument cuda_launch_kernel_1_arg;
        cuda_occupancy_available_dsmpb_1_argument
            cuda_occupancy_available_dsmpb_1_arg;
        cuda_occupancy_max_active_bpm_1_argument
            cuda_occupancy_max_active_bpm_1_arg;
        cuda_occupancy_max_active_bpm_with_flags_1_argument
            cuda_occupancy_max_active_bpm_with_flags_1_arg;
        ptr cuda_array_get_info_1_arg;
        ptr cuda_array_get_sparse_properties_1_arg;
        ptr cuda_free_1_arg;
        ptr cuda_free_array_1_arg;
        int cuda_free_host_1_arg;
        ptr cuda_get_symbol_address_1_arg;
        ptr cuda_get_symbol_size_1_arg;
        cuda_host_alloc_1_argument cuda_host_alloc_1_arg;
        cuda_host_alloc_regshm_1_argument cuda_host_alloc_regshm_1_arg;
        cuda_host_get_device_pointer_1_argument
            cuda_host_get_device_pointer_1_arg;
        ptr cuda_host_get_flags_1_arg;
        size_t cuda_malloc_1_arg;
        cuda_malloc_3d_1_argument cuda_malloc_3d_1_arg;
        cuda_malloc_3d_array_1_argument cuda_malloc_3d_array_1_arg;
        cuda_malloc_array_1_argument cuda_malloc_array_1_arg;
        cuda_malloc_pitch_1_argument cuda_malloc_pitch_1_arg;
        cuda_mem_advise_1_argument cuda_mem_advise_1_arg;
        cuda_mem_prefetch_async_1_argument cuda_mem_prefetch_async_1_arg;
        cuda_memcpy_htod_1_argument cuda_memcpy_htod_1_arg;
        cuda_memcpy_dtoh_1_argument cuda_memcpy_dtoh_1_arg;
        cuda_memcpy_shm_1_argument cuda_memcpy_shm_1_arg;
        cuda_memcpy_dtod_1_argument cuda_memcpy_dtod_1_arg;
        cuda_memcpy_to_symbol_1_argument cuda_memcpy_to_symbol_1_arg;
        cuda_memcpy_to_symbol_shm_1_argument cuda_memcpy_to_symbol_shm_1_arg;
        cuda_memcpy_ib_1_argument cuda_memcpy_ib_1_arg;
        cuda_memcpy_mt_htod_1_argument cuda_memcpy_mt_htod_1_arg;
        cuda_memcpy_mt_dtoh_1_argument cuda_memcpy_mt_dtoh_1_arg;
        int cuda_memcpy_mt_sync_1_arg;
        cuda_memset_1_argument cuda_memset_1_arg;
        cuda_memset_2d_1_argument cuda_memset_2d_1_arg;
        cuda_memset_2d_async_1_argument cuda_memset_2d_async_1_arg;
        cuda_memset_3d_1_argument cuda_memset_3d_1_arg;
        cuda_memset_3d_async_1_argument cuda_memset_3d_async_1_arg;
        cuda_memset_async_1_argument cuda_memset_async_1_arg;
        cuda_device_can_access_peer_1_argument cuda_device_can_access_peer_1_arg;
        int cuda_device_disable_peer_access_1_arg;
        cuda_device_enable_peer_access_1_argument
            cuda_device_enable_peer_access_1_arg;
        int rpc_cuinit_1_arg;
        int rpc_cudeviceget_1_arg;
        int rpc_cudevicegetname_1_arg;
        int rpc_cudevicetotalmem_1_arg;
        rpc_cudevicegetattribute_1_argument rpc_cudevicegetattribute_1_arg;
        int rpc_cudevicegetuuid_1_arg;
        ptr rpc_cuctxsetcurrent_1_arg;
        int rpc_cudeviceprimaryctxretain_1_arg;
        rpc_cumodulegetfunction_1_argument rpc_cumodulegetfunction_1_arg;
        ptr rpc_cumemalloc_1_arg;
        rpc_cumemcpyhtod_1_argument rpc_cumemcpyhtod_1_arg;
        rpc_culaunchkernel_1_argument rpc_culaunchkernel_1_arg;
        char *rpc_cumoduleload_1_arg;
        int rpc_cugeterrorstring_1_arg;
        ptr rpc_cumoduleunload_1_arg;
        int rpc_cudeviceprimaryctxgetstate_1_arg;
        int rpc_cudevicegetproperties_1_arg;
        int rpc_cudevicecomputecapability_1_arg;
        rpc_cudevicegetp2pattribute_1_argument rpc_cudevicegetp2pattribute_1_arg;
        rpc_cusolverdnsetstream_1_argument rpc_cusolverdnsetstream_1_arg;
        rpc_cusolverdndgetrf_buffersize_1_argument
            rpc_cusolverdndgetrf_buffersize_1_arg;
        rpc_cusolverdndgetrf_1_argument rpc_cusolverdndgetrf_1_arg;
        rpc_cusolverdndgetrs_1_argument rpc_cusolverdndgetrs_1_arg;
        ptr rpc_cusolverdndestroy_1_arg;
        rpc_cublasdgemm_1_argument rpc_cublasdgemm_1_arg;
        ptr rpc_cublasdestroy_1_arg;
        rpc_cublassgemm_1_argument rpc_cublassgemm_1_arg;
        rpc_cublassgemv_1_argument rpc_cublassgemv_1_arg;
        rpc_cublasdgemv_1_argument rpc_cublasdgemv_1_arg;
        rpc_cublassgemmex_1_argument rpc_cublassgemmex_1_arg;
        rpc_cublassetstream_1_argument rpc_cublassetstream_1_arg;
        rpc_cublassetworkspace_1_argument rpc_cublassetworkspace_1_arg;
        rpc_cublassetmathmode_1_argument rpc_cublassetmathmode_1_arg;
        int rpc_nvmlinitwithflags_1_arg;
        int rpc_cudnngeterrorstring_1_arg;
        rpc_cudnnqueryruntimeerror_1_argument rpc_cudnnqueryruntimeerror_1_arg;
        int rpc_cudnngetproperty_1_arg;
        ptr rpc_cudnndestroy_1_arg;
        rpc_cudnnsetstream_1_argument rpc_cudnnsetstream_1_arg;
        ptr rpc_cudnngetstream_1_arg;
        rpc_cudnnsettensor4ddescriptor_1_argument
            rpc_cudnnsettensor4ddescriptor_1_arg;
        rpc_cudnnsettensor4ddescriptorex_1_argument
            rpc_cudnnsettensor4ddescriptorex_1_arg;
        ptr rpc_cudnngettensor4ddescriptor_1_arg;
        rpc_cudnnsettensornddescriptor_1_argument
            rpc_cudnnsettensornddescriptor_1_arg;
        rpc_cudnnsettensornddescriptorex_1_argument
            rpc_cudnnsettensornddescriptorex_1_arg;
        rpc_cudnngettensornddescriptor_1_argument
            rpc_cudnngettensornddescriptor_1_arg;
        ptr rpc_cudnngettensorsizeinbytes_1_arg;
        ptr rpc_cudnndestroytensordescriptor_1_arg;
        rpc_cudnntransformtensor_1_argument rpc_cudnntransformtensor_1_arg;
        rpc_cudnnaddtensor_1_argument rpc_cudnnaddtensor_1_arg;
        rpc_cudnnsetfilter4ddescriptor_1_argument
            rpc_cudnnsetfilter4ddescriptor_1_arg;
        ptr rpc_cudnngetfilter4ddescriptor_1_arg;
        rpc_cudnnsetfilternddescriptor_1_argument
            rpc_cudnnsetfilternddescriptor_1_arg;
        rpc_cudnngetfilternddescriptor_1_argument
            rpc_cudnngetfilternddescriptor_1_arg;
        ptr rpc_cudnngetfiltersizeinbytes_1_arg;
        rpc_cudnntransformfilter_1_argument rpc_cudnntransformfilter_1_arg;
        ptr rpc_cudnndestroyfilterdescriptor_1_arg;
        rpc_cudnnsoftmaxforward_1_argument rpc_cudnnsoftmaxforward_1_arg;
        rpc_cudnnsetpooling2ddescriptor_1_argument
            rpc_cudnnsetpooling2ddescriptor_1_arg;
        ptr rpc_cudnngetpooling2ddescriptor_1_arg;
        rpc_cudnnsetpoolingnddescriptor_1_argument
            rpc_cudnnsetpoolingnddescriptor_1_arg;
        rpc_cudnngetpoolingnddescriptor_1_argument
            rpc_cudnngetpoolingnddescriptor_1_arg;
        rpc_cudnngetpoolingndforwardoutputdim_1_argument
            rpc_cudnngetpoolingndforwardoutputdim_1_arg;
        rpc_cudnngetpooling2dforwardoutputdim_1_argument
            rpc_cudnngetpooling2dforwardoutputdim_1_arg;
        ptr rpc_cudnndestroypoolingdescriptor_1_arg;
        rpc_cudnnpoolingforward_1_argument rpc_cudnnpoolingforward_1_arg;
        rpc_cudnnsetactivationdescriptor_1_argument
            rpc_cudnnsetactivationdescriptor_1_arg;
        ptr rpc_cudnngetactivationdescriptor_1_arg;
        rpc_cudnnsetactivationdescriptorswishbeta_1_argument
            rpc_cudnnsetactivationdescriptorswishbeta_1_arg;
        ptr rpc_cudnngetactivationdescriptorswishbeta_1_arg;
        ptr rpc_cudnndestroyactivationdescriptor_1_arg;
        rpc_cudnnactivationforward_1_argument rpc_cudnnactivationforward_1_arg;
        rpc_cudnnsetlrndescriptor_1_argument rpc_cudnnsetlrndescriptor_1_arg;
        ptr rpc_cudnngetlrndescriptor_1_arg;
        ptr rpc_cudnndestroylrndescriptor_1_arg;
        rpc_cudnnlrncrosschannelforward_1_argument
            rpc_cudnnlrncrosschannelforward_1_arg;
        ptr rpc_cudnndestroyconvolutiondescriptor_1_arg;
        rpc_cudnngetconvolutionndforwardoutputdim_1_argument
            rpc_cudnngetconvolutionndforwardoutputdim_1_arg;
        rpc_cudnnsetconvolutionnddescriptor_1_argument
            rpc_cudnnsetconvolutionnddescriptor_1_arg;
        rpc_cudnngetconvolutionforwardalgorithm_v7_1_argument
            rpc_cudnngetconvolutionforwardalgorithm_v7_1_arg;
        rpc_cudnnfindconvolutionforwardalgorithm_1_argument
            rpc_cudnnfindconvolutionforwardalgorithm_1_arg;
        rpc_cudnngetconvolutionforwardworkspacesize_1_argument
            rpc_cudnngetconvolutionforwardworkspacesize_1_arg;
        rpc_cudnnconvolutionforward_1_argument rpc_cudnnconvolutionforward_1_arg;
        rpc_cudnnsetconvolutiongroupcount_1_argument
            rpc_cudnnsetconvolutiongroupcount_1_arg;
        rpc_cudnnsetconvolutionmathtype_1_argument
            rpc_cudnnsetconvolutionmathtype_1_arg;
        rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1_argument
            rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1_arg;
        rpc_cudnnbatchnormalizationforwardtrainingex_1_argument
            rpc_cudnnbatchnormalizationforwardtrainingex_1_arg;
        rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1_argument
            rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1_arg;
        rpc_cudnnbatchnormalizationbackwardex_1_argument
            rpc_cudnnbatchnormalizationbackwardex_1_arg;
        rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1_argument
            rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1_arg;
        rpc_cudnnconvolutionbackwarddata_1_argument
            rpc_cudnnconvolutionbackwarddata_1_arg;
        rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1_argument
            rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1_arg;
        rpc_cudnnconvolutionbackwardfilter_1_argument
            rpc_cudnnconvolutionbackwardfilter_1_arg;
        rpc_cudnnbatchnormalizationforwardinference_1_argument
            rpc_cudnnbatchnormalizationforwardinference_1_arg;
    } argument;
    union {
        int rpc_checkpoint_1_res;
        int rpc_deinit_1_res;
        int rpc_printmessage_1_res;
        int rpc_dlopen_1_res;
        ptr_result rpc_register_function_1_res;
        int rpc_elf_load_1_res;
        int rpc_elf_unload_1_res;
        int rpc_register_var_1_res;
        int_result cuda_choose_device_1_res;
        int_result cuda_device_get_attribute_1_res;
        int_result cuda_device_get_by_pci_bus_id_1_res;
        int_result cuda_device_get_cache_config_1_res;
        u64_result cuda_device_get_limit_1_res;
        int_result cuda_device_get_p2p_attribute_1_res;
        str_result cuda_device_get_pci_bus_id_1_res;
        int_result cuda_device_get_shared_mem_config_1_res;
        dint_result cuda_device_get_stream_priority_range_1_res;
        u64_result cuda_device_get_texture_lmw_1_res;
        int cuda_device_reset_1_res;
        int cuda_device_set_cache_config_1_res;
        int cuda_device_set_limit_1_res;
        int cuda_device_set_shared_mem_config_1_res;
        int cuda_device_synchronize_1_res;
        int_result cuda_get_device_1_res;
        int_result cuda_get_device_count_1_res;
        int_result cuda_get_device_flags_1_res;
        cuda_device_prop_result cuda_get_device_properties_1_res;
        int cuda_set_device_1_res;
        int cuda_set_device_flags_1_res;
        int cuda_set_valid_devices_1_res;
        str_result cuda_get_error_name_1_res;
        str_result cuda_get_error_string_1_res;
        int cuda_get_last_error_1_res;
        int cuda_peek_at_last_error_1_res;
        int cuda_ctx_reset_persisting_l2cache_1_res;
        int cuda_stream_copy_attributes_1_res;
        ptr_result cuda_stream_create_1_res;
        ptr_result cuda_stream_create_with_flags_1_res;
        ptr_result cuda_stream_create_with_priority_1_res;
        int cuda_stream_destroy_1_res;
        int_result cuda_stream_get_flags_1_res;
        int_result cuda_stream_get_priority_1_res;
        int_result cuda_stream_is_capturing_1_res;
        int cuda_stream_query_1_res;
        int cuda_stream_synchronize_1_res;
        int cuda_stream_wait_event_1_res;
        int_result cuda_thread_exchange_stream_capture_mode_1_res;
        ptr_result cuda_event_create_1_res;
        ptr_result cuda_event_create_with_flags_1_res;
        int cuda_event_destroy_1_res;
        float_result cuda_event_elapsed_time_1_res;
        int cuda_event_query_1_res;
        int cuda_event_record_1_res;
        int cuda_event_record_with_flags_1_res;
        int cuda_event_synchronize_1_res;
        mem_result cuda_func_get_attributes_1_res;
        int cuda_func_set_attributes_1_res;
        int cuda_func_set_cache_config_1_res;
        int cuda_func_set_shared_mem_config_1_res;
        int cuda_launch_cooperative_kernel_1_res;
        int cuda_launch_kernel_1_res;
        u64_result cuda_occupancy_available_dsmpb_1_res;
        int_result cuda_occupancy_max_active_bpm_1_res;
        int_result cuda_occupancy_max_active_bpm_with_flags_1_res;
        mem_result cuda_array_get_info_1_res;
        mem_result cuda_array_get_sparse_properties_1_res;
        int cuda_free_1_res;
        int cuda_free_array_1_res;
        int cuda_free_host_1_res;
        ptr_result cuda_get_symbol_address_1_res;
        u64_result cuda_get_symbol_size_1_res;
        sz_result cuda_host_alloc_1_res;
        int cuda_host_alloc_regshm_1_res;
        ptr_result cuda_host_get_device_pointer_1_res;
        int_result cuda_host_get_flags_1_res;
        ptr_result cuda_malloc_1_res;
        pptr_result cuda_malloc_3d_1_res;
        ptr_result cuda_malloc_3d_array_1_res;
        ptr_result cuda_malloc_array_1_res;
        ptrsz_result cuda_malloc_pitch_1_res;
        int cuda_mem_advise_1_res;
        dsz_result cuda_mem_get_info_1_res;
        int cuda_mem_prefetch_async_1_res;
        int cuda_memcpy_htod_1_res;
        mem_result cuda_memcpy_dtoh_1_res;
        int cuda_memcpy_shm_1_res;
        int cuda_memcpy_dtod_1_res;
        int cuda_memcpy_to_symbol_1_res;
        int cuda_memcpy_to_symbol_shm_1_res;
        int cuda_memcpy_ib_1_res;
        dint_result cuda_memcpy_mt_htod_1_res;
        dint_result cuda_memcpy_mt_dtoh_1_res;
        int cuda_memcpy_mt_sync_1_res;
        int cuda_memset_1_res;
        int cuda_memset_2d_1_res;
        int cuda_memset_2d_async_1_res;
        int cuda_memset_3d_1_res;
        int cuda_memset_3d_async_1_res;
        int cuda_memset_async_1_res;
        int_result cuda_device_can_access_peer_1_res;
        int cuda_device_disable_peer_access_1_res;
        int cuda_device_enable_peer_access_1_res;
        int_result cuda_driver_get_version_1_res;
        int_result cuda_runtime_get_version_1_res;
        int cuda_profiler_start_1_res;
        int cuda_profiler_stop_1_res;
        int_result rpc_cudevicegetcount_1_res;
        int rpc_cuinit_1_res;
        int_result rpc_cudrivergetversion_1_res;
        int_result rpc_cudeviceget_1_res;
        str_result rpc_cudevicegetname_1_res;
        u64_result rpc_cudevicetotalmem_1_res;
        int_result rpc_cudevicegetattribute_1_res;
        str_result rpc_cudevicegetuuid_1_res;
        ptr_result rpc_cuctxgetcurrent_1_res;
        int rpc_cuctxsetcurrent_1_res;
        ptr_result rpc_cudeviceprimaryctxretain_1_res;
        ptr_result rpc_cumodulegetfunction_1_res;
        ptr_result rpc_cumemalloc_1_res;
        int_result rpc_cuctxgetdevice_1_res;
        int rpc_cumemcpyhtod_1_res;
        int rpc_culaunchkernel_1_res;
        ptr_result rpc_cumoduleload_1_res;
        str_result rpc_cugeterrorstring_1_res;
        int rpc_cumoduleunload_1_res;
        dint_result rpc_cudeviceprimaryctxgetstate_1_res;
        mem_result rpc_cudevicegetproperties_1_res;
        dint_result rpc_cudevicecomputecapability_1_res;
        int_result rpc_cudevicegetp2pattribute_1_res;
        ptr_result rpc_cusolverdncreate_1_res;
        int rpc_cusolverdnsetstream_1_res;
        int_result rpc_cusolverdndgetrf_buffersize_1_res;
        int rpc_cusolverdndgetrf_1_res;
        int rpc_cusolverdndgetrs_1_res;
        int rpc_cusolverdndestroy_1_res;
        ptr_result rpc_cublascreate_1_res;
        int rpc_cublasdgemm_1_res;
        int rpc_cublasdestroy_1_res;
        int rpc_cublassgemm_1_res;
        int rpc_cublassgemv_1_res;
        int rpc_cublasdgemv_1_res;
        int rpc_cublassgemmex_1_res;
        int rpc_cublassetstream_1_res;
        int rpc_cublassetworkspace_1_res;
        int rpc_cublassetmathmode_1_res;
        int_result rpc_nvmldevicegetcount_v2_1_res;
        int rpc_nvmlinitwithflags_1_res;
        int rpc_nvmlinit_v2_1_res;
        int rpc_nvmlshutdown_1_res;
        size_t rpc_cudnngetversion_1_res;
        size_t rpc_cudnngetmaxdeviceversion_1_res;
        size_t rpc_cudnngetcudartversion_1_res;
        char *rpc_cudnngeterrorstring_1_res;
        int_result rpc_cudnnqueryruntimeerror_1_res;
        int_result rpc_cudnngetproperty_1_res;
        ptr_result rpc_cudnncreate_1_res;
        int rpc_cudnndestroy_1_res;
        int rpc_cudnnsetstream_1_res;
        ptr_result rpc_cudnngetstream_1_res;
        ptr_result rpc_cudnncreatetensordescriptor_1_res;
        int rpc_cudnnsettensor4ddescriptor_1_res;
        int rpc_cudnnsettensor4ddescriptorex_1_res;
        int9_result rpc_cudnngettensor4ddescriptor_1_res;
        int rpc_cudnnsettensornddescriptor_1_res;
        int rpc_cudnnsettensornddescriptorex_1_res;
        mem_result rpc_cudnngettensornddescriptor_1_res;
        sz_result rpc_cudnngettensorsizeinbytes_1_res;
        int rpc_cudnndestroytensordescriptor_1_res;
        int rpc_cudnntransformtensor_1_res;
        int rpc_cudnnaddtensor_1_res;
        ptr_result rpc_cudnncreatefilterdescriptor_1_res;
        int rpc_cudnnsetfilter4ddescriptor_1_res;
        int6_result rpc_cudnngetfilter4ddescriptor_1_res;
        int rpc_cudnnsetfilternddescriptor_1_res;
        mem_result rpc_cudnngetfilternddescriptor_1_res;
        sz_result rpc_cudnngetfiltersizeinbytes_1_res;
        int rpc_cudnntransformfilter_1_res;
        int rpc_cudnndestroyfilterdescriptor_1_res;
        int rpc_cudnnsoftmaxforward_1_res;
        ptr_result rpc_cudnncreatepoolingdescriptor_1_res;
        int rpc_cudnnsetpooling2ddescriptor_1_res;
        int8_result rpc_cudnngetpooling2ddescriptor_1_res;
        int rpc_cudnnsetpoolingnddescriptor_1_res;
        mem_result rpc_cudnngetpoolingnddescriptor_1_res;
        mem_result rpc_cudnngetpoolingndforwardoutputdim_1_res;
        int4_result rpc_cudnngetpooling2dforwardoutputdim_1_res;
        int rpc_cudnndestroypoolingdescriptor_1_res;
        int rpc_cudnnpoolingforward_1_res;
        ptr_result rpc_cudnncreateactivationdescriptor_1_res;
        int rpc_cudnnsetactivationdescriptor_1_res;
        int2d1_result rpc_cudnngetactivationdescriptor_1_res;
        int rpc_cudnnsetactivationdescriptorswishbeta_1_res;
        d_result rpc_cudnngetactivationdescriptorswishbeta_1_res;
        int rpc_cudnndestroyactivationdescriptor_1_res;
        int rpc_cudnnactivationforward_1_res;
        ptr_result rpc_cudnncreatelrndescriptor_1_res;
        int rpc_cudnnsetlrndescriptor_1_res;
        int1d3_result rpc_cudnngetlrndescriptor_1_res;
        int rpc_cudnndestroylrndescriptor_1_res;
        int rpc_cudnnlrncrosschannelforward_1_res;
        ptr_result rpc_cudnncreateconvolutiondescriptor_1_res;
        int rpc_cudnndestroyconvolutiondescriptor_1_res;
        mem_result rpc_cudnngetconvolutionndforwardoutputdim_1_res;
        int rpc_cudnnsetconvolutionnddescriptor_1_res;
        mem_result rpc_cudnngetconvolutionforwardalgorithm_v7_1_res;
        mem_result rpc_cudnnfindconvolutionforwardalgorithm_1_res;
        sz_result rpc_cudnngetconvolutionforwardworkspacesize_1_res;
        int rpc_cudnnconvolutionforward_1_res;
        int rpc_cudnnsetconvolutiongroupcount_1_res;
        int rpc_cudnnsetconvolutionmathtype_1_res;
        sz_result
            rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1_res;
        int rpc_cudnnbatchnormalizationforwardtrainingex_1_res;
        sz_result rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1_res;
        int rpc_cudnnbatchnormalizationbackwardex_1_res;
        mem_result rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1_res;
        int rpc_cudnnconvolutionbackwarddata_1_res;
        mem_result rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1_res;
        int rpc_cudnnconvolutionbackwardfilter_1_res;
        int rpc_cudnnbatchnormalizationforwardinference_1_res;
    } result;
    bool_t retval;
    xdrproc_t _xdr_argument, _xdr_result;
    bool_t (*local)(char *, void *, struct svc_req *);

    switch (proc_id) {
    case rpc_checkpoint:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_checkpoint_1;
        break;

    case rpc_deinit:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_deinit_1;
        break;

    case rpc_printmessage:
        _xdr_argument = (xdrproc_t)xdr_wrapstring;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_printmessage_1;
        break;

    case rpc_dlopen:
        _xdr_argument = (xdrproc_t)xdr_wrapstring;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_dlopen_1;
        break;

    case rpc_register_function:
        _xdr_argument = (xdrproc_t)xdr_rpc_register_function_1_argument;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_register_function_1;
        break;

    case rpc_elf_load:
        _xdr_argument = (xdrproc_t)xdr_rpc_elf_load_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_elf_load_1;
        break;

    case rpc_elf_unload:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_elf_unload_1;
        break;

    case rpc_register_var:
        _xdr_argument = (xdrproc_t)xdr_rpc_register_var_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_register_var_1;
        break;

    case CUDA_CHOOSE_DEVICE:
        _xdr_argument = (xdrproc_t)xdr_mem_data;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_choose_device_1;
        break;

    case CUDA_DEVICE_GET_ATTRIBUTE:
        _xdr_argument = (xdrproc_t)xdr_cuda_device_get_attribute_1_argument;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_get_attribute_1;
        break;

    case CUDA_DEVICE_GET_BY_PCI_BUS_ID:
        _xdr_argument = (xdrproc_t)xdr_wrapstring;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_get_by_pci_bus_id_1;
        break;

    case CUDA_DEVICE_GET_CACHE_CONFIG:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_get_cache_config_1;
        break;

    case CUDA_DEVICE_GET_LIMIT:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_u64_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_get_limit_1;
        break;

    case CUDA_DEVICE_GET_P2P_ATTRIBUTE:
        _xdr_argument = (xdrproc_t)xdr_cuda_device_get_p2p_attribute_1_argument;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_get_p2p_attribute_1;
        break;

    case CUDA_DEVICE_GET_PCI_BUS_ID:
        _xdr_argument = (xdrproc_t)xdr_cuda_device_get_pci_bus_id_1_argument;
        _xdr_result = (xdrproc_t)xdr_str_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_get_pci_bus_id_1;
        break;

    case CUDA_DEVICE_GET_SHARED_MEM_CONFIG:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_cuda_device_get_shared_mem_config_1;
        break;

    case CUDA_DEVICE_GET_STREAM_PRIORITY_RANGE:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_dint_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _cuda_device_get_stream_priority_range_1;
        break;

    case CUDA_DEVICE_GET_TEXTURE_LMW:
        _xdr_argument = (xdrproc_t)xdr_cuda_device_get_texture_lmw_1_argument;
        _xdr_result = (xdrproc_t)xdr_u64_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_get_texture_lmw_1;
        break;

    case CUDA_DEVICE_RESET:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_device_reset_1;
        break;

    case CUDA_DEVICE_SET_CACHE_CONFIG:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_set_cache_config_1;
        break;

    case CUDA_DEVICE_SET_LIMIT:
        _xdr_argument = (xdrproc_t)xdr_cuda_device_set_limit_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_set_limit_1;
        break;

    case CUDA_DEVICE_SET_SHARED_MEM_CONFIG:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_cuda_device_set_shared_mem_config_1;
        break;

    case CUDA_DEVICE_SYNCHRONIZE:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_synchronize_1;
        break;

    case CUDA_GET_DEVICE:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_get_device_1;
        break;

    case CUDA_GET_DEVICE_COUNT:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_get_device_count_1;
        break;

    case CUDA_GET_DEVICE_FLAGS:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_get_device_flags_1;
        break;

    case CUDA_GET_DEVICE_PROPERTIES:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_cuda_device_prop_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_get_device_properties_1;
        break;

    case CUDA_SET_DEVICE:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_set_device_1;
        break;

    case CUDA_SET_DEVICE_FLAGS:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_set_device_flags_1;
        break;

    case CUDA_SET_VALID_DEVICES:
        _xdr_argument = (xdrproc_t)xdr_cuda_set_valid_devices_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_set_valid_devices_1;
        break;

    case CUDA_GET_ERROR_NAME:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_str_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_get_error_name_1;
        break;

    case CUDA_GET_ERROR_STRING:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_str_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_get_error_string_1;
        break;

    case CUDA_GET_LAST_ERROR:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_get_last_error_1;
        break;

    case CUDA_PEEK_AT_LAST_ERROR:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_peek_at_last_error_1;
        break;

    case CUDA_CTX_RESET_PERSISTING_L2CACHE:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_cuda_ctx_reset_persisting_l2cache_1;
        break;

    case CUDA_STREAM_COPY_ATTRIBUTES:
        _xdr_argument = (xdrproc_t)xdr_cuda_stream_copy_attributes_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_stream_copy_attributes_1;
        break;

    case CUDA_STREAM_CREATE:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_stream_create_1;
        break;

    case CUDA_STREAM_CREATE_WITH_FLAGS:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_stream_create_with_flags_1;
        break;

    case CUDA_STREAM_CREATE_WITH_PRIORITY:
        _xdr_argument =
            (xdrproc_t)xdr_cuda_stream_create_with_priority_1_argument;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_cuda_stream_create_with_priority_1;
        break;

    case CUDA_STREAM_DESTROY:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_stream_destroy_1;
        break;

    case CUDA_STREAM_GET_FLAGS:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_stream_get_flags_1;
        break;

    case CUDA_STREAM_GET_PRIORITY:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_stream_get_priority_1;
        break;

    case CUDA_STREAM_IS_CAPTURING:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_stream_is_capturing_1;
        break;

    case CUDA_STREAM_QUERY:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_stream_query_1;
        break;

    case CUDA_STREAM_SYNCHRONIZE:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_stream_synchronize_1;
        break;

    case CUDA_STREAM_WAIT_EVENT:
        _xdr_argument = (xdrproc_t)xdr_cuda_stream_wait_event_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_stream_wait_event_1;
        break;

    case CUDA_THREAD_EXCHANGE_STREAM_CAPTURE_MODE:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _cuda_thread_exchange_stream_capture_mode_1;
        break;

    case CUDA_EVENT_CREATE:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_event_create_1;
        break;

    case CUDA_EVENT_CREATE_WITH_FLAGS:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_event_create_with_flags_1;
        break;

    case CUDA_EVENT_DESTROY:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_event_destroy_1;
        break;

    case CUDA_EVENT_ELAPSED_TIME:
        _xdr_argument = (xdrproc_t)xdr_cuda_event_elapsed_time_1_argument;
        _xdr_result = (xdrproc_t)xdr_float_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_event_elapsed_time_1;
        break;

    case CUDA_EVENT_QUERY:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_event_query_1;
        break;

    case CUDA_EVENT_RECORD:
        _xdr_argument = (xdrproc_t)xdr_cuda_event_record_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_event_record_1;
        break;

    case CUDA_EVENT_RECORD_WITH_FLAGS:
        _xdr_argument = (xdrproc_t)xdr_cuda_event_record_with_flags_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_event_record_with_flags_1;
        break;

    case CUDA_EVENT_SYNCHRONIZE:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_event_synchronize_1;
        break;

    case CUDA_FUNC_GET_ATTRIBUTES:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_func_get_attributes_1;
        break;

    case CUDA_FUNC_SET_ATTRIBUTES:
        _xdr_argument = (xdrproc_t)xdr_cuda_func_set_attributes_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_func_set_attributes_1;
        break;

    case CUDA_FUNC_SET_CACHE_CONFIG:
        _xdr_argument = (xdrproc_t)xdr_cuda_func_set_cache_config_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_func_set_cache_config_1;
        break;

    case CUDA_FUNC_SET_SHARED_MEM_CONFIG:
        _xdr_argument =
            (xdrproc_t)xdr_cuda_func_set_shared_mem_config_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_func_set_shared_mem_config_1;
        break;

    case CUDA_LAUNCH_COOPERATIVE_KERNEL:
        _xdr_argument =
            (xdrproc_t)xdr_cuda_launch_cooperative_kernel_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_launch_cooperative_kernel_1;
        break;

    case CUDA_LAUNCH_KERNEL:
        _xdr_argument = (xdrproc_t)xdr_cuda_launch_kernel_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_launch_kernel_1;
        break;

    case CUDA_OCCUPANCY_AVAILABLE_DSMPB:
        _xdr_argument =
            (xdrproc_t)xdr_cuda_occupancy_available_dsmpb_1_argument;
        _xdr_result = (xdrproc_t)xdr_u64_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_occupancy_available_dsmpb_1;
        break;

    case CUDA_OCCUPANCY_MAX_ACTIVE_BPM:
        _xdr_argument = (xdrproc_t)xdr_cuda_occupancy_max_active_bpm_1_argument;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_occupancy_max_active_bpm_1;
        break;

    case CUDA_OCCUPANCY_MAX_ACTIVE_BPM_WITH_FLAGS:
        _xdr_argument =
            (xdrproc_t)xdr_cuda_occupancy_max_active_bpm_with_flags_1_argument;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _cuda_occupancy_max_active_bpm_with_flags_1;
        break;

    case CUDA_ARRAY_GET_INFO:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_array_get_info_1;
        break;

    case CUDA_ARRAY_GET_SPARSE_PROPERTIES:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_cuda_array_get_sparse_properties_1;
        break;

    case CUDA_FREE:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_free_1;
        break;

    case CUDA_FREE_ARRAY:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_free_array_1;
        break;

    case CUDA_FREE_HOST:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_free_host_1;
        break;

    case CUDA_GET_SYMBOL_ADDRESS:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_get_symbol_address_1;
        break;

    case CUDA_GET_SYMBOL_SIZE:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_u64_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_get_symbol_size_1;
        break;

    case CUDA_HOST_ALLOC:
        _xdr_argument = (xdrproc_t)xdr_cuda_host_alloc_1_argument;
        _xdr_result = (xdrproc_t)xdr_sz_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_host_alloc_1;
        break;

    case CUDA_HOST_ALLOC_REGSHM:
        _xdr_argument = (xdrproc_t)xdr_cuda_host_alloc_regshm_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_host_alloc_regshm_1;
        break;

    case CUDA_HOST_GET_DEVICE_POINTER:
        _xdr_argument = (xdrproc_t)xdr_cuda_host_get_device_pointer_1_argument;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_host_get_device_pointer_1;
        break;

    case CUDA_HOST_GET_FLAGS:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_host_get_flags_1;
        break;

    case CUDA_MALLOC:
        _xdr_argument = (xdrproc_t)xdr_size_t;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_malloc_1;
        break;

    case CUDA_MALLOC_3D:
        _xdr_argument = (xdrproc_t)xdr_cuda_malloc_3d_1_argument;
        _xdr_result = (xdrproc_t)xdr_pptr_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_malloc_3d_1;
        break;

    case CUDA_MALLOC_3D_ARRAY:
        _xdr_argument = (xdrproc_t)xdr_cuda_malloc_3d_array_1_argument;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_malloc_3d_array_1;
        break;

    case CUDA_MALLOC_ARRAY:
        _xdr_argument = (xdrproc_t)xdr_cuda_malloc_array_1_argument;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_malloc_array_1;
        break;

    case CUDA_MALLOC_PITCH:
        _xdr_argument = (xdrproc_t)xdr_cuda_malloc_pitch_1_argument;
        _xdr_result = (xdrproc_t)xdr_ptrsz_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_malloc_pitch_1;
        break;

    case CUDA_MEM_ADVISE:
        _xdr_argument = (xdrproc_t)xdr_cuda_mem_advise_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_mem_advise_1;
        break;

    case CUDA_MEM_GET_INFO:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_dsz_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_mem_get_info_1;
        break;

    case CUDA_MEM_PREFETCH_ASYNC:
        _xdr_argument = (xdrproc_t)xdr_cuda_mem_prefetch_async_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_mem_prefetch_async_1;
        break;

    case CUDA_MEMCPY_HTOD:
        _xdr_argument = (xdrproc_t)xdr_cuda_memcpy_htod_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_memcpy_htod_1;
        break;

    case CUDA_MEMCPY_DTOH:
        _xdr_argument = (xdrproc_t)xdr_cuda_memcpy_dtoh_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_memcpy_dtoh_1;
        break;

    case CUDA_MEMCPY_SHM:
        _xdr_argument = (xdrproc_t)xdr_cuda_memcpy_shm_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_memcpy_shm_1;
        break;

    case CUDA_MEMCPY_DTOD:
        _xdr_argument = (xdrproc_t)xdr_cuda_memcpy_dtod_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_memcpy_dtod_1;
        break;

    case CUDA_MEMCPY_TO_SYMBOL:
        _xdr_argument = (xdrproc_t)xdr_cuda_memcpy_to_symbol_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_memcpy_to_symbol_1;
        break;

    case CUDA_MEMCPY_TO_SYMBOL_SHM:
        _xdr_argument = (xdrproc_t)xdr_cuda_memcpy_to_symbol_shm_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_memcpy_to_symbol_shm_1;
        break;

    case CUDA_MEMCPY_IB:
        _xdr_argument = (xdrproc_t)xdr_cuda_memcpy_ib_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_memcpy_ib_1;
        break;

    case CUDA_MEMCPY_MT_HTOD:
        _xdr_argument = (xdrproc_t)xdr_cuda_memcpy_mt_htod_1_argument;
        _xdr_result = (xdrproc_t)xdr_dint_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_memcpy_mt_htod_1;
        break;

    case CUDA_MEMCPY_MT_DTOH:
        _xdr_argument = (xdrproc_t)xdr_cuda_memcpy_mt_dtoh_1_argument;
        _xdr_result = (xdrproc_t)xdr_dint_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_memcpy_mt_dtoh_1;
        break;

    case CUDA_MEMCPY_MT_SYNC:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_memcpy_mt_sync_1;
        break;

    case CUDA_MEMSET:
        _xdr_argument = (xdrproc_t)xdr_cuda_memset_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_memset_1;
        break;

    case CUDA_MEMSET_2D:
        _xdr_argument = (xdrproc_t)xdr_cuda_memset_2d_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_memset_2d_1;
        break;

    case CUDA_MEMSET_2D_ASYNC:
        _xdr_argument = (xdrproc_t)xdr_cuda_memset_2d_async_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_memset_2d_async_1;
        break;

    case CUDA_MEMSET_3D:
        _xdr_argument = (xdrproc_t)xdr_cuda_memset_3d_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_cuda_memset_3d_1;
        break;

    case CUDA_MEMSET_3D_ASYNC:
        _xdr_argument = (xdrproc_t)xdr_cuda_memset_3d_async_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_memset_3d_async_1;
        break;

    case CUDA_MEMSET_ASYNC:
        _xdr_argument = (xdrproc_t)xdr_cuda_memset_async_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_memset_async_1;
        break;

    case CUDA_DEVICE_CAN_ACCESS_PEER:
        _xdr_argument = (xdrproc_t)xdr_cuda_device_can_access_peer_1_argument;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_can_access_peer_1;
        break;

    case CUDA_DEVICE_DISABLE_PEER_ACCESS:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_disable_peer_access_1;
        break;

    case CUDA_DEVICE_ENABLE_PEER_ACCESS:
        _xdr_argument =
            (xdrproc_t)xdr_cuda_device_enable_peer_access_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_device_enable_peer_access_1;
        break;

    case CUDA_DRIVER_GET_VERSION:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_driver_get_version_1;
        break;

    case CUDA_RUNTIME_GET_VERSION:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_cuda_runtime_get_version_1;
        break;

    case CUDA_PROFILER_START:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_profiler_start_1;
        break;

    case CUDA_PROFILER_STOP:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_cuda_profiler_stop_1;
        break;

    case rpc_cuDeviceGetCount:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudevicegetcount_1;
        break;

    case rpc_cuInit:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_cuinit_1;
        break;

    case rpc_cuDriverGetVersion:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudrivergetversion_1;
        break;

    case rpc_cuDeviceGet:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_cudeviceget_1;
        break;

    case rpc_cuDeviceGetName:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_str_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cudevicegetname_1;
        break;

    case rpc_cuDeviceTotalMem:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_u64_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudevicetotalmem_1;
        break;

    case rpc_cuDeviceGetAttribute:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudevicegetattribute_1_argument;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudevicegetattribute_1;
        break;

    case rpc_cuDeviceGetUuid:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_str_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cudevicegetuuid_1;
        break;

    case rpc_cuCtxGetCurrent:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cuctxgetcurrent_1;
        break;

    case rpc_cuCtxSetCurrent:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cuctxsetcurrent_1;
        break;

    case rpc_cuDevicePrimaryCtxRetain:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudeviceprimaryctxretain_1;
        break;

    case rpc_cuModuleGetFunction:
        _xdr_argument = (xdrproc_t)xdr_rpc_cumodulegetfunction_1_argument;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cumodulegetfunction_1;
        break;

    case rpc_cuMemAlloc:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_cumemalloc_1;
        break;

    case rpc_cuCtxGetDevice:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cuctxgetdevice_1;
        break;

    case rpc_cuMemcpyHtoD:
        _xdr_argument = (xdrproc_t)xdr_rpc_cumemcpyhtod_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cumemcpyhtod_1;
        break;

    case rpc_cuLaunchKernel:
        _xdr_argument = (xdrproc_t)xdr_rpc_culaunchkernel_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_culaunchkernel_1;
        break;

    case rpc_cuModuleLoad:
        _xdr_argument = (xdrproc_t)xdr_wrapstring;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cumoduleload_1;
        break;

    case rpc_cuGetErrorString:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_str_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cugeterrorstring_1;
        break;

    case rpc_cuModuleUnload:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cumoduleunload_1;
        break;

    case rpc_cuDevicePrimaryCtxGetState:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_dint_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudeviceprimaryctxgetstate_1;
        break;

    case rpc_cuDeviceGetProperties:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudevicegetproperties_1;
        break;

    case rpc_cuDeviceComputeCapability:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_dint_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudevicecomputecapability_1;
        break;

    case rpc_cuDeviceGetP2PAttribute:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudevicegetp2pattribute_1_argument;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudevicegetp2pattribute_1;
        break;

    case rpc_cusolverDnCreate:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cusolverdncreate_1;
        break;

    case rpc_cusolverDnSetStream:
        _xdr_argument = (xdrproc_t)xdr_rpc_cusolverdnsetstream_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cusolverdnsetstream_1;
        break;

    case rpc_cusolverDnDgetrf_bufferSize:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cusolverdndgetrf_buffersize_1_argument;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cusolverdndgetrf_buffersize_1;
        break;

    case rpc_cusolverDnDgetrf:
        _xdr_argument = (xdrproc_t)xdr_rpc_cusolverdndgetrf_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cusolverdndgetrf_1;
        break;

    case rpc_cusolverDnDgetrs:
        _xdr_argument = (xdrproc_t)xdr_rpc_cusolverdndgetrs_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cusolverdndgetrs_1;
        break;

    case rpc_cusolverDnDestroy:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cusolverdndestroy_1;
        break;

    case rpc_cublasCreate:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cublascreate_1;
        break;

    case rpc_cublasDgemm:
        _xdr_argument = (xdrproc_t)xdr_rpc_cublasdgemm_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_cublasdgemm_1;
        break;

    case rpc_cublasDestroy:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cublasdestroy_1;
        break;

    case rpc_cublasSgemm:
        _xdr_argument = (xdrproc_t)xdr_rpc_cublassgemm_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_cublassgemm_1;
        break;

    case rpc_cublasSgemv:
        _xdr_argument = (xdrproc_t)xdr_rpc_cublassgemv_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_cublassgemv_1;
        break;

    case rpc_cublasDgemv:
        _xdr_argument = (xdrproc_t)xdr_rpc_cublasdgemv_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_cublasdgemv_1;
        break;

    case rpc_cublasSgemmEx:
        _xdr_argument = (xdrproc_t)xdr_rpc_cublassgemmex_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cublassgemmex_1;
        break;

    case rpc_cublasSetStream:
        _xdr_argument = (xdrproc_t)xdr_rpc_cublassetstream_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cublassetstream_1;
        break;

    case rpc_cublasSetWorkspace:
        _xdr_argument = (xdrproc_t)xdr_rpc_cublassetworkspace_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cublassetworkspace_1;
        break;

    case rpc_cublasSetMathMode:
        _xdr_argument = (xdrproc_t)xdr_rpc_cublassetmathmode_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cublassetmathmode_1;
        break;

    case rpc_nvmlDeviceGetCount_v2:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_nvmldevicegetcount_v2_1;
        break;

    case rpc_nvmlInitWithFlags:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_nvmlinitwithflags_1;
        break;

    case rpc_nvmlInit_v2:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_nvmlinit_v2_1;
        break;

    case rpc_nvmlShutdown:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_nvmlshutdown_1;
        break;

    case rpc_cudnnGetVersion:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_size_t;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cudnngetversion_1;
        break;

    case rpc_cudnnGetMaxDeviceVersion:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_size_t;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngetmaxdeviceversion_1;
        break;

    case rpc_cudnnGetCudartVersion:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_size_t;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngetcudartversion_1;
        break;

    case rpc_cudnnGetErrorString:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_wrapstring;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngeterrorstring_1;
        break;

    case rpc_cudnnQueryRuntimeError:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnnqueryruntimeerror_1_argument;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnqueryruntimeerror_1;
        break;

    case rpc_cudnnGetProperty:
        _xdr_argument = (xdrproc_t)xdr_int;
        _xdr_result = (xdrproc_t)xdr_int_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngetproperty_1;
        break;

    case rpc_cudnnCreate:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))_rpc_cudnncreate_1;
        break;

    case rpc_cudnnDestroy:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cudnndestroy_1;
        break;

    case rpc_cudnnSetStream:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnnsetstream_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cudnnsetstream_1;
        break;

    case rpc_cudnnGetStream:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cudnngetstream_1;
        break;

    case rpc_cudnnCreateTensorDescriptor:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnncreatetensordescriptor_1;
        break;

    case rpc_cudnnSetTensor4dDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsettensor4ddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnsettensor4ddescriptor_1;
        break;

    case rpc_cudnnSetTensor4dDescriptorEx:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsettensor4ddescriptorex_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnnsettensor4ddescriptorex_1;
        break;

    case rpc_cudnnGetTensor4dDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int9_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngettensor4ddescriptor_1;
        break;

    case rpc_cudnnSetTensorNdDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsettensornddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnsettensornddescriptor_1;
        break;

    case rpc_cudnnSetTensorNdDescriptorEx:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsettensornddescriptorex_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnnsettensornddescriptorex_1;
        break;

    case rpc_cudnnGetTensorNdDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnngettensornddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngettensornddescriptor_1;
        break;

    case rpc_cudnnGetTensorSizeInBytes:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_sz_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngettensorsizeinbytes_1;
        break;

    case rpc_cudnnDestroyTensorDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnndestroytensordescriptor_1;
        break;

    case rpc_cudnnTransformTensor:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnntransformtensor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnntransformtensor_1;
        break;

    case rpc_cudnnAddTensor:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnnaddtensor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *, struct svc_req *))_rpc_cudnnaddtensor_1;
        break;

    case rpc_cudnnCreateFilterDescriptor:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnncreatefilterdescriptor_1;
        break;

    case rpc_cudnnSetFilter4dDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsetfilter4ddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnsetfilter4ddescriptor_1;
        break;

    case rpc_cudnnGetFilter4dDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int6_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngetfilter4ddescriptor_1;
        break;

    case rpc_cudnnSetFilterNdDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsetfilternddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnsetfilternddescriptor_1;
        break;

    case rpc_cudnnGetFilterNdDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnngetfilternddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngetfilternddescriptor_1;
        break;

    case rpc_cudnnGetFilterSizeInBytes:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_sz_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngetfiltersizeinbytes_1;
        break;

    case rpc_cudnnTransformFilter:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnntransformfilter_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnntransformfilter_1;
        break;

    case rpc_cudnnDestroyFilterDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnndestroyfilterdescriptor_1;
        break;

    case rpc_cudnnSoftmaxForward:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnnsoftmaxforward_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnsoftmaxforward_1;
        break;

    case rpc_cudnnCreatePoolingDescriptor:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnncreatepoolingdescriptor_1;
        break;

    case rpc_cudnnSetPooling2dDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsetpooling2ddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnsetpooling2ddescriptor_1;
        break;

    case rpc_cudnnGetPooling2dDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int8_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngetpooling2ddescriptor_1;
        break;

    case rpc_cudnnSetPoolingNdDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsetpoolingnddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnsetpoolingnddescriptor_1;
        break;

    case rpc_cudnnGetPoolingNdDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnngetpoolingnddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngetpoolingnddescriptor_1;
        break;

    case rpc_cudnnGetPoolingNdForwardOutputDim:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnngetpoolingndforwardoutputdim_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetpoolingndforwardoutputdim_1;
        break;

    case rpc_cudnnGetPooling2dForwardOutputDim:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnngetpooling2dforwardoutputdim_1_argument;
        _xdr_result = (xdrproc_t)xdr_int4_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetpooling2dforwardoutputdim_1;
        break;

    case rpc_cudnnDestroyPoolingDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnndestroypoolingdescriptor_1;
        break;

    case rpc_cudnnPoolingForward:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnnpoolingforward_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnpoolingforward_1;
        break;

    case rpc_cudnnCreateActivationDescriptor:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnncreateactivationdescriptor_1;
        break;

    case rpc_cudnnSetActivationDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsetactivationdescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnnsetactivationdescriptor_1;
        break;

    case rpc_cudnnGetActivationDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int2d1_result;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnngetactivationdescriptor_1;
        break;

    case rpc_cudnnSetActivationDescriptorSwishBeta:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsetactivationdescriptorswishbeta_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnnsetactivationdescriptorswishbeta_1;
        break;

    case rpc_cudnnGetActivationDescriptorSwishBeta:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_d_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetactivationdescriptorswishbeta_1;
        break;

    case rpc_cudnnDestroyActivationDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnndestroyactivationdescriptor_1;
        break;

    case rpc_cudnnActivationForward:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnnactivationforward_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnactivationforward_1;
        break;

    case rpc_cudnnCreateLRNDescriptor:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnncreatelrndescriptor_1;
        break;

    case rpc_cudnnSetLRNDescriptor:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnnsetlrndescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnsetlrndescriptor_1;
        break;

    case rpc_cudnnGetLRNDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int1d3_result;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnngetlrndescriptor_1;
        break;

    case rpc_cudnnDestroyLRNDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnndestroylrndescriptor_1;
        break;

    case rpc_cudnnLRNCrossChannelForward:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnlrncrosschannelforward_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnlrncrosschannelforward_1;
        break;

    case rpc_cudnnCreateConvolutionDescriptor:
        _xdr_argument = (xdrproc_t)xdr_void;
        _xdr_result = (xdrproc_t)xdr_ptr_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnncreateconvolutiondescriptor_1;
        break;

    case rpc_cudnnDestroyConvolutionDescriptor:
        _xdr_argument = (xdrproc_t)xdr_ptr;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnndestroyconvolutiondescriptor_1;
        break;

    case rpc_cudnnGetConvolutionNdForwardOutputDim:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnngetconvolutionndforwardoutputdim_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetconvolutionndforwardoutputdim_1;
        break;

    case rpc_cudnnSetConvolutionNdDescriptor:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsetconvolutionnddescriptor_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnnsetconvolutionnddescriptor_1;
        break;

    case rpc_cudnnGetConvolutionForwardAlgorithm_v7:
        _xdr_argument = (xdrproc_t)
            xdr_rpc_cudnngetconvolutionforwardalgorithm_v7_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetconvolutionforwardalgorithm_v7_1;
        break;

    case rpc_cudnnFindConvolutionForwardAlgorithm:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnfindconvolutionforwardalgorithm_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnnfindconvolutionforwardalgorithm_1;
        break;

    case rpc_cudnnGetConvolutionForwardWorkspaceSize:
        _xdr_argument = (xdrproc_t)
            xdr_rpc_cudnngetconvolutionforwardworkspacesize_1_argument;
        _xdr_result = (xdrproc_t)xdr_sz_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetconvolutionforwardworkspacesize_1;
        break;

    case rpc_cudnnConvolutionForward:
        _xdr_argument = (xdrproc_t)xdr_rpc_cudnnconvolutionforward_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnconvolutionforward_1;
        break;

    case rpc_cudnnSetConvolutionGroupCount:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsetconvolutiongroupcount_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnnsetconvolutiongroupcount_1;
        break;

    case rpc_cudnnSetConvolutionMathType:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnsetconvolutionmathtype_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *,
                           struct svc_req *))_rpc_cudnnsetconvolutionmathtype_1;
        break;

    case rpc_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize:
        _xdr_argument = (xdrproc_t)
            xdr_rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1_argument;
        _xdr_result = (xdrproc_t)xdr_sz_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1;
        break;

    case rpc_cudnnBatchNormalizationForwardTrainingEx:
        _xdr_argument = (xdrproc_t)
            xdr_rpc_cudnnbatchnormalizationforwardtrainingex_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnnbatchnormalizationforwardtrainingex_1;
        break;

    case rpc_cudnnGetBatchNormalizationBackwardExWorkspaceSize:
        _xdr_argument = (xdrproc_t)
            xdr_rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1_argument;
        _xdr_result = (xdrproc_t)xdr_sz_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1;
        break;

    case rpc_cudnnBatchNormalizationBackwardEx:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnbatchnormalizationbackwardex_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnnbatchnormalizationbackwardex_1;
        break;

    case rpc_cudnnGetConvolutionBackwardDataAlgorithm_v7:
        _xdr_argument = (xdrproc_t)
            xdr_rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1;
        break;

    case rpc_cudnnConvolutionBackwardData:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnconvolutionbackwarddata_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnnconvolutionbackwarddata_1;
        break;

    case rpc_cudnnGetConvolutionBackwardFilterAlgorithm_v7:
        _xdr_argument = (xdrproc_t)
            xdr_rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1_argument;
        _xdr_result = (xdrproc_t)xdr_mem_result;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1;
        break;

    case rpc_cudnnConvolutionBackwardFilter:
        _xdr_argument =
            (xdrproc_t)xdr_rpc_cudnnconvolutionbackwardfilter_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local =
            (bool_t(*)(char *, void *,
                       struct svc_req *))_rpc_cudnnconvolutionbackwardfilter_1;
        break;

    case rpc_cudnnBatchNormalizationForwardInference:
        _xdr_argument = (xdrproc_t)
            xdr_rpc_cudnnbatchnormalizationforwardinference_1_argument;
        _xdr_result = (xdrproc_t)xdr_int;
        local = (bool_t(*)(char *, void *, struct svc_req *))
            _rpc_cudnnbatchnormalizationforwardinference_1;
        break;

    default:
        // svcerr_noproc(transp);
        return;
    }

    time_start(svc_apis, proc_id, SERIALIZATION_TIME);
    memset((char *)&argument, 0, sizeof(argument));
    (*_xdr_argument)(*xdrs_arg, (caddr_t)&argument);
    time_end(svc_apis, proc_id, SERIALIZATION_TIME);

    struct svc_req *rqstp = (struct svc_req *)malloc(sizeof(struct svc_req));
    rqstp->rq_prog = rqstp->rq_vers = 0;
    rqstp->rq_proc = proc_id;

    retval = (bool_t)(*local)((char *)&argument, (void *)&result, rqstp);
    XDR *xdrs_res = new_xdrmemory(XDR_ENCODE);
    if (retval > 0) {
        time_start(svc_apis, proc_id, SERIALIZATION_TIME);
        (*_xdr_result)(xdrs_res, (caddr_t)&result);
        time_end(svc_apis, proc_id, SERIALIZATION_TIME);
    } else {
        std::cout << "Local call failed." << std::endl;
        exit(1);
    }

    (*xdrs_arg)->x_op = XDR_FREE;
    if (!(*_xdr_argument)(*xdrs_arg, (caddr_t)&argument)) {
        std::cout << "Unable to free arguments." << std::endl;
        exit(1);
    }
    destroy_xdrmemory(xdrs_arg);

    if (!rpc_cd_prog_1_freeresult(NULL, _xdr_result, (caddr_t)&result)) {
        std::cout << "Unable to free results." << std::endl;
        exit(1);
    }

    return xdrs_res;
}
