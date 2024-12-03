#include <rpc/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

//for strerror
#include <string.h>
#include <errno.h>

#include "cpu_rpc_prot.h"
#include "cpu-server.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "resource-mg.h"
#define WITH_RECORDER
#include "api-recorder.h"
#include "cpu-server-cublas.h"
#include "gsched.h"
#include "cpu-measurement.h"

extern cpu_measurement_info vanillas[CPU_API_COUNT];


static unsigned long long Hash(const void* ptr, size_t size)
{
    static unsigned p = 19260817;
    unsigned long long hash = 0;
    for (size_t i=0; i < size; ++i) {
        hash = hash * p + ((char*)ptr)[i];
    }
    return hash;
}

int cublas_init(int bypass, resource_mg *memory)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cublas, bypass);
    return ret;
}

resource_mg *cublas_get_rm(void)
{
    return &rm_cublas;
}

int cublas_deinit(void)
{
    resource_mg_free(&rm_cublas);
    return 0;

}

bool_t rpc_cublascreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
#ifdef POS_ENABLE

    result->err = pos_cuda_ws->pos_process( 
        /* api_id */ rpc_cublasCreate, 
        /* uuid */ 0, 
        /* param_desps */ {},
        /* ret_data */ &(result->ptr_result_u.ptr),
        /* ret_data_len */ sizeof(cublasHandle_t)
    );

#else // POS_ENABLE

    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cublasCreate_v2");

    GSCHED_RETAIN;
    int proc = 3001;
    cpu_time_start(vanillas, proc);
    result->err = cublasCreate_v2((cublasHandle_t*)&result->ptr_result_u.ptr);
    resource_mg_create(&rm_cublas, (void*)result->ptr_result_u.ptr);
    cpu_time_end(vanillas, proc);
    GSCHED_RELEASE;
    
    RECORD_RESULT(ptr_result_u, *result);

#endif // POS_ENABLE

    return 1;
}

bool_t rpc_cublasdgemm_1_svc(ptr handle, int transa, int transb, int m, int n, int k, double alpha,
            ptr A, int lda,
            ptr B, int ldb, double beta,
            ptr C, int ldc,
            int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublasdgemm_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, transa);
    RECORD_ARG(3, transb);
    RECORD_ARG(4, m);
    RECORD_ARG(5, n);
    RECORD_ARG(6, k);
    RECORD_ARG(7, alpha);
    RECORD_ARG(8, A);
    RECORD_ARG(9, lda);
    RECORD_ARG(10, B);
    RECORD_ARG(11, ldb);
    RECORD_ARG(12, beta);
    RECORD_ARG(13, C);
    RECORD_ARG(14, ldc);
    LOGE(LOG_DEBUG, "cublasDgemm");
    GSCHED_RETAIN;
    *result = cublasDgemm(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) transa,
                    (cublasOperation_t) transb,
                    m, n, k, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), lda,
                    resource_mg_get(&rm_memory, (void*)B), ldb, &beta,
                    resource_mg_get(&rm_memory, (void*)C), ldc
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublasdestroy_1_svc(ptr handle, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(handle);
    LOGE(LOG_DEBUG, "cublasDestroy_v2");
    GSCHED_RETAIN;
    *result = cublasDestroy_v2(resource_mg_get(&rm_cublas, (void*)handle));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassgemm_1_svc(ptr handle, int transa, int transb, int m, int n, int k, float alpha,
            ptr A, int lda,
            ptr B, int ldb, float beta,
            ptr C, int ldc,
            int *result, struct svc_req *rqstp)
{
#ifdef POS_ENABLE

    *result = pos_cuda_ws->pos_process( 
        /* api_id */ rpc_cublasSgemm, 
        /* uuid */ 0, 
        /* param_desps */ {
            { .value = &handle, .size = sizeof(ptr) },
            { .value = &transa, .size = sizeof(cublasOperation_t) },
            { .value = &transb, .size = sizeof(cublasOperation_t) },
            { .value = &m, .size = sizeof(int) },
            { .value = &n, .size = sizeof(int) },
            { .value = &k, .size = sizeof(int) },
            { .value = &alpha, .size = sizeof(float) },
            { .value = &A, .size = sizeof(ptr) },
            { .value = &lda, .size = sizeof(int) },
            { .value = &B, .size = sizeof(ptr) },
            { .value = &ldb, .size = sizeof(int) },
            { .value = &beta, .size = sizeof(float) },
            { .value = &C, .size = sizeof(ptr) },
            { .value = &ldc, .size = sizeof(int) },
        }
    );

#else // POS_ENABLE

    RECORD_API(rpc_cublassgemm_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, transa);
    RECORD_ARG(3, transb);
    RECORD_ARG(4, m);
    RECORD_ARG(5, n);
    RECORD_ARG(6, k);
    RECORD_ARG(7, alpha);
    RECORD_ARG(8, A);
    RECORD_ARG(9, lda);
    RECORD_ARG(10, B);
    RECORD_ARG(11, ldb);
    RECORD_ARG(12, beta);
    RECORD_ARG(13, C);
    RECORD_ARG(14, ldc);
    LOGE(LOG_DEBUG, "cublasSgemm");
    GSCHED_RETAIN;
    int proc = 3004;
    cpu_time_start(vanillas, proc);
    *result = cublasSgemm(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) transa,
                    (cublasOperation_t) transb,
                    m, n, k, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), lda,
                    resource_mg_get(&rm_memory, (void*)B), ldb, &beta,
                    resource_mg_get(&rm_memory, (void*)C), ldc
    );
    cpu_time_end(vanillas, proc);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);

#endif // POS_ENABLE

    return 1;
}

bool_t rpc_cublassgemv_1_svc(ptr handle, int trans, int m, 
            int n, float alpha,
            ptr A, int lda,
            ptr x, int incx, float beta,
            ptr y, int incy,
            int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublassgemv_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, trans);
    RECORD_ARG(3, m);
    RECORD_ARG(4, n);
    RECORD_ARG(5, alpha);
    RECORD_ARG(6, A);
    RECORD_ARG(7, lda);
    RECORD_ARG(8, x);
    RECORD_ARG(9, incx);
    RECORD_ARG(10, beta);
    RECORD_ARG(11, y);
    RECORD_ARG(12, incy);
    LOGE(LOG_DEBUG, "cublasSgemv");
    GSCHED_RETAIN;
    *result = cublasSgemv(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) trans,
                    m, n, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), lda,
                    resource_mg_get(&rm_memory, (void*)x), incx, &beta,
                    resource_mg_get(&rm_memory, (void*)y), incy
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublasdgemv_1_svc(ptr handle, int trans, int m, 
            int n, double alpha,
            ptr A, int lda,
            ptr x, int incx, double beta,
            ptr y, int incy,
            int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublasdgemv_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, trans);
    RECORD_ARG(3, m);
    RECORD_ARG(4, n);
    RECORD_ARG(5, alpha);
    RECORD_ARG(6, A);
    RECORD_ARG(7, lda);
    RECORD_ARG(8, x);
    RECORD_ARG(9, incx);
    RECORD_ARG(10, beta);
    RECORD_ARG(11, y);
    RECORD_ARG(12, incy);
    LOGE(LOG_DEBUG, "cublasDgemv");
    GSCHED_RETAIN;
    *result = cublasDgemv(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) trans,
                    m, n, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), lda,
                    resource_mg_get(&rm_memory, (void*)x), incx, &beta,
                    resource_mg_get(&rm_memory, (void*)y), incy
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassgemmex_1_svc(ptr handle, int transa, int transb, int m, int n, int k, float alpha,
            ptr A, int Atype, int lda,
            ptr B, int Btype, int ldb, float beta,
            ptr C, int Ctype, int ldc,
            int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublassgemmex_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, transa);
    RECORD_ARG(3, transb);
    RECORD_ARG(4, m);
    RECORD_ARG(5, n);
    RECORD_ARG(6, k);
    RECORD_ARG(7, alpha);
    RECORD_ARG(8, A);
    RECORD_ARG(9, Atype);
    RECORD_ARG(10, lda);
    RECORD_ARG(11, B);
    RECORD_ARG(12, Btype);
    RECORD_ARG(13, ldb);
    RECORD_ARG(14, beta);
    RECORD_ARG(15, C);
    RECORD_ARG(16, Ctype);
    RECORD_ARG(17, ldc);
    LOGE(LOG_DEBUG, "cublasSgemmEx");
    GSCHED_RETAIN;
    *result = cublasSgemmEx(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) transa,
                    (cublasOperation_t) transb,
                    m, n, k, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), (cudaDataType_t)Atype, lda,
                    resource_mg_get(&rm_memory, (void*)B), (cudaDataType_t)Btype, ldb, &beta,
                    resource_mg_get(&rm_memory, (void*)C), (cudaDataType_t)Ctype, ldc
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassetstream_1_svc(ptr handle, ptr streamId, int *result, struct svc_req* rqstp)
{
#ifdef POS_ENABLE

    *result = pos_cuda_ws->pos_process( 
        /* api_id */ rpc_cublasSetStream, 
        /* uuid */ 0, 
        /* param_desps */ {
            { .value = &handle, .size = sizeof(ptr) },
            { .value = &streamId, .size = sizeof(ptr) }
        }
    );

#else // POS_ENABLE

    RECORD_API(rpc_cublassetstream_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(streamId);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    int proc = 3008;
    cpu_time_start(vanillas, proc);
    *result = cublasSetStream(
        resource_mg_get(&rm_cublas, (void*)handle),
        resource_mg_get(&rm_streams, (void*)streamId));
    cpu_time_end(vanillas, proc);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);

#endif // POS_ENABLE

    return 1;
}

bool_t rpc_cublassetworkspace_1_svc(ptr handle, ptr workspace, size_t workspaceSizeInBytes, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cublassetworkspace_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(workspace);
    RECORD_NARG(workspaceSizeInBytes);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
#if CUBLAS_VERSION >= 11000
    *result = cublasSetWorkspace(
        resource_mg_get(&rm_cublas, (void*)handle),
        resource_mg_get(&rm_memory, (void*)workspace),
        workspaceSizeInBytes);
#else
    LOGE(LOG_ERROR, "cublassetworkspace not supported in this version");
    *result = -1;
#endif
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cublassetmathmode_1_svc(ptr handle, int mode, int *result, struct svc_req *rqstp)
{
#ifdef POS_ENABLE

    *result = pos_cuda_ws->pos_process( 
        /* api_id */ rpc_cublasSetMathMode, 
        /* uuid */ 0, 
        /* param_desps */ {
            { .value = &handle, .size = sizeof(ptr) },
            { .value = &mode, .size = sizeof(int) },
        }
    );

#else // POS_ENABLE

    RECORD_API(rpc_cublassetmathmode_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(mode);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    int proc = 3010;
    cpu_time_start(vanillas, proc);
    *result = cublasSetMathMode(
        resource_mg_get(&rm_cublas, (void*)handle),
        (cublasMath_t)mode);
    cpu_time_end(vanillas, proc);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);

#endif // POS_ENABLE

    return 1;
}

bool_t rpc_cublassgemmstridedbatched_1_svc(ptr handle, int transa, int transb, int m, int n, int k, float alpha,
            ptr A, int lda, dint sA,
            ptr B, int ldb, dint sB,
            float beta,
            ptr C, int ldc, dint sC,
            int batchCount,
            int *result, struct svc_req *rqstp)
{
#ifdef POS_ENABLE

    long long int strideA = ((long long int)sA.i1 << 32) | sA.i2,
        strideB = ((long long int)sB.i1 << 32) | sB.i2,
        strideC = ((long long int)sC.i1 << 32) | sC.i2;

    *result = pos_cuda_ws->pos_process( 
        /* api_id */ rpc_cublasSgemmStridedBatched, 
        /* uuid */ 0, 
        /* param_desps */ {
            { .value = &handle, .size = sizeof(ptr) },
            { .value = &transa, .size = sizeof(cublasOperation_t) },
            { .value = &transb, .size = sizeof(cublasOperation_t) },
            { .value = &m, .size = sizeof(int) },
            { .value = &n, .size = sizeof(int) },
            { .value = &k, .size = sizeof(int) },
            { .value = &alpha, .size = sizeof(float) },
            { .value = &A, .size = sizeof(ptr) },
            { .value = &lda, .size = sizeof(int) },
            { .value = &strideA, .size = sizeof(long long int) },
            { .value = &B, .size = sizeof(ptr) },
            { .value = &ldb, .size = sizeof(int) },
            { .value = &strideB, .size = sizeof(long long int) },
            { .value = &beta, .size = sizeof(float) },
            { .value = &C, .size = sizeof(ptr) },
            { .value = &ldc, .size = sizeof(int) },
            { .value = &strideC, .size = sizeof(long long int) },
            { .value = &batchCount, .size = sizeof(int) },
        }
    );

#else // POS_ENABLE

    RECORD_API(rpc_cublassgemmstridedbatched_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, transa);
    RECORD_ARG(3, transb);
    RECORD_ARG(4, m);
    RECORD_ARG(5, n);
    RECORD_ARG(6, k);
    RECORD_ARG(7, alpha);
    RECORD_ARG(8, A);
    RECORD_ARG(9, lda);
    RECORD_ARG(10, sA);
    RECORD_ARG(11, B);
    RECORD_ARG(12, ldb);
    RECORD_ARG(13, sB);
    RECORD_ARG(14, beta);
    RECORD_ARG(15, C);
    RECORD_ARG(16, ldc);
    RECORD_ARG(17, sC);
    RECORD_ARG(18, batchCount);
    LOGE(LOG_DEBUG, "%s", __func__);
    /*
        strideA is split into sA.i1 and sA.i2
    */
    long long int strideA = ((long long int)sA.i1 << 32) | sA.i2,
        strideB = ((long long int)sB.i1 << 32) | sB.i2,
        strideC = ((long long int)sC.i1 << 32) | sC.i2;
    GSCHED_RETAIN;
    int proc = 3011;
    cpu_time_start(vanillas, proc);
    *result = cublasSgemmStridedBatched(resource_mg_get(&rm_cublas, (void*)handle),
                    (cublasOperation_t) transa,
                    (cublasOperation_t) transb,
                    m, n, k, &alpha,
                    resource_mg_get(&rm_memory, (void*)A), lda, strideA,
                    resource_mg_get(&rm_memory, (void*)B), ldb, strideB, &beta,
                    resource_mg_get(&rm_memory, (void*)C), ldc, strideC,
                    batchCount
    );
    cpu_time_end(vanillas, proc);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);

#endif // POS_ENABLE

    return 1;
}
