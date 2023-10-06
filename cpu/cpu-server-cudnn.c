
#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <stdbool.h>

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "resource-mg.h"
#include "gsched.h"

#define WITH_RECORDER
#include "api-recorder.h"

#include "cpu-server-cudnn.h"
#include "cpu-measurement.h"

extern measurement_info vanillas[6000];

int server_cudnn_init(int bypass)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cudnn, bypass);
    ret &= resource_mg_init(&rm_cudnn_tensors, bypass);
    ret &= resource_mg_init(&rm_cudnn_filters, bypass);
    ret &= resource_mg_init(&rm_cudnn_poolings, bypass);
    ret &= resource_mg_init(&rm_cudnn_activations, bypass);
    ret &= resource_mg_init(&rm_cudnn_lrns, bypass);
    ret &= resource_mg_init(&rm_cudnn_convs, bypass);
    ret &= resource_mg_init(&rm_cudnn_backendds, bypass);
    return ret;
}

int server_cudnn_deinit(void)
{
    resource_mg_free(&rm_cudnn);
    resource_mg_free(&rm_cudnn_tensors);
    resource_mg_free(&rm_cudnn_filters);
    resource_mg_free(&rm_cudnn_poolings);
    resource_mg_free(&rm_cudnn_activations);
    resource_mg_free(&rm_cudnn_lrns);
    resource_mg_free(&rm_cudnn_convs);
    resource_mg_free(&rm_cudnn_backendds);
    return 0;

}

bool_t rpc_cudnngetversion_1_svc(size_t *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnGetVersion();
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetmaxdeviceversion_1_svc(size_t *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    // *result = cudnnGetMaxDeviceVersion();
    *result = 0;
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetcudartversion_1_svc(size_t *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnGetCudartVersion();
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngeterrorstring_1_svc(int status, char **result, struct svc_req *rqstp)
{
    const char* str;
    *result = malloc(128);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    str = cudnnGetErrorString((cudnnStatus_t)status);
    strncpy(*result, str, 128);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnqueryruntimeerror_1_svc(ptr handle, int mode, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    cudnnRuntimeTag_t *tag;

    GSCHED_RETAIN;
    result->err = cudnnQueryRuntimeError(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnStatus_t*)&result->int_result_u.data, (cudnnErrQueryMode_t)mode, tag);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetproperty_1_svc(int type, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetProperty((libraryPropertyType)type, &result->int_result_u.data); 
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnncreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreate((cudnnHandle_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnndestroy_1_svc(ptr handle, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(handle);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroy(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsetstream_1_svc(ptr handle, ptr streamId, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetstream_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(streamId);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetStream(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudaStream_t)resource_mg_get(&rm_streams, (void*)streamId));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetstream_1_svc(ptr handle, ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetStream(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudaStream_t*)&result->ptr_result_u.ptr);

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnncreatetensordescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    int proc = 5010;
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    time_start(vanillas, proc);
    result->err = cudnnCreateTensorDescriptor((cudnnTensorDescriptor_t*)&result->ptr_result_u.ptr);
    time_end(vanillas, proc);
    if (resource_mg_create(&rm_cudnn_tensors, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsettensor4ddescriptor_1_svc(ptr tensorDesc, int format, int dataType, int n, int c, int h, int w, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensor4ddescriptor_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(format);
    RECORD_NARG(dataType);
    RECORD_NARG(n);
    RECORD_NARG(c);
    RECORD_NARG(h);
    RECORD_NARG(w);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetTensor4dDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnTensorFormat_t)format,
        (cudnnDataType_t)dataType,
        n, c, h, w);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsettensor4ddescriptorex_1_svc(ptr tensorDesc, int dataType, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensor4ddescriptorex_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(n);
    RECORD_NARG(c);
    RECORD_NARG(h);
    RECORD_NARG(w);
    RECORD_NARG(nStride);
    RECORD_NARG(cStride);
    RECORD_NARG(hStride);
    RECORD_NARG(wStride);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetTensor4dDescriptorEx(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnDataType_t)dataType,
        n, c, h, w, nStride, cStride, hStride, wStride);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngettensor4ddescriptor_1_svc(ptr tensorDesc, int9_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetTensor4dDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnDataType_t*)&result->int9_result_u.data[0],
        &result->int9_result_u.data[1],
        &result->int9_result_u.data[2],
        &result->int9_result_u.data[3],
        &result->int9_result_u.data[4],
        &result->int9_result_u.data[5],
        &result->int9_result_u.data[6],
        &result->int9_result_u.data[7],
        &result->int9_result_u.data[8]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsettensornddescriptor_1_svc(ptr tensorDesc, int dataType, int nbDims, mem_data dimA, mem_data strideA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensornddescriptor_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(nbDims);
    RECORD_NARG(dimA);
    RECORD_NARG(strideA);
    
    //TODO: Recording dimA and strideA is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (dimA.mem_data_len != nbDims * sizeof(int) || strideA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetTensorNdDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnDataType_t)dataType,
        nbDims,
        (const int*)dimA.mem_data_val,
        (const int*)strideA.mem_data_val);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsettensornddescriptorex_1_svc(ptr tensorDesc, int format, int dataType, int nbDims, mem_data dimA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensornddescriptorex_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(format);
    RECORD_NARG(dataType);
    RECORD_NARG(nbDims);
    RECORD_NARG(dimA);
    
    //TODO: Recording dimA and strideA is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (dimA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetTensorNdDescriptorEx(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        (cudnnTensorFormat_t)format,   
        (cudnnDataType_t)dataType,
        nbDims,
        (const int*)dimA.mem_data_val);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngettensornddescriptor_1_svc(ptr tensorDesc, int nbDimsRequested, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    result->mem_result_u.data.mem_data_len = sizeof(cudnnDataType_t) + sizeof(int) + nbDimsRequested*sizeof(int)*2;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    
    GSCHED_RETAIN;
    result->err = cudnnGetTensorNdDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        nbDimsRequested,
        (cudnnDataType_t*)result->mem_result_u.data.mem_data_val,
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(int)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(int)+nbDimsRequested*sizeof(int)]);

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngettensorsizeinbytes_1_svc(ptr tensorDesc, sz_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cudnnGetTensorSizeInBytes(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc),
        &result->sz_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroytensordescriptor_1_svc(ptr tensorDesc, int *result, struct svc_req *rqstp)
{
    int proc = 5018;
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(tensorDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    time_start(vanillas, proc);
    *result = cudnnDestroyTensorDescriptor(
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)tensorDesc));
    time_end(vanillas, proc);
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}


bool_t rpc_cudnncreatefilterdescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreateFilterDescriptor((cudnnFilterDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_filters, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetfilter4ddescriptor_1_svc(ptr filterDesc, int dataType, int format, int k, int c, int h, int w, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetfilter4ddescriptor_1_argument);
    RECORD_NARG(filterDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(format);
    RECORD_NARG(k);
    RECORD_NARG(c);
    RECORD_NARG(h);
    RECORD_NARG(w);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetFilter4dDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        (cudnnDataType_t)dataType,
        (cudnnTensorFormat_t)format,
        k, c, h, w);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetfilter4ddescriptor_1_svc(ptr filterDesc, int6_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetFilter4dDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        (cudnnDataType_t*)&result->int6_result_u.data[0],
        (cudnnTensorFormat_t*)&result->int6_result_u.data[1],
        &result->int6_result_u.data[2],
        &result->int6_result_u.data[3],
        &result->int6_result_u.data[4],
        &result->int6_result_u.data[5]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsetfilternddescriptor_1_svc(ptr filterDesc, int dataType, int format, int nbDims, mem_data filterDimA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetfilternddescriptor_1_argument);
    RECORD_NARG(filterDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(format);
    RECORD_NARG(nbDims);
    RECORD_NARG(filterDimA);
    
    //TODO: Recording filterDimA is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (filterDimA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimension not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetFilterNdDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        (cudnnDataType_t)dataType,
        (cudnnTensorFormat_t)format,
        nbDims,
        (const int*)filterDimA.mem_data_val);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetfilternddescriptor_1_svc(ptr filterDesc, int nbDimsRequested, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    result->mem_result_u.data.mem_data_len = sizeof(cudnnDataType_t) + sizeof(cudnnTensorFormat_t) + sizeof(int) + nbDimsRequested*sizeof(int);
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    
    GSCHED_RETAIN;
    result->err = cudnnGetFilterNdDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        nbDimsRequested,
        (cudnnDataType_t*)result->mem_result_u.data.mem_data_val,
        (cudnnTensorFormat_t*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(cudnnTensorDescriptor_t)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(cudnnTensorDescriptor_t)+sizeof(int)]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetfiltersizeinbytes_1_svc(ptr filterDesc, sz_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cudnnGetFilterSizeInBytes(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        &result->sz_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnntransformfilter_1_svc(ptr handle, ptr transDesc, cudnn_scaling_t alpha, ptr srcDesc, ptr srcData, cudnn_scaling_t beta, ptr destDesc, ptr destData, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnntransformfilter_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(transDesc);
    RECORD_NARG(alpha);
    RECORD_NARG(srcDesc);
    RECORD_NARG(srcData);
    RECORD_NARG(beta);
    RECORD_NARG(destDesc);
    RECORD_NARG(destData);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnTransformFilter(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (const cudnnTensorTransformDescriptor_t)resource_mg_get(&rm_cudnn_tensortransform, (void*)transDesc),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (const cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)srcDesc),
        (const void*)srcData,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (const cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)destDesc),
        (void*)destData);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnndestroyfilterdescriptor_1_svc(ptr filterDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(filterDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyFilterDescriptor(
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnncreatepoolingdescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreatePoolingDescriptor((cudnnPoolingDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_poolings, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetpooling2ddescriptor_1_svc(ptr poolingDesc, int mode, int maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetpooling2ddescriptor_1_argument);
    RECORD_NARG(poolingDesc);
    RECORD_NARG(mode);
    RECORD_NARG(maxpoolingNanOpt);
    RECORD_NARG(windowHeight);
    RECORD_NARG(windowWidth);
    RECORD_NARG(verticalPadding);
    RECORD_NARG(horizontalPadding);
    RECORD_NARG(verticalStride);
    RECORD_NARG(horizontalStride);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetPooling2dDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnPoolingMode_t)mode,
        (cudnnNanPropagation_t)maxpoolingNanOpt,
        windowHeight, windowWidth,
        verticalPadding, horizontalPadding,
        verticalStride, horizontalStride);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetpooling2ddescriptor_1_svc(ptr poolingDesc, int8_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetPooling2dDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnPoolingMode_t*)&result->int8_result_u.data[0],
        (cudnnNanPropagation_t*)&result->int8_result_u.data[1],
        &result->int8_result_u.data[2],
        &result->int8_result_u.data[3],
        &result->int8_result_u.data[4],
        &result->int8_result_u.data[5],
        &result->int8_result_u.data[6],
        &result->int8_result_u.data[7]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsetpoolingnddescriptor_1_svc(ptr poolingDesc, int mode, int maxpoolingNanOpt, int nbDims, mem_data windowDimA, mem_data paddingA, mem_data strideA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetpoolingnddescriptor_1_argument);
    RECORD_NARG(poolingDesc);
    RECORD_NARG(mode);
    RECORD_NARG(maxpoolingNanOpt);
    RECORD_NARG(nbDims);
    RECORD_NARG(windowDimA);
    RECORD_NARG(paddingA);
    RECORD_NARG(strideA);
    //TODO: Recording windowDimA, paddingA and strideA are not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (windowDimA.mem_data_len != nbDims * sizeof(int) ||
        paddingA.mem_data_len != nbDims * sizeof(int) ||
        strideA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetPoolingNdDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnPoolingMode_t)mode,
        (cudnnNanPropagation_t)maxpoolingNanOpt,
        nbDims,
        (const int*)windowDimA.mem_data_val,
        (const int*)paddingA.mem_data_val,
        (const int*)strideA.mem_data_val);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetpoolingnddescriptor_1_svc(ptr poolingDesc, int nbDimsRequested, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    result->mem_result_u.data.mem_data_len = sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + nbDimsRequested * sizeof(int) * 3;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    
    size_t offsets[] = {
        0,
        sizeof(cudnnPoolingMode_t),
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t),
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + sizeof(int),
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + sizeof(int) + sizeof(int) * nbDimsRequested,
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + sizeof(int) + sizeof(int) * nbDimsRequested * 2,
    };
    
    GSCHED_RETAIN;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
    result->err = cudnnGetPoolingNdDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        nbDimsRequested,
        (cudnnPoolingMode_t*)result->mem_result_u.data.mem_data_val[offsets[0]],
        (cudnnNanPropagation_t*)result->mem_result_u.data.mem_data_val[offsets[1]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[2]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[3]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[4]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[5]]);
#pragma GCC diagnostic pop

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetpoolingndforwardoutputdim_1_svc(ptr poolingDesc, ptr inputTensorDesc, int nbDims, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->mem_result_u.data.mem_data_len = sizeof(int) * nbDims;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    result->err = cudnnGetPoolingNdForwardOutputDim(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)inputTensorDesc),
        nbDims,
        (int*)result->mem_result_u.data.mem_data_val);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetpooling2dforwardoutputdim_1_svc(ptr poolingDesc, ptr inputTensorDesc, int4_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cudnnGetPooling2dForwardOutputDim(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)inputTensorDesc),
        (int*)&result->int4_result_u.data[0],
        (int*)&result->int4_result_u.data[1],
        (int*)&result->int4_result_u.data[2],
        (int*)&result->int4_result_u.data[3]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroypoolingdescriptor_1_svc(ptr poolingDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(poolingDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyPoolingDescriptor(
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnncreateactivationdescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreateActivationDescriptor((cudnnActivationDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_activations, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetactivationdescriptor_1_svc(ptr activationDesc, int mode, int reluNanOpt, double coef, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetactivationdescriptor_1_argument);
    RECORD_NARG(activationDesc);
    RECORD_NARG(mode);
    RECORD_NARG(reluNanOpt);
    RECORD_NARG(coef);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetActivationDescriptor(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        (cudnnActivationMode_t)mode,
        (cudnnNanPropagation_t)reluNanOpt,
        coef);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetactivationdescriptor_1_svc(ptr activationDesc, int2d1_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetActivationDescriptor(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        (cudnnActivationMode_t*)&result->int2d1_result_u.data.i[0],
        (cudnnNanPropagation_t*)&result->int2d1_result_u.data.i[1],
        &result->int2d1_result_u.data.d);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsetactivationdescriptorswishbeta_1_svc(ptr activationDesc, double swish_beta, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetactivationdescriptorswishbeta_1_argument);
    RECORD_NARG(activationDesc);
    RECORD_NARG(swish_beta);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetActivationDescriptorSwishBeta(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        swish_beta);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetactivationdescriptorswishbeta_1_svc(ptr activationDesc, d_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetActivationDescriptorSwishBeta(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        &result->d_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroyactivationdescriptor_1_svc(ptr activationDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(activationDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyActivationDescriptor(
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnncreatelrndescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreateLRNDescriptor((cudnnLRNDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_lrns, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetlrndescriptor_1_svc(ptr normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetlrndescriptor_1_argument);
    RECORD_NARG(normDesc);
    RECORD_NARG(lrnN);
    RECORD_NARG(lrnAlpha);
    RECORD_NARG(lrnBeta);
    RECORD_NARG(lrnK);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnSetLRNDescriptor(
        (cudnnLRNDescriptor_t)resource_mg_get(&rm_cudnn_lrns, (void*)normDesc),
        lrnN,
        lrnAlpha,
        lrnBeta,
        lrnK);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetlrndescriptor_1_svc(ptr normDesc, int1d3_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetLRNDescriptor(
        (cudnnLRNDescriptor_t)resource_mg_get(&rm_cudnn_lrns, (void*)normDesc),
        (unsigned int*)&result->int1d3_result_u.data.i,
        &result->int1d3_result_u.data.d[0],
        &result->int1d3_result_u.data.d[1],
        &result->int1d3_result_u.data.d[2]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroylrndescriptor_1_svc(ptr lrnDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(lrnDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyLRNDescriptor(
        (cudnnLRNDescriptor_t)resource_mg_get(&rm_cudnn_lrns, (void*)lrnDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnpoolingforward_1_svc(ptr handle, ptr poolingDesc,           cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnpoolingforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(poolingDesc);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnPoolingForward(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnPoolingDescriptor_t)resource_mg_get(&rm_cudnn_poolings, (void*)poolingDesc),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)x),
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (void*)resource_mg_get(&rm_memory, (void*)y));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnactivationforward_1_svc(ptr handle, ptr activationDesc, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnactivationforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(activationDesc);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnActivationForward(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)x),
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (void*)resource_mg_get(&rm_memory, (void*)y));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnlrncrosschannelforward_1_svc(ptr handle, ptr normDesc, int lrnMode, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnlrncrosschannelforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(normDesc);
    RECORD_NARG(lrnMode);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnLRNCrossChannelForward(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnLRNDescriptor_t)resource_mg_get(&rm_cudnn_lrns, (void*)normDesc),
        (cudnnLRNMode_t)lrnMode,
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)x),
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (void*)resource_mg_get(&rm_memory, (void*)y));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsoftmaxforward_1_svc(ptr handle, int algo, int mode, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsoftmaxforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(algo);
    RECORD_NARG(mode);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnSoftmaxForward(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnSoftmaxAlgorithm_t)algo,
        (cudnnSoftmaxMode_t)mode,
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)x),
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (void*)resource_mg_get(&rm_memory, (void*)y));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

/* cudnn cnn inference */
bool_t rpc_cudnngetconvolutionndforwardoutputdim_1_svc(ptr convDesc, ptr inputTensorDesc, ptr filterDesc, int nbDims, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->mem_result_u.data.mem_data_len = sizeof(int) * nbDims;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    result->err = cudnnGetConvolutionNdForwardOutputDim(
        (cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)inputTensorDesc),
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        nbDims,
        (int*)result->mem_result_u.data.mem_data_val);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnncreateconvolutiondescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnCreateConvolutionDescriptor((cudnnConvolutionDescriptor_t*)&result->ptr_result_u.ptr);
    if (resource_mg_create(&rm_cudnn_convs, (void*)result->ptr_result_u.ptr) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnndestroyconvolutiondescriptor_1_svc(ptr convDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(convDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnDestroyConvolutionDescriptor(
        (cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsetconvolutionnddescriptor_1_svc(ptr convDesc, int arrayLength, mem_data padA, mem_data filterStrideA, mem_data dilationA, int mode, int computeType, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetconvolutionnddescriptor_1_argument);
    RECORD_NARG(convDesc);
    RECORD_NARG(arrayLength);
    RECORD_NARG(padA);
    RECORD_NARG(filterStrideA);
    RECORD_NARG(dilationA);
    RECORD_NARG(mode);
    RECORD_NARG(computeType);
    //TODO: Recording mem_data is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (padA.mem_data_len != arrayLength * sizeof(int) ||
        filterStrideA.mem_data_len != arrayLength * sizeof(int) ||
        dilationA.mem_data_len != arrayLength * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetConvolutionNdDescriptor(
        (cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        arrayLength,
        (const int*)padA.mem_data_val,
        (const int*)filterStrideA.mem_data_val,
        (const int*)dilationA.mem_data_val,
        (cudnnConvolutionMode_t)mode,
        (cudnnDataType_t)computeType);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetconvolutionforwardalgorithm_v7_1_svc(ptr handle, ptr srcDesc, ptr filterDesc, ptr convDesc, ptr destDesc, int requestedAlgoCount, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->mem_result_u.data.mem_data_len = sizeof(int) + sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    result->err = cudnnGetConvolutionForwardAlgorithm_v7(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)srcDesc),
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)filterDesc),
        (cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)destDesc),
        requestedAlgoCount,
        (int*)result->mem_result_u.data.mem_data_val,
        (cudnnConvolutionFwdAlgoPerf_t*)(result->mem_result_u.data.mem_data_val + sizeof(int)));
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnfindconvolutionforwardalgorithm_1_svc(ptr handle, ptr xDesc, ptr wDesc, ptr convDesc, ptr yDesc, int requestedAlgoCount, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->mem_result_u.data.mem_data_len = sizeof(int) + sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    result->err = cudnnFindConvolutionForwardAlgorithm(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)wDesc),
        (cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        requestedAlgoCount,
        (int*)result->mem_result_u.data.mem_data_val,
        (cudnnConvolutionFwdAlgoPerf_t*)(result->mem_result_u.data.mem_data_val + sizeof(int)));
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetconvolutionforwardworkspacesize_1_svc(ptr handle, ptr xDesc, ptr wDesc, ptr convDesc, ptr yDesc, int algo, sz_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cudnnGetConvolutionForwardWorkspaceSize(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)wDesc),
        (cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (cudnnConvolutionFwdAlgo_t)algo,
        (size_t*)&result->sz_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnconvolutionforward_1_svc(ptr handle, cudnn_scaling_t alpha, ptr xDesc, ptr x, ptr wDesc, ptr w, ptr convDesc, int algo, ptr workSpace, size_t workSpaceSizeInBytes, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnconvolutionforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(wDesc);
    RECORD_NARG(w);
    RECORD_NARG(convDesc);
    RECORD_NARG(algo);
    RECORD_NARG(workSpace);
    RECORD_NARG(workSpaceSizeInBytes);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnConvolutionForward(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)x),
        (cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)wDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)w),
        (cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        algo,
        (void*)resource_mg_get(&rm_memory, (void*)workSpace),
        workSpaceSizeInBytes,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (void*)resource_mg_get(&rm_memory, (void*)y));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsetconvolutiongroupcount_1_svc(ptr convDesc, int groupCnt, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetconvolutiongroupcount_1_argument);
    RECORD_NARG(convDesc);
    RECORD_NARG(groupCnt);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnSetConvolutionGroupCount(
        (cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        groupCnt
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsetconvolutionmathtype_1_svc(ptr convDesc, int mathType, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetconvolutionmathtype_1_argument);
    RECORD_NARG(convDesc);
    RECORD_NARG(mathType);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnSetConvolutionMathType(
        (cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        (cudnnMathType_t)mathType
    );
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnaddtensor_1_svc(ptr handle, cudnn_scaling_t alpha, ptr aDesc, ptr A, cudnn_scaling_t beta, ptr cDesc, ptr C, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnaddtensor_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(alpha);
    RECORD_NARG(aDesc);
    RECORD_NARG(A);
    RECORD_NARG(beta);
    RECORD_NARG(cDesc);
    RECORD_NARG(C);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnAddTensor(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)aDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)A),
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)cDesc),
        (void*)resource_mg_get(&rm_memory, (void*)C));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnntransformtensor_1_svc(ptr handle, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnntransformtensor_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnTransformTensor(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)x),
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (void*)resource_mg_get(&rm_memory, (void*)y));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1_svc(ptr handle, int mode, int bnOps, ptr xDesc, ptr zDesc, ptr yDesc, ptr bnScaleBiasMeanVarDesc, ptr activationDesc, sz_result* result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnngetbatchnormalizationforwardtrainingexworkspacesize_1_argument);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnBatchNormMode_t)mode,
        (cudnnBatchNormOps_t)bnOps,
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)zDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)bnScaleBiasMeanVarDesc),
        (const cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        (size_t*)&result->sz_result_u.data
    );
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnbatchnormalizationforwardtrainingex_1_svc(ptr handle, int mode, int bnOps, cudnn_scaling_t alpha, cudnn_scaling_t beta, ptr xDesc, ptr xData, ptr zDesc, ptr zData, ptr yDesc, ptr yData, ptr bnScaleBiasMeanVarDesc, ptr bnScaleData, ptr bnBiasData, double exponentialAverageFactor, ptr resultRunningMeanData, ptr resultRunningVarianceData, double epsilon, ptr saveMean, ptr saveInvVariance, ptr activationDesc, ptr workspace, size_t workSpaceSizeInBytes, ptr reserveSpace, size_t reserveSpaceSizeInBytes, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnbatchnormalizationforwardtrainingex_1_argument);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnBatchNormalizationForwardTrainingEx(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnBatchNormMode_t)mode,
        (cudnnBatchNormOps_t)bnOps,
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)xData),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)zDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)zData),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (void*)resource_mg_get(&rm_memory, (void*)yData),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)bnScaleBiasMeanVarDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)bnScaleData),
        (const void*)resource_mg_get(&rm_memory, (void*)bnBiasData),
        exponentialAverageFactor,
        (void*)resource_mg_get(&rm_memory, (void*)resultRunningMeanData),
        (void*)resource_mg_get(&rm_memory, (void*)resultRunningVarianceData),
        epsilon,
        (void*)resource_mg_get(&rm_memory, (void*)saveMean),
        (void*)resource_mg_get(&rm_memory, (void*)saveInvVariance),
        (const cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        (void*)resource_mg_get(&rm_memory, (void*)workspace),
        workSpaceSizeInBytes,
        (void*)resource_mg_get(&rm_memory, (void*)reserveSpace),
        reserveSpaceSizeInBytes
    );
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1_svc(ptr handle, int mode, int bnOps, ptr xDesc, ptr yDesc, ptr dyDesc, ptr dzDesc, ptr dxDesc, ptr dBnScaleBiasDesc, ptr activationDesc, sz_result *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnngetbatchnormalizationbackwardexworkspacesize_1_argument);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnBatchNormMode_t)mode,
        (cudnnBatchNormOps_t)bnOps,
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dyDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dzDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dxDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dBnScaleBiasDesc),
        (const cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        (size_t*)&result->sz_result_u.data
    );
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnbatchnormalizationbackwardex_1_svc(ptr handle, int mode, int bnOps, cudnn_scaling_t alphaDataDiff, cudnn_scaling_t betaDataDiff, cudnn_scaling_t alphaParamDiff, cudnn_scaling_t betaParamDiff, ptr xDesc, ptr xData, ptr yDesc, ptr yData, ptr dyDesc, ptr dyData, ptr dzDesc, ptr dzData, ptr dxDesc, ptr dxData, ptr dBnScaleBiasDesc, ptr bnScaleData, ptr bnBiasData, ptr dBnScaleData, ptr dBnBiasData, double epsilon, ptr saveMean, ptr saveInvVariance, ptr activationDesc, ptr workspace, size_t workSpaceSizeInBytes, ptr reserveSpace, size_t reserveSpaceSizeInBytes, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnbatchnormalizationbackwardex_1_argument);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnBatchNormalizationBackwardEx(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnBatchNormMode_t)mode,
        (cudnnBatchNormOps_t)bnOps,
        (alphaDataDiff.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alphaDataDiff.cudnn_scaling_t_u.d : (const void*)&alphaDataDiff.cudnn_scaling_t_u.f),
        (betaDataDiff.dataType == CUDNN_DATA_DOUBLE ? (const void*)&betaDataDiff.cudnn_scaling_t_u.d : (const void*)&betaDataDiff.cudnn_scaling_t_u.f),
        (alphaParamDiff.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alphaParamDiff.cudnn_scaling_t_u.d : (const void*)&alphaParamDiff.cudnn_scaling_t_u.f),
        (betaParamDiff.dataType == CUDNN_DATA_DOUBLE ? (const void*)&betaParamDiff.cudnn_scaling_t_u.d : (const void*)&betaParamDiff.cudnn_scaling_t_u.f), 
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)xData),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)yData),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dyDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)dyData),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dzDesc),
        (void*)resource_mg_get(&rm_memory, (void*)dzData),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dxDesc),
        (void*)resource_mg_get(&rm_memory, (void*)dxData),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dBnScaleBiasDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)bnScaleData),
        (const void*)resource_mg_get(&rm_memory, (void*)bnBiasData),
        (void*)resource_mg_get(&rm_memory, (void*)dBnScaleData),
        (void*)resource_mg_get(&rm_memory, (void*)dBnBiasData),
        epsilon,
        (const void*)resource_mg_get(&rm_memory, (void*)saveMean),
        (const void*)resource_mg_get(&rm_memory, (void*)saveInvVariance),
        (const cudnnActivationDescriptor_t)resource_mg_get(&rm_cudnn_activations, (void*)activationDesc),
        (void*)resource_mg_get(&rm_memory, (void*)workspace),
        (size_t)workSpaceSizeInBytes,
        (void*)resource_mg_get(&rm_memory, (void*)reserveSpace),
        (size_t)reserveSpaceSizeInBytes
    );
    GSCHED_RELEASE;
    return 1;
}

// 5315
bool_t rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1_svc(ptr handle, ptr wDesc, ptr dyDesc, ptr convDesc, ptr dxDesc, int requestedAlgoCount, mem_result *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnngetconvolutionbackwarddataalgorithm_v7_1_argument);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->mem_result_u.data.mem_data_len = sizeof(int) + sizeof(cudnnConvolutionBwdDataAlgoPerf_t) * requestedAlgoCount;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    result->err = cudnnGetConvolutionBackwardDataAlgorithm_v7(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn,(void*)handle),
        (const cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)wDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dyDesc),
        (const cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dxDesc),
        requestedAlgoCount,
        (int*)result->mem_result_u.data.mem_data_val,
        (cudnnConvolutionBwdDataAlgoPerf_t*)(result->mem_result_u.data.mem_data_val + sizeof(int))
    );
    GSCHED_RELEASE;
    return 1;
}

// 5316
bool_t rpc_cudnnconvolutionbackwarddata_1_svc(ptr handle, cudnn_scaling_t alpha, ptr wDesc, ptr w, ptr dyDesc, ptr dy, ptr convDesc, int algo, ptr workSpace, size_t workSpaceSizeInBytes, cudnn_scaling_t beta, ptr dxDesc, ptr dx, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnconvolutionbackwarddata_1_argument);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnConvolutionBackwardData(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (const cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)wDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)w),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dyDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)dy),
        (const cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        (cudnnConvolutionBwdDataAlgo_t) algo,
        (void*)resource_mg_get(&rm_memory, (void*)workSpace),
        workSpaceSizeInBytes,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (const cudnnTensorDescriptor_t) resource_mg_get(&rm_cudnn_tensors, (void*)dxDesc),
        (void*)resource_mg_get(&rm_memory, (void*)dx)
    );
    GSCHED_RELEASE;
    return 1;
}


// 5317
bool_t rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1_svc(ptr handle, ptr xDesc, ptr dyDesc, ptr convDesc, ptr dwDesc, int requestedAlgoCount, mem_result *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnngetconvolutionbackwardfilteralgorithm_v7_1_argument);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->mem_result_u.data.mem_data_len = sizeof(int) + sizeof(cudnnConvolutionBwdFilterAlgoPerf_t) * requestedAlgoCount;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    result->err = cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn,(void*)handle),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dyDesc),
        (const cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        (const cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)dwDesc),
        (const int)requestedAlgoCount,
        (int*)result->mem_result_u.data.mem_data_val,
        (cudnnConvolutionBwdFilterAlgoPerf_t*)(result->mem_result_u.data.mem_data_val + sizeof(int))
    );
    GSCHED_RELEASE;
    return 1;
}

// 5318

bool_t rpc_cudnnconvolutionbackwardfilter_1_svc(ptr handle, cudnn_scaling_t alpha, ptr xDesc, ptr x, ptr dyDesc, ptr dy, ptr convDesc, int algo, ptr workSpace, size_t workSpaceSizeInBytes, cudnn_scaling_t beta, ptr dwDesc, ptr dw, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnconvolutionbackwardfilter_1_argument);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnConvolutionBackwardFilter(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)x),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)dyDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)dy),
        (const cudnnConvolutionDescriptor_t)resource_mg_get(&rm_cudnn_convs, (void*)convDesc),
        (cudnnConvolutionBwdFilterAlgo_t)algo,
        (void*)resource_mg_get(&rm_memory, (void*)workSpace),
        workSpaceSizeInBytes,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),  
        (const cudnnFilterDescriptor_t)resource_mg_get(&rm_cudnn_filters, (void*)dwDesc),
        (void*)resource_mg_get(&rm_memory, (void*)dw)
    );
    GSCHED_RELEASE;
    return 1;
}

// 5319
bool_t rpc_cudnnbatchnormalizationforwardinference_1_svc(ptr handle, int mode, cudnn_scaling_t alpha, cudnn_scaling_t beta, ptr xDesc, ptr x, ptr yDesc, ptr y, ptr bnScaleBiasMeanVarDesc, ptr bnScale, ptr bnBias, ptr estimatedMean, ptr estimatedVariance, double epsilon, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnbatchnormalizationforwardinference_1_argument);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    *result = cudnnBatchNormalizationForwardInference(
        (cudnnHandle_t)resource_mg_get(&rm_cudnn, (void*)handle),
        (cudnnBatchNormMode_t)mode,
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)xDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)x),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)yDesc),
        (void*)resource_mg_get(&rm_memory, (void*)y),
        (const cudnnTensorDescriptor_t)resource_mg_get(&rm_cudnn_tensors, (void*)bnScaleBiasMeanVarDesc),
        (const void*)resource_mg_get(&rm_memory, (void*)bnScale),
        (const void*)resource_mg_get(&rm_memory, (void*)bnBias),
        (const void*)resource_mg_get(&rm_memory, (void*)estimatedMean),
        (const void*)resource_mg_get(&rm_memory, (void*)estimatedVariance),
        epsilon
    );
    GSCHED_RELEASE;
    return 1;
}