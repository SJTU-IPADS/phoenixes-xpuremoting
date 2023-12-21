#include "trace.h"

#include <iostream>
#include <ostream>

int on_trace = 0;
extern "C" void startTrace() {
    on_trace = 1;
}

std::map<std::string, int> *api_dict;
std::vector<APIRecord> *api_records;

static void add_api(int api_id, const std::string &api_name) {
    (*api_dict)[api_name] = api_id;
}

TraceProfile::TraceProfile(const std::string &name) {
    if (!on_trace) {
        return;
    }
    api_name = name;
    call_start = std::chrono::steady_clock::now();
}

TraceProfile::~TraceProfile() {
    if (!on_trace) {
        return;
    }
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - call_start);
    api_records->emplace_back(api_name, duration.count());
}

void __attribute__((constructor)) trace_init(void) {
    api_dict = new std::map<std::string, int>();
    int id = 0;
    add_api(++id, "__cudaPushCallConfiguration");
    add_api(++id, "__cudaPopCallConfiguration");
    add_api(++id, "__cudaRegisterFatBinary");
    add_api(++id, "__cudaRegisterFatBinaryEnd");
    add_api(++id, "__cudaRegisterFunction");
    add_api(++id, "__cudaRegisterVar");
    add_api(++id, "__cudaUnregisterFatBinary");
    add_api(++id, "cuDevicePrimaryCtxGetState");
    add_api(++id, "cublasCreate_v2");
    add_api(++id, "cublasSetMathMode");
    add_api(++id, "cublasSetStream_v2");
    add_api(++id, "cublasSgemmStridedBatched");
    add_api(++id, "cublasSgemm_v2");
    add_api(++id, "cudaDeviceGetAttribute");
    add_api(++id, "cudaFuncGetAttributes");
    add_api(++id, "cudaGetDevice");
    add_api(++id, "cudaGetDeviceCount");
    add_api(++id, "cudaGetDeviceProperties");
    add_api(++id, "cudaGetLastError");
    add_api(++id, "cudaLaunchKernel");
    add_api(++id, "cudaMalloc");
    add_api(++id, "cudaMemcpyAsync");
    add_api(++id, "cudaMemsetAsync");
    add_api(++id, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    add_api(++id, "cudaPeekAtLastError");
    add_api(++id, "cudaStreamIsCapturing");
    add_api(++id, "cudaStreamSynchronize");
    add_api(++id, "cudnnBatchNormalizationForwardInference");
    add_api(++id, "cudnnConvolutionForward");
    add_api(++id, "cudnnCreate");
    add_api(++id, "cudnnCreateConvolutionDescriptor");
    add_api(++id, "cudnnCreateFilterDescriptor");
    add_api(++id, "cudnnCreateTensorDescriptor");
    add_api(++id, "cudnnDestroyConvolutionDescriptor");
    add_api(++id, "cudnnDestroyFilterDescriptor");
    add_api(++id, "cudnnDestroyTensorDescriptor");
    add_api(++id, "cudnnGetConvolutionForwardAlgorithm_v7");
    add_api(++id, "cudnnSetConvolutionGroupCount");
    add_api(++id, "cudnnSetConvolutionMathType");
    add_api(++id, "cudnnSetConvolutionNdDescriptor");
    add_api(++id, "cudnnSetFilterNdDescriptor");
    add_api(++id, "cudnnSetStream");
    add_api(++id, "cudnnSetTensorNdDescriptor");
    add_api(++id, "nvmlDeviceGetCount_v2");
    add_api(++id, "nvmlInitWithFlags");
    add_api(++id, "nvmlInit_v2");

    api_records = new std::vector<APIRecord>();
}

static void print_api_records() {
    std::map<std::string, int> api_count;
    for (auto &api_record : *api_records) {
        if (api_dict->find(api_record.api_name) == api_dict->end()) {
            std::cout << "[ERROR] API " << api_record.api_name << " not set in dict!" << std::endl;
            exit(1);
        }
        api_count[api_record.api_name]++;
        if (api_record.api_name.find("__cuda") != std::string::npos ||
            api_record.api_name.find("rpc_") != std::string::npos) {
            api_record.api_name = "";
        }
    }

    std::cout << std::endl << std::endl << "API Lists:" << std::endl;
    for (auto &api : api_count) {
        std::cout << api.first << " " << api.second << std::endl;
    }

    std::cout << std::endl << std::endl << "API Traces: api, interval(ns)" << std::endl;
    for (auto &api_record : *api_records) {
        if (api_record.api_name == "") {
            continue;
        }
        std::cout << api_record.api_name << " " << api_record.interval << std::endl;
    }
}

void __attribute__((destructor)) trace_deinit(void) {
    print_api_records();
    delete api_dict;
    delete api_records;
}
