#include "api_trace.h"
#include <iostream>

int on_trace = 0;
extern "C" void startTrace() { on_trace = 1; }

APITrace::APITrace(const int id, const uint64_t interval)
    : api_id(id), interval(interval)
{
}

std::map<int, std::string> *api_dict;
std::vector<APITrace> *api_traces;

TraceProfile::TraceProfile(int id)
{
    if (!on_trace) {
        return;
    }
    api_id = id;
    call_start = rdtscp();
}

TraceProfile::~TraceProfile()
{
    if (!on_trace) {
        return;
    }
    api_traces->emplace_back(api_id, cycles_2_ns(rdtscp() - call_start));
}

static void add_api(int api_id, const std::string &api_name)
{
    (*api_dict)[api_id] = api_name;
}

static void print_api_traces()
{
    std::map<std::string, int> api_count;
    for (auto &api_trace : *api_traces) {
        if (api_dict->find(api_trace.api_id) == api_dict->end()) {
            std::cout << "[ERROR] API " << api_trace.api_id
                      << " not set in dict!" << std::endl;
            exit(1);
        }
        std::string &api_name = (*api_dict)[api_trace.api_id];
        api_count[api_name]++;
        if (api_name.find("__cuda") != std::string::npos ||
            api_name.find("rpc_") != std::string::npos) {
            api_trace.api_id = -1;
        }
    }

    std::cout << std::endl << std::endl << "API Lists:" << std::endl;
    for (auto &api : api_count) {
        std::cout << api.first << " " << api.second << std::endl;
    }

    std::cout << std::endl
              << std::endl
              << "API Traces: api, interval(ns)" << std::endl;
    for (auto &api_trace : *api_traces) {
        if (api_trace.api_id == -1) {
            continue;
        }
        std::cout << (*api_dict)[api_trace.api_id] << " " << api_trace.interval
                  << std::endl;
    }
}

void init_api_traces()
{
#ifdef API_TRACE_SWITCH
    api_dict = new std::map<int, std::string>();
    api_traces = new std::vector<APITrace>();

    add_api(1, "rpc_deinit");
    add_api(2, "rpc_printmessage");
    add_api(50, "__cudaRegisterFunction");
    add_api(51, "__cudaRegisterFatBinary");
    add_api(53, "__cudaRegisterVar");
    add_api(1010, "cuCtxGetCurrent");
    add_api(1022, "cuDevicePrimaryCtxGetState");
    add_api(1018, "cuLaunchKernel");
    add_api(1013, "cuModuleGetFunction");
    add_api(1026, "cuModuleLoadData");
    add_api(3001, "cublasCreate_v2");
    add_api(3010, "cublasSetMathMode");
    add_api(3008, "cublasSetStream_v2");
    add_api(3011, "cublasSgemmStridedBatched");
    add_api(3004, "cublasSgemm_v2");
    add_api(102, "cudaDeviceGetAttribute");
    add_api(310, "cudaFuncGetAttributes");
    add_api(117, "cudaGetDevice");
    add_api(118, "cudaGetDeviceCount");
    add_api(120, "cudaGetDeviceProperties");
    add_api(142, "cudaGetLastError");
    add_api(317, "cudaLaunchKernel");
    add_api(414, "cudaMalloc");
    add_api(440, "cudaMemcpyAsync");
    add_api(441, "cudaMemcpyAsync");
    add_api(443, "cudaMemcpyAsync");
    add_api(475, "cudaMemsetAsync");
    add_api(332, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    add_api(143, "cudaPeekAtLastError");
    add_api(264, "cudaStreamIsCapturing");
    add_api(267, "cudaStreamSynchronize");
    add_api(5319, "cudnnBatchNormalizationForwardInference");
    add_api(5308, "cudnnConvolutionForward");
    add_api(5006, "cudnnCreate");
    add_api(5301, "cudnnCreateConvolutionDescriptor");
    add_api(5041, "cudnnCreateFilterDescriptor");
    add_api(5010, "cudnnCreateTensorDescriptor");
    add_api(5302, "cudnnDestroyConvolutionDescriptor");
    add_api(5048, "cudnnDestroyFilterDescriptor");
    add_api(5018, "cudnnDestroyTensorDescriptor");
    add_api(5305, "cudnnGetConvolutionForwardAlgorithm_v7");
    add_api(5309, "cudnnSetConvolutionGroupCount");
    add_api(5310, "cudnnSetConvolutionMathType");
    add_api(5304, "cudnnSetConvolutionNdDescriptor");
    add_api(5044, "cudnnSetFilterNdDescriptor");
    add_api(5008, "cudnnSetStream");
    add_api(5014, "cudnnSetTensorNdDescriptor");
    add_api(4000, "nvmlDeviceGetCount_v2");
    add_api(4001, "nvmlInitWithFlags");
    add_api(4002, "nvmlInit_v2");
#endif
}

void deinit_api_traces()
{
#ifdef API_TRACE_SWITCH
    print_api_traces();
    delete api_dict;
    delete api_traces;
#endif
}
