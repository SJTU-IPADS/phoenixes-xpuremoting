#ifndef API_TRACE_H
#define API_TRACE_H

#include "rdtscp.h"
#include <map>
#include <vector>

// #define API_TRACE_SWITCH

extern "C" {
void startTrace();
}

class APITrace
{
public:
    int api_id;
    uint64_t interval;

    APITrace(const int id, const uint64_t interval);
};

class TraceProfile
{
public:
    TraceProfile(int id);
    ~TraceProfile();

private:
    int api_id;
    uint64_t call_start;

    TraceProfile(const TraceProfile &) = delete;
    void operator=(const TraceProfile &) = delete;
};

#ifdef API_TRACE_SWITCH
#define TRACE_PROFILE(id) TraceProfile _tp_##id_(id)
#else
#define TRACE_PROFILE(id)
#endif

void init_api_traces();
void deinit_api_traces();

#endif // API_TRACE_H
