#ifndef TRACE_H
#define TRACE_H

#include <chrono>
#include <map>
#include <string>
#include <vector>

// #define HOOK_TRACE_SWITCH

extern "C" {
void startTrace();
}

class APIRecord {
public:
    std::string api_name;
    long interval;

    APIRecord(const std::string &name, const long itv) : api_name(name), interval(itv) {}
};

extern std::map<std::string, int> *api_dict;
extern std::vector<APIRecord> *api_records;

class TraceProfile {
public:
    TraceProfile(const std::string &name);
    ~TraceProfile();

private:
    std::string api_name;
    std::chrono::steady_clock::time_point call_start;

    TraceProfile(const TraceProfile &) = delete;
    void operator=(const TraceProfile &) = delete;
};

#ifdef HOOK_TRACE_SWITCH
#define HOOK_TRACE_PROFILE(name) TraceProfile _tp_##name_(name)
#else
#define HOOK_TRACE_PROFILE(name)
#endif

#endif  // TRACE_H
