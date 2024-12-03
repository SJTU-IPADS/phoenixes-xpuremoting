#ifndef TRACE_H
#define TRACE_H

#include <map>
#include <string>
#include <vector>
#include <cstdint>

__inline__ uint64_t rdtscp(void)
{
    uint32_t lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi));
    return ((uint64_t)lo) | (((uint64_t)hi) << 32);
}

__inline__ uint64_t cycles_2_ns(uint64_t cycles)
{
    static uint64_t hz = 2200000000;
    return cycles * (1000000000.0 / hz);
}

// #define HOOK_TRACE_SWITCH

extern "C" {
void startTrace();
}

class APIRecord {
public:
    std::string api_name;
    uint64_t interval;

    APIRecord(const std::string &name, const uint64_t itv) : api_name(name), interval(itv) {}
};

extern std::map<std::string, int> *api_dict;
extern std::vector<APIRecord> *api_records;

class TraceProfile {
public:
    TraceProfile(const std::string &name);
    ~TraceProfile();

private:
    std::string api_name;
    uint64_t call_start;

    TraceProfile(const TraceProfile &) = delete;
    void operator=(const TraceProfile &) = delete;
};

#ifdef HOOK_TRACE_SWITCH
#define HOOK_TRACE_PROFILE(name) TraceProfile _tp_##name_(name)
#else
#define HOOK_TRACE_PROFILE(name)
#endif

#endif  // TRACE_H
