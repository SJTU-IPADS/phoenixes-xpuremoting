#ifndef RDTSCP_H
#define RDTSCP_H

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

#endif /* RDTSCP_H */
