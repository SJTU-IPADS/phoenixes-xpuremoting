#ifndef CPU_MEASUREMENT_H
#define CPU_MEASUREMENT_H

#define CPU_MEASUREMRNT_SWITCH

#include "proxy/rdtscp.h"

typedef struct _measurement_info {
    int id;
    int cnt;
    long long time;
    uint64_t start;
} cpu_measurement_info;

void cpu_time_start(cpu_measurement_info *infos, int id);
void cpu_time_end(cpu_measurement_info *infos, int id);
void cpu_print_measurement_info(const char *str, cpu_measurement_info *infos,
                            int length);

#define CPU_API_COUNT 6000

#ifdef NO_OPTIMIZATION
extern "C" {
void startTrace();
}
#endif

#endif // CPU_MEASUREMENT_H