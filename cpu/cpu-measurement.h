#ifndef CPU_MEASUREMENT_H
#define CPU_MEASUREMENT_H

#define CPU_MEASUREMRNT_SWITCH

#include <sys/time.h>
#include <time.h>

typedef struct _measurement_info {
    int id;
    int cnt;
    long long time;
    struct timeval start, end;
} cpu_measurement_info;

void cpu_time_start(cpu_measurement_info *infos, int id);
void cpu_time_end(cpu_measurement_info *infos, int id);
void cpu_print_measurement_info(const char *str, cpu_measurement_info *infos,
                            int length);

#define CPU_API_COUNT 6000

#endif // CPU_MEASUREMENT_H