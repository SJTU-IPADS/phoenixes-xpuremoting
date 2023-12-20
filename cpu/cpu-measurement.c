#include "cpu-measurement.h"
#include <stdio.h>
#include <stdlib.h>

cpu_measurement_info totals[CPU_API_COUNT];
cpu_measurement_info vanillas[CPU_API_COUNT];

static int measurement_info_cmp_time(const void *a, const void *b)
{
    return ((const cpu_measurement_info *)a)->time <
           ((const cpu_measurement_info *)b)->time;
}

void cpu_time_start(cpu_measurement_info *infos, int id){
#ifdef CPU_MEASUREMRNT_SWITCH
    infos[id].id = id;
    infos[id].cnt++;
    infos[id].start = rdtscp();
#endif
}

void cpu_time_end(cpu_measurement_info *infos, int id){
#ifdef CPU_MEASUREMRNT_SWITCH
    infos[id].time += cycles_2_ns(rdtscp() - infos[id].start);
#endif
}

void cpu_print_measurement_info(const char *str, cpu_measurement_info *infos,
                            int length)
{
#ifdef CPU_MEASUREMRNT_SWITCH
    printf("----%sinfos----\n", str);
    qsort(infos, length, sizeof(cpu_measurement_info), measurement_info_cmp_time);
    for (int i = 0; i < length; ++i) {
        if (infos[i].cnt == 0)
            break;
        printf("api %d: count %d, %stime %lf\n", infos[i].id, infos[i].cnt,
               str, 1.0 * infos[i].time / infos[i].cnt);
    }
#endif
}

#ifdef NO_OPTIMIZATION
extern "C" {
void startTrace() {}
}
#endif
