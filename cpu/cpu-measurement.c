#include "cpu-measurement.h"
#include <stdio.h>
#include <stdlib.h>

measurement_info totals[6000];
measurement_info vanillas[6000];

static long long time_diff(const struct timeval *t1, const struct timeval *t2)
{
    return 1ll * 1000000 * (t2->tv_sec - t1->tv_sec) +
           (t2->tv_usec - t1->tv_usec);
}

static int measurement_info_cmp_time(const void *a, const void *b)
{
    return ((const measurement_info *)a)->time <
           ((const measurement_info *)b)->time;
}

void time_start(measurement_info *infos, int id){
#ifdef MEASUREMRNT_SWITCH
    infos[id].id = id;
    infos[id].cnt++;
    gettimeofday(&infos[id].start, NULL);
#endif
}

void time_end(measurement_info *infos, int id){
#ifdef MEASUREMRNT_SWITCH
    gettimeofday(&infos[id].end, NULL);
    infos[id].time += time_diff(&infos[id].start, &infos[id].end);
#endif
}

void print_measurement_info(const char *str, measurement_info *infos,
                            int length)
{
#ifdef MEASUREMRNT_SWITCH
    printf("----%sinfos----\n", str);
    qsort(infos, length, sizeof(measurement_info), measurement_info_cmp_time);
    for (int i = 0; i < length; ++i) {
        if (infos[i].cnt == 0)
            break;
        printf("api %d: count %d, %stime %lf\n", infos[i].id, infos[i].cnt,
               str, 1.0 * infos[i].time / infos[i].cnt);
    }
#endif
}
