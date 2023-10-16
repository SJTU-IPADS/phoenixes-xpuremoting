#include "measurement.h"
#include <algorithm>
#include <cstring>
#include <iostream>

static long long time_diff(const struct timeval *t1, const struct timeval *t2)
{
    return 1ll * 1000000 * (t2->tv_sec - t1->tv_sec) +
           (t2->tv_usec - t1->tv_usec);
}

static int detailed_info_cmp_time(const void *a, const void *b)
{
    return ((const detailed_info *)a)->cnt < ((const detailed_info *)b)->cnt;
}

void add_cnt(detailed_info *infos, int id)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    infos[id].id = id;
    ++infos[id].cnt;
#endif
}

void set_start(detailed_info *infos, int id, int type, struct timeval *start)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    infos[id].start[type] = *start;
#endif
}

void set_end(detailed_info *infos, int id, int type, struct timeval *end)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    infos[id].end[type] = *end;
#endif
}

void time_start(detailed_info *infos, int id, int type)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    gettimeofday(&infos[id].start[type], NULL);
#endif
}

void time_end(detailed_info *infos, int id, int type)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    gettimeofday(&infos[id].end[type], NULL);
    infos[id].time[type] +=
        time_diff(&infos[id].start[type], &infos[id].end[type]);
#endif
}

void add_payload_size(detailed_info *infos, int id, long long size)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    infos[id].payload_size += size;
#endif
}

void print_detailed_info(detailed_info *infos, int length, const char *str)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    std::cout << "----" << str << " detailed infos----" << std::endl;
    qsort(infos, length, sizeof(detailed_info), detailed_info_cmp_time);
    for (int i = 0; i < length; ++i) {
        if (infos[i].cnt == 0)
            break;
        printf("api %d: count %d, payload_size %lf, total_time %lf, "
               "serialization_time %lf, network_time %lf\n",
               infos[i].id, infos[i].cnt,
               1.0 * infos[i].payload_size / infos[i].cnt,
               1.0 * infos[i].time[TOTAL_TIME] / infos[i].cnt,
               1.0 * infos[i].time[SERIALIZATION_TIME] / infos[i].cnt,
               1.0 * infos[i].time[NETWORK_TIME] / infos[i].cnt);
        infos[i].cnt = 0;
        infos[i].payload_size = 0;
        memset(infos[i].time, 0, sizeof(infos[i].time));
    }
#endif
}
