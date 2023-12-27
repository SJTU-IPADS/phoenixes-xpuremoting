#include "measurement.h"
#include <algorithm>
#include <cstring>
#include <iostream>

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

void set_start(detailed_info *infos, int id, int type, uint64_t start)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    infos[id].start[type] = start;
#endif
}

void time_start(detailed_info *infos, int id, int type)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    infos[id].start[type] = rdtscp();
#endif
}

void time_end(detailed_info *infos, int id, int type)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    infos[id].time[type] += cycles_2_ns(rdtscp() - infos[id].start[type]);
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
               "serialization_time %lf, network_send_time %lf, network_receive_time %lf\n",
               infos[i].id, infos[i].cnt,
               1.0 * infos[i].payload_size / infos[i].cnt,
               1.0 * infos[i].time[TOTAL_TIME] / infos[i].cnt,
               1.0 * infos[i].time[SERIALIZATION_TIME] / infos[i].cnt,
               1.0 * infos[i].time[NETWORK_SEND_TIME] / infos[i].cnt,
               1.0 * infos[i].time[NETWORK_RECEIVE_TIME] / infos[i].cnt);
        infos[i].cnt = 0;
        infos[i].payload_size = 0;
        memset(infos[i].time, 0, sizeof(infos[i].time));
    }
    fflush(stdout);
#endif
}
