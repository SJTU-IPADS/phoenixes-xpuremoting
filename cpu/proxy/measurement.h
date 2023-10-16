#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#define MEASUREMENT_DETAILED_SWITCH

#include <sys/time.h>
#include <time.h>

#define TIMETYPE 3
enum {
    TOTAL_TIME = 0,
    SERIALIZATION_TIME,
    NETWORK_TIME
};

typedef struct _detailed_info {
    int id;
    int cnt;
    long long time[TIMETYPE];
    struct timeval start[TIMETYPE], end[TIMETYPE];
    long long payload_size;
} detailed_info;

#define API_COUNT 6000

void add_cnt(detailed_info *infos, int id);
void set_start(detailed_info *infos, int id, int type, struct timeval *start);
void set_end(detailed_info *infos, int id, int type, struct timeval *end);
void time_start(detailed_info *infos, int id, int type);
void time_end(detailed_info *infos, int id, int type);
void add_payload_size(detailed_info *infos, int id, long long size);
void print_detailed_info(detailed_info *infos, int length, const char* str);

#endif /* MEASUREMENT_H */
