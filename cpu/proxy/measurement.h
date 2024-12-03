#ifndef MEASUREMENT_H
#define MEASUREMENT_H

// #define MEASUREMENT_DETAILED_SWITCH

#include "rdtscp.h"

enum {
    TOTAL_TIME = 0,
    SERIALIZATION_TIME,
    NETWORK_SEND_TIME,
    NETWORK_RECEIVE_TIME,
    TIME_TYPE_COUNT
};

typedef struct _detailed_info {
    int id;
    int cnt;
    uint64_t time[TIME_TYPE_COUNT];
    uint64_t start[TIME_TYPE_COUNT];
    uint64_t payload_size;
} detailed_info;

#define API_COUNT 6000

void add_cnt(detailed_info *infos, int id);
void set_start(detailed_info *infos, int id, int type, uint64_t start);
void time_start(detailed_info *infos, int id, int type);
void time_end(detailed_info *infos, int id, int type);
void add_payload_size(detailed_info *infos, int id, long long size);
void print_detailed_info(detailed_info *infos, int length, const char *str);

#endif /* MEASUREMENT_H */
