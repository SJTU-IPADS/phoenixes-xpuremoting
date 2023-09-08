#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#define MEASUREMRNT_SWITCH

#include <sys/time.h>
#include <time.h>

typedef struct _measurement_info {
    int id;
    int cnt;
    long long time;
    struct timeval start, end;
} measurement_info;

void time_start(measurement_info *infos, int id);
void time_end(measurement_info *infos, int id);
void print_measurement_info(const char *str, measurement_info *infos,
                            int length);

#endif // MEASUREMENT_H