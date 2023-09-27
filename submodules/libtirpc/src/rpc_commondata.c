/*
 * Copyright (c) 2009, Sun Microsystems, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of Sun Microsystems, Inc. nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <rpc/rpc.h>
#include "rpc_com.h"


/*
 * This file should only contain common data (global data) that is exported
 * by public interfaces 
 */
struct opaque_auth _null_auth;
fd_set svc_fdset;
int svc_maxfd = -1;
struct pollfd *svc_pollfd;
int svc_max_pollfd;

#define MEASUREMENT_DETAILED
#ifdef MEASUREMENT_DETAILED
static long long time_diff(const struct timeval *t1, const struct timeval *t2)
{
    return 1ll * 1000000 * (t2->tv_sec - t1->tv_sec) +
           (t2->tv_usec - t1->tv_usec);
}

static int detailed_info_cmp_time(const void *a, const void *b)
{
    return ((const detailed_info *)a)->cnt <
           ((const detailed_info *)b)->cnt;
}

void add_cnt(detailed_info *infos, int id){
#ifdef MEASUREMENT_DETAILED_SWITCH
    infos[id].id = id;
    ++infos[id].cnt;
#endif
}

void time_start(detailed_info *infos, int id, int type){
#ifdef MEASUREMENT_DETAILED_SWITCH
    gettimeofday(&infos[id].start[type], NULL);
#endif
}

void time_end(detailed_info *infos, int id, int type){
#ifdef MEASUREMENT_DETAILED_SWITCH
    gettimeofday(&infos[id].end[type], NULL);
    infos[id].time[type] += time_diff(&infos[id].start[type], &infos[id].end[type]);
#endif
}

void add_payload_size(detailed_info *infos, int id, long long size){
#ifdef MEASUREMENT_DETAILED_SWITCH
    infos[id].payload_size += size;
#endif
}

void print_detailed_info(detailed_info *infos, int length, const char* str)
{
#ifdef MEASUREMENT_DETAILED_SWITCH
    printf("----%s detailed infos----\n", str);
    qsort(infos, length, sizeof(detailed_info), detailed_info_cmp_time);
    for (int i = 0; i < length; ++i) {
        if (infos[i].cnt == 0)
            break;
        printf("api %d: count %d, payload_size %lf, total_time %lf, "
               "serialization_time %lf, network_time %lf, tcpip_time %lf\n",
               infos[i].id, infos[i].cnt,
               1.0 * infos[i].payload_size / infos[i].cnt,
               1.0 * infos[i].time[0] / infos[i].cnt,
               1.0 * (infos[i].time[1] - infos[i].time[2]) / infos[i].cnt,
               1.0 * infos[i].time[2] / infos[i].cnt,
               1.0 * infos[i].time[3] / infos[i].cnt);
    }
#endif
}

#endif
