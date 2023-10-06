#ifndef _RESOURCE_MG_H_
#define _RESOURCE_MG_H_

#include "list.h"

typedef struct resource_mg_map_elem_t {
    void* client_address;
    void* cuda_address;
} resource_mg_map_elem;

typedef struct resource_mg_t {
    /* Restored resources where client address != cuda address
     * are stored here. This is a sorted list, enabling binary searching.
     * It contains elements of type resource_mg_map_elem
     */
    list map_res;
    /* During this run created resources where we use actual addresses on
     * the client side. This is an unordered list. We never have to search
     * this though. It containts elements of type void*.
     */
    list new_res;
    int bypass;
} resource_mg;


//Runtime API RMs
extern resource_mg rm_streams;
extern resource_mg rm_events;
extern resource_mg rm_arrays;
extern resource_mg rm_memory;
extern resource_mg rm_kernels;

//Driver API RMs
extern resource_mg rm_modules;
extern resource_mg rm_functions;
extern resource_mg rm_globals;

//Other RMs
extern resource_mg rm_cusolver;
extern resource_mg rm_cublas;

//CUDNN RMs
extern resource_mg rm_cudnn;
extern resource_mg rm_cudnn_tensors;
extern resource_mg rm_cudnn_filters;
extern resource_mg rm_cudnn_tensortransform;
extern resource_mg rm_cudnn_poolings;
extern resource_mg rm_cudnn_activations;
extern resource_mg rm_cudnn_lrns;
extern resource_mg rm_cudnn_convs;
extern resource_mg rm_cudnn_backendds;


/** initializes the resource manager
 *
 * @bypass: if unequal to zero, searches for resources
 * will be bypassed, reducing the overhead. This is useful
 * for the original launch of an application as resources still
 * use their original pointers
 * @return 0 on success
 **/
int resource_mg_init(resource_mg *mg, int bypass);
void resource_mg_free(resource_mg *mg);

int resource_mg_add_sorted(resource_mg *mg, void* client_address, void* cuda_address);
int resource_mg_create(resource_mg *mg, void* cuda_address);

void* resource_mg_get(resource_mg *mg, void* client_address);

void resource_mg_print(resource_mg *mg);

#endif //_RESOURCE_MG_H_
