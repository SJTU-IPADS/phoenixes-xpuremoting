#include "resource-mg.h"
#include "list.h"
#include "log.h"
#include <assert.h>


int resource_mg_init(resource_mg *mg, int bypass)
{
    if (mg == NULL) {
        LOGE(LOG_ERROR, "resource manager mg is NULL");
        return -1;
    }
    mg->bypass = bypass;
    return 0;
}

void resource_mg_free(resource_mg *mg)
{
}

int resource_mg_create(resource_mg *mg, void *cuda_address)
{
    if (mg == NULL) {
        LOGE(LOG_ERROR, "resource manager mg is NULL");
        return -1;
    }
    mg->new_resources.push_back(cuda_address);
    return 0;
}

static void* resource_mg_search_map(resource_mg *mg, void *client_address)
{
    if (mg == NULL) {
        LOGE(LOG_ERROR, "resource manager mg is NULL");
        return NULL;
    }
    auto it = mg->resources_mapping.find(client_address);
    if (it == mg->resources_mapping.end()) {
        LOGE(LOG_DEBUG, "no find: %p", client_address);
        return client_address;
    } else {
        return it->second;
    }
}

void resource_mg_print(resource_mg *mg)
{
    size_t i;
    resource_mg_map_elem *elem;
    if (mg == NULL) {
        LOGE(LOG_ERROR, "resource manager mg is NULL");
        return;
    }
    LOG(LOG_DEBUG, "new_res:");
    for (i = 0; i < mg->new_resources.size(); i++) {
        LOG(LOG_DEBUG, "%p", mg->new_resources[i]);
    }
    if (mg->bypass == 0) {
        LOG(LOG_DEBUG, "map_res:");
        for (auto [k, v] : mg->resources_mapping) {
            LOG(LOG_DEBUG, "%p -> %p", k, v);
        }
    }
}

void* resource_mg_get(resource_mg *mg, void* client_address)
{
    if (mg->bypass) {
        return client_address;
    } else {
        return resource_mg_search_map(mg, client_address);
    }
    return 0;
}

#include <stdio.h>
int resource_mg_add_sorted(resource_mg *mg, void* client_address, void* cuda_address)
{
    if (mg == NULL) {
        LOGE(LOG_ERROR, "resource manager mg is NULL");
        return 1;
    }
    if (mg->bypass) {
        LOGE(LOG_ERROR, "cannot add to bypassed resource manager");
        return 1;
    }
    auto it = mg->resources_mapping.find(client_address);
    if (it != mg->resources_mapping.end()) {
        LOGE(LOG_WARNING, "duplicate resource! The first resource will be overwritten");
    }
    mg->resources_mapping[client_address] = cuda_address;
    return 0;
}
