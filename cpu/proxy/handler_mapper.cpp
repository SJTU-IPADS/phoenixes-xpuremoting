#include "handler_mapper.h"

HandlerMapper::HandlerMapper() {}

HandlerMapper::~HandlerMapper() {}

void *HandlerMapper::getMapping(void *virtual_handler)
{
    if (handler_map.find(virtual_handler) == handler_map.end()) {
        printf("%s:%d: no handler found for %p\n", __func__, __LINE__,
               virtual_handler);
        exit(1);
    }
    return handler_map[virtual_handler];
}

void HandlerMapper::setMapping(void *virtual_handler, void *physical_handler)
{
    handler_map[virtual_handler] = physical_handler;
}

void HandlerMapper::removeMapping(void *virtual_handler)
{
    if (handler_map.find(virtual_handler) == handler_map.end()) {
        printf("%s:%d: no handler found for %p\n", __func__, __LINE__,
               virtual_handler);
        exit(1);
    }
    handler_map.erase(virtual_handler);
}
