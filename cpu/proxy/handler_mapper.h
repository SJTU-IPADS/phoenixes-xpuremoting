#ifndef HANDLER_MAPPER_H
#define HANDLER_MAPPER_H

#include <unordered_map>

// mapper for virtual handler to physical handler
class HandlerMapper
{
public:
    HandlerMapper();
    ~HandlerMapper();

    void *getMapping(void *virtual_handler);
    void setMapping(void *virtual_handler, void *physical_handler);
    void removeMapping(void *virtual_handler);

private:
    std::unordered_map<void *, void *> handler_map;
};

extern HandlerMapper handler_mapper;

#endif // HANDLER_MAPPER_H
