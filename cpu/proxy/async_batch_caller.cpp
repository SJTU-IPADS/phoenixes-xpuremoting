#include "async_batch_caller.h"
#include <cassert>
#include <iostream>
#include <unordered_set>

int AsyncBatch::Size() {
    return queue.size();
}

AsyncCall& AsyncBatch::Front() {
    return queue.front();
}

void AsyncBatch::Pop() {
    queue.pop_front();
}

void AsyncBatch::Clear() {
    queue.clear();
}

void AsyncBatch::Push(AsyncCall& call) {
    queue.push_back(call);
}
