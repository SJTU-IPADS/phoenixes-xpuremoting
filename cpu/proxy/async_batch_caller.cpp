#include "async_batch_caller.h"
#include <cassert>
#include <iostream>

AsyncBatch batch;

int AsyncBatch::Size() {
    int size = queue.size();
    assert(size >= 0);
    return size;
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

// AsyncBatch::AsyncBatch() {
//     std::cout << "in " << __func__ << std::endl;
//     assert(0);
//     int size = queue.size();
//     assert(size >= 0);
// }
