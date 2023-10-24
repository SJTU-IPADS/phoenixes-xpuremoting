#ifndef ASYNC_BATCH_CALLER_H
#define ASYNC_BATCH_CALLER_H

#include <vector>
#include <deque>

class AsyncCall {
    public:
        int device_num;
        int proc_id;
        std::vector<char> payload;
        AsyncCall(int proc_id, int device_num, std::vector<char>& payload) : proc_id(proc_id), device_num(device_num), payload(payload) {}
        AsyncCall(int proc_id, int device_num, std::vector<char>&& payload) : proc_id(proc_id), device_num(device_num), payload(payload) {}
};

class AsyncBatch {
    public:
        int Size();
        AsyncCall& Front();
        void Pop();
        void Clear();
        void Push(AsyncCall& call);
        // AsyncBatch();
    private:
        std::deque<AsyncCall> queue = {};
};

// extern AsyncBatch batch;

#endif
