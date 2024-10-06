#pragma once

#include <memory>
#include <string>
#include <vector>

typedef struct POSAPIParamDesp {
  void *value;
  size_t size;
} POSAPIParamDesp_t;

class POSWorkspace {
public:
  POSWorkspace(int argc, char *argv[]) {
    if (argc > 1) {
      job_name = std::string(argv[argc - 1]);
    }
    process_cnt = 0;
  }

  ~POSWorkspace() { clear(); }

  void clear();
  void init();
  int pos_process(uint64_t api_id, uint64_t uuid,
                  std::vector<POSAPIParamDesp_t> param_desps);

private:
  std::string job_name;
  bool shutdown;
  int process_cnt;
};

extern "C" {
void *new_pos_workspace(int argc, char *argv[]);
void pos_workspace_init(POSWorkspace *workspace);
int call_pos_process(POSWorkspace *workspace, uint64_t api_id,
                              uint64_t uuid, uint64_t *param_desps,
                              int param_num);
}
