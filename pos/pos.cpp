#include <iostream>
#include <memory>
#include <string>

#include "./pos.h"

void POSWorkspace::clear() {
  shutdown = true;
  std::cout << "POSWorkspace cleared" << '\n';
}

void POSWorkspace::init() {
  shutdown = false;
  if (job_name == "") {
    job_name = "default_test";
  }
  std::cout << "POSWorkspace inited with job_name: " << job_name << '\n';
}

int POSWorkspace::pos_process(uint64_t api_id, uint64_t uuid,
                              std::vector<POSAPIParamDesp_t> param_desps) {
  std::cout << "params:\n";
  for (auto param : param_desps) {
    std::cout << '\t' << param.value << ' ' << param.size << '\n';
  }
  return 0;
}

// FFI
void *new_pos_workspace(int argc, char *argv[]) {
  auto pos = new POSWorkspace(argc, argv);
  return (void *)pos;
}

void pos_workspace_init(POSWorkspace *workspace) {
  workspace->init();
}

int call_pos_process(POSWorkspace *workspace, uint64_t api_id,
                              uint64_t uuid, uint64_t *param_desps, int param_num) {
  std::cout << "api_id: " << api_id << ", uuid: " << uuid << ", param_num: " << param_num << '\n';
  std::vector<POSAPIParamDesp_t> params(param_num);
  for (int i = 0; i < param_num; i++) {
    params[i] =
        POSAPIParamDesp_t{(void *)param_desps[2 * i], param_desps[2 * i + 1]};
  }
  return workspace->pos_process(api_id, uuid, params);
}
