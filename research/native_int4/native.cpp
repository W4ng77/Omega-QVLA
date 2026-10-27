#include <torch/extension.h>

torch::Tensor group_mma(torch::Tensor a, torch::Tensor b, torch::Tensor scales,
                        torch::Tensor act, int64_t m, int64_t n, int64_t k,
                        int64_t group, int64_t mode, int64_t warps_n, int64_t split_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("group_mma", &group_mma, "Packed S4 x S4 group-scaled MMA (research)");
}
