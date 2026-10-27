#include <torch/extension.h>

void holoq_nunchaku_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor as,
                         torch::Tensor ws, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &holoq_nunchaku_gemm);
}
