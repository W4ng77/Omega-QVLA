#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <gemm.h>
at::Tensor matmul(at::Tensor a, at::Tensor b) {
 TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device()==b.device(), "Same CUDA device required");
 TORCH_CHECK(a.scalar_type()==at::kByte && b.scalar_type()==at::kByte, "Packed uint8 signed-nibble matrices required");
 TORCH_CHECK(a.dim()==2 && b.dim()==2 && a.size(1)==b.size(1) && a.is_contiguous() && b.is_contiguous(), "Invalid packed layout");
 c10::cuda::CUDAGuard guard(a.device());
 auto c=at::empty({a.size(0), b.size(0)},a.options().dtype(at::kInt));
 matmul_host(reinterpret_cast<const Int4Storage*>(a.data_ptr<uint8_t>()), reinterpret_cast<const Int4Storage*>(b.data_ptr<uint8_t>()),
             a.size(0), b.size(0), a.size(1)*2, c.data_ptr<int32_t>());
 return c;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("matmul",&matmul);}
