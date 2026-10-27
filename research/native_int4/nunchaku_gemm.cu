// Official Nunchaku kernel and default epilogue, with checked Torch pointers.
// No SVDQuant low-rank branch, activation recalibration or weight requantization.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernels/zgemm/gemm_w4a4.cuh"

void holoq_nunchaku_gemm(at::Tensor a, at::Tensor b, at::Tensor as,
                         at::Tensor ws, at::Tensor out) {
  using namespace nunchaku::kernels;
  using G = GEMM_W4A4<GEMMConfig_W4A4_FP16>;
  using E = G::EpilogueDefault;
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && as.is_cuda() && ws.is_cuda() && out.is_cuda());
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && as.is_contiguous() && ws.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(a.scalar_type() == at::kInt && b.scalar_type() == at::kInt);
  TORCH_CHECK(as.scalar_type() == at::kHalf && ws.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf);
  int M = a.size(0), N = b.size(0), K = a.size(1) * 8;
  TORCH_CHECK(M % G::BLOCK_M == 0 && N % G::BLOCK_N == 0 && K % 64 == 0);
  TORCH_CHECK(b.size(1) * 8 == K && as.numel() == M * K / 64 && ws.numel() == N * K / 64);
  TORCH_CHECK(out.size(0) <= M && M - out.size(0) < G::BLOCK_M && out.size(1) == N);
  auto fn = invoke_kernel<typename G::template gemm_w4a4_kernel<E, false>,
      const G::packed_act_t*, const G::packed_wgt_t*, const G::packed_ascale_t*,
      const G::packed_wscale_t*, int, int, int, E::Arguments, bool, bool>;
  dim3 grid(M / G::BLOCK_M, N / G::BLOCK_N);
  fn<<<grid, G::WARP_SIZE * G::NUM_WARPS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const G::packed_act_t*>(a.data_ptr<int>()),
      reinterpret_cast<const G::packed_wgt_t*>(b.data_ptr<int>()),
      reinterpret_cast<const G::packed_ascale_t*>(as.data_ptr<at::Half>()),
      reinterpret_cast<const G::packed_wscale_t*>(ws.data_ptr<at::Half>()),
      M, N, K, E::Arguments{reinterpret_cast<half*>(out.data_ptr<at::Half>()), int(out.size(0)), N}, false, false);
  checkCUDA(cudaGetLastError());
}
