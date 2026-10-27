#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fragment mapping follows PTX ISA 8.4, mma.m16n8k64 (.s4), section 9.7.13.4.11.
// A/B are prepacked into coalesced [fragment-word, lane] planes. No INT8
// expansion, shared-memory fragment conversion or activation quantization in
// this GEMM. Per-group FP32 scales remain inside the K reduction.
__device__ __forceinline__ void mma_s4(int &c0, int &c1, int &c2, int &c3,
                                     uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                     uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template<int WN, bool GROUPED>
__global__ void native_group_kernel(const uint32_t *__restrict__ A,
                                    const uint32_t *__restrict__ B,
                                    const float *__restrict__ S,
                                    const float *__restrict__ AS,
                                    float *__restrict__ Y,
                                    int M, int N, int K64, int NG, int G64, int SK) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int mt = blockIdx.x;
  int nt = blockIdx.y * WN + warp; // Each warp computes one 16x8 output tile.
  if (nt >= (N + 7) / 8) return;
  int n0 = nt * 8 + (lane & 3) * 2;
  int n1 = n0 + 1;
  int m0 = mt * 16 + (lane >> 2);
  int m1 = m0 + 8;
  int begin = (NG * blockIdx.z) / SK;
  int end = (NG * (blockIdx.z + 1)) / SK;
  float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;
  for (int group = begin; group < end; ++group) {
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int sub = 0; sub < G64; ++sub) {
      int kb = group * G64 + sub;
      uint32_t a0=0, a1=0, a2=0, a3=0, b0=0, b1=0;
      if (kb < K64) {
        auto ap = A + ((mt * K64 + kb) * 4) * 32 + lane;
        auto bp = B + ((nt * K64 + kb) * 2) * 32 + lane;
        a0 = __ldg(ap); a1 = __ldg(ap + 32);
        a2 = __ldg(ap + 64); a3 = __ldg(ap + 96);
        b0 = __ldg(bp); b1 = __ldg(bp + 32);
      }
      mma_s4(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
    float s0 = n0 < N ? __ldg(S + group * N + n0) : 0.f;
    float s1 = n1 < N ? __ldg(S + group * N + n1) : 0.f;
    float p0 = __int2float_rn(c0), p1 = __int2float_rn(c1);
    float p2 = __int2float_rn(c2), p3 = __int2float_rn(c3);
    if (GROUPED) {
      float a = __ldg(AS + group);
      p0 = __fmul_rn(p0,a); p1 = __fmul_rn(p1,a);
      p2 = __fmul_rn(p2,a); p3 = __fmul_rn(p3,a);
    }
    f0 = fmaf(p0,s0,f0); f1 = fmaf(p1,s1,f1);
    f2 = fmaf(p2,s0,f2); f3 = fmaf(p3,s1,f3);
  }
  int64_t off = static_cast<int64_t>(blockIdx.z) * M * N;
  if (m0 < M && n0 < N) Y[off + m0 * N + n0] = f0;
  if (m0 < M && n1 < N) Y[off + m0 * N + n1] = f1;
  if (m1 < M && n0 < N) Y[off + m1 * N + n0] = f2;
  if (m1 < M && n1 < N) Y[off + m1 * N + n1] = f3;
}

at::Tensor group_mma(at::Tensor a, at::Tensor b, at::Tensor scales, at::Tensor act,
                     int64_t m, int64_t n, int64_t k, int64_t group,
                     int64_t mode, int64_t wn, int64_t sk) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && scales.is_cuda() && act.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(a.device()==b.device() && a.device()==scales.device() && a.device()==act.device(), "Device mismatch");
  TORCH_CHECK(a.scalar_type()==at::kInt && b.scalar_type()==at::kInt, "Packed words must be INT32");
  TORCH_CHECK(scales.scalar_type()==at::kFloat && act.scalar_type()==at::kFloat, "Scales must be FP32");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && scales.is_contiguous() && act.is_contiguous(), "Contiguous tensors required");
  TORCH_CHECK(m>0 && n>0 && k>0 && group>=64 && group%64==0, "Positive sizes and a 64-multiple group required");
  TORCH_CHECK(mode==1 || mode==2, "Supports step_weight=1 and group_int=2 only");
  int64_t kt = (k+63)/64, ng=(k+group-1)/group;
  TORCH_CHECK(a.numel()==((m+15)/16)*kt*128 && b.numel()==((n+7)/8)*kt*64, "Invalid MMA packing");
  TORCH_CHECK(scales.numel()==ng*n && sk>=1 && sk<=ng, "Invalid scales or split K");
  TORCH_CHECK(mode!=2 || act.numel()==ng, "Expected grouped activation scales");
  c10::cuda::CUDAGuard guard(a.device());
  auto y=at::empty({sk,m,n}, scales.options());
  dim3 grid((m+15)/16,(n+8*wn-1)/(8*wn),sk);
  auto stream=at::cuda::getCurrentCUDAStream();
  auto ap=reinterpret_cast<const uint32_t*>(a.data_ptr<int32_t>());
  auto bp=reinterpret_cast<const uint32_t*>(b.data_ptr<int32_t>());
  #define LAUNCH(W) do { \
    if(mode==2) native_group_kernel<W,true><<<grid,W*32,0,stream>>>(ap,bp,scales.data_ptr<float>(),act.data_ptr<float>(),y.data_ptr<float>(),m,n,kt,ng,group/64,sk); \
    else native_group_kernel<W,false><<<grid,W*32,0,stream>>>(ap,bp,scales.data_ptr<float>(),act.data_ptr<float>(),y.data_ptr<float>(),m,n,kt,ng,group/64,sk); \
  } while(0)
  switch(wn) {case 1: LAUNCH(1);break; case 2:LAUNCH(2);break; case 4:LAUNCH(4);break; case 8:LAUNCH(8);break;
    default: TORCH_CHECK(false,"warps_n must be 1,2,4,8");}
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
