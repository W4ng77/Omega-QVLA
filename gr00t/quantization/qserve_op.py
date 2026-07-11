"""QServe W4A8 per-channel GEMM as a torch custom op (compile-friendly).

y[m, n] = s_a[m] * ( sum_k a_i8[m,k] * wq4[n,k] ) * s1[n]  -  s_a[m]*sum_i8[m]*szeros[n]
computed by omniserve_backend.qgemm_w4a8_per_chn on INT8 tensor cores with
int4-packed weights. input_sum convention: (sum_k a_i8[m,k]) * s_a[m], fp16.
"""

from __future__ import annotations

import torch

import omniserve_backend.qgemm_w4a8_per_chn as _qg


@torch.library.custom_op("gr00t::qserve_w4a8_gemm", mutates_args=())
def qserve_w4a8_gemm(
    a_i8: torch.Tensor,      # [M, K] int8
    qweight: torch.Tensor,   # [N, K//2] int8 packed
    s1: torch.Tensor,        # [N] fp16
    s_a: torch.Tensor,       # [M] fp16
    szeros: torch.Tensor,    # [N] fp16
    isum: torch.Tensor,      # [M] fp16  (= sum(a_i8) * s_a)
) -> torch.Tensor:
    out = torch.empty(a_i8.shape[0], qweight.shape[0], dtype=torch.float16, device=a_i8.device)
    _qg.gemm_forward_cuda(a_i8, qweight, s1, s_a, szeros, isum, out)
    return out


@qserve_w4a8_gemm.register_fake
def _(a_i8, qweight, s1, s_a, szeros, isum):
    return a_i8.new_empty((a_i8.shape[0], qweight.shape[0]), dtype=torch.float16)


@torch.library.custom_op("gr00t::fp8_rowwise_mm", mutates_args=())
def fp8_rowwise_mm(
    a8: torch.Tensor, w_t: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor
) -> torch.Tensor:
    return torch._scaled_mm(a8, w_t, scale_a=s_a, scale_b=s_w, out_dtype=torch.bfloat16)


@fp8_rowwise_mm.register_fake
def _(a8, w_t, s_a, s_w):
    return a8.new_empty((a8.shape[0], w_t.shape[1]), dtype=torch.bfloat16)


def qserve_linear(x2d: torch.Tensor, qweight: torch.Tensor, s1: torch.Tensor,
                  szeros: torch.Tensor) -> torch.Tensor:
    """fp16/bf16 [M, K] -> fp16 [M, N] with per-token dynamic A8."""
    xf = x2d.float()
    s_a = (xf.abs().amax(1) / 127.0).clamp_min(1e-6)
    a = torch.clamp(torch.round(xf / s_a[:, None]), -128, 127).to(torch.int8).contiguous()
    isum = (a.float().sum(1) * s_a).to(torch.float16).contiguous()
    return qserve_w4a8_gemm(a, qweight, s1, s_a.to(torch.float16).contiguous(), szeros, isum)
