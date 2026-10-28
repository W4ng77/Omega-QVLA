"""Experimental faithful W4A4 storage backend, with BF16 Tensor Core compute.

This backend preserves the original quantized weight values and the caller's
per-channel activation scale. It does NOT use INT4 MMA, requantize weights,
fold rotations, average timestep scales, or dispatch large M to dense GEMM.
The first version consumes already fake-quantized activations so reference
rounding is preserved; packing and GEMM are separate GPU kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _pack_a4(X, S, P, K: tl.constexpr, TOTAL: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    k = (2 * i) % K
    s0 = tl.load(S + k, i < TOTAL, other=1).to(tl.float32)
    s1 = tl.load(S + k + 1, i < TOTAL, other=1).to(tl.float32)
    x0 = tl.load(X + 2 * i, i < TOTAL, other=0).to(tl.float32)
    x1 = tl.load(X + 2 * i + 1, i < TOTAL, other=0).to(tl.float32)
    # Recover the existing A4 codes, rather than estimate a new scale.
    q0 = tl.minimum(tl.maximum(libdevice.rint(x0 / s0), -8), 7).to(tl.int32) + 8
    q1 = tl.minimum(tl.maximum(libdevice.rint(x1 / s1), -8), 7).to(tl.int32) + 8
    tl.store(P + i, (q0 | (q1 << 4)).to(tl.uint8), i < TOTAL)


@triton.jit
def _gemm_a4w4(
    A, W, CBOOK, SA, Y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
    A_FP16: tl.constexpr, A_BF16: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        pa = tl.load(A + m[:, None] * (K // 2) + k[None, :] // 2,
                     (m[:, None] < M) & (k[None, :] < K), other=136)
        qa = ((pa.to(tl.int32) >> ((k[None, :] % 2) * 4)) & 15) - 8
        sa = tl.load(SA + k, k < K, other=0).to(tl.float32)
        a = qa.to(tl.float32) * sa[None, :]
        # Match the fake-quant output's rounding before BF16 GEMM promotion.
        if A_FP16:
            a = a.to(tl.float16).to(tl.float32)
        elif A_BF16:
            a = a.to(tl.bfloat16).to(tl.float32)

        pw = tl.load(W + n[None, :] * (K // 2) + k[:, None] // 2,
                     (n[None, :] < N) & (k[:, None] < K), other=0)
        qw = (pw.to(tl.int32) >> ((k[:, None] % 2) * 4)) & 15
        group = n[None, :] * (K // G) + k[:, None] // G
        w = tl.load(CBOOK + group * 16 + qw,
                    (n[None, :] < N) & (k[:, None] < K), other=0)
        acc += tl.dot(a.to(tl.bfloat16), w.to(tl.bfloat16))
    tl.store(Y + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@torch.library.custom_op("gr00t::packed_w4a4_bf16_gemm", mutates_args=())
def packed_w4a4_linear(
    x_q: torch.Tensor, scale: torch.Tensor, packed_w: torch.Tensor,
    codebook: torch.Tensor, group_size: int,
) -> torch.Tensor:
    """Already A4-quantized [M,K] -> BF16 [M,N], for every M."""
    if x_q.ndim != 2 or not x_q.is_cuda or not x_q.is_contiguous():
        raise ValueError("Expected a contiguous CUDA [M,K] activation")
    m, k = x_q.shape
    n = packed_w.shape[0]
    if k % 2 or k % group_size or packed_w.shape[1] != k // 2:
        raise ValueError("Incompatible packed weight or group size")
    if scale.shape != (k,) or codebook.shape != (n * (k // group_size), 16):
        raise ValueError("Expected one activation scale per K and one codebook per weight group")
    packed_a = torch.empty((m, k // 2), device=x_q.device, dtype=torch.uint8)
    _pack_a4[(triton.cdiv(m * k // 2, 256),)](
        x_q, scale, packed_a, k, m * k // 2, 256,
    )
    out = torch.empty((m, n), device=x_q.device, dtype=torch.bfloat16)
    _gemm_a4w4[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
        packed_a, packed_w, codebook, scale, out, m, n, k, group_size,
        x_q.dtype == torch.float16, x_q.dtype == torch.bfloat16,
        32, 64, 64, num_warps=4, num_stages=2,
    )
    return out


@packed_w4a4_linear.register_fake
def _fake(x_q, scale, packed_w, codebook, group_size):
    return x_q.new_empty((x_q.shape[0], packed_w.shape[0]), dtype=torch.bfloat16)


def weight_codebooks(record: dict, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode metadata into exact small lookup tables; never requantize W."""
    encoding = record["encoding"]
    if encoding == "int4_codebook_v1":
        table = record["codebook"].to(dtype).contiguous()
    elif encoding == "int4_symmetric_v1":
        s = record["scales"].float()
        # unpack_int4 first rounds to the pack's stored dtype; the model loader
        # then casts to its own dtype. FP32 -> BF16 is NOT equivalent to
        # FP32 -> FP16 -> BF16 at halfway values.
        stored_dtype = record["fallback_codebooks"].dtype
        table = (s[:, None] * (torch.arange(16, device=s.device) - 8)).to(stored_dtype).to(dtype)
        ids = record["fallback_indices"].long()
        if ids.numel():
            table[ids] = record["fallback_codebooks"].to(dtype)
    else:
        raise ValueError(f"Unsupported lossless W4 encoding: {encoding}")
    n, k = record["shape"]
    return record["packed"].reshape(n, k // 2).contiguous(), table.contiguous()


@torch.no_grad()
def enable_packed_w4a4(model: torch.nn.Module, records: dict, version: int = 1) -> dict:
    """Install after reference calibration; fail instead of silently falling back.

    This first version requires A4 tables, rank zero, and an existing lossless
    weight pack for every selected GptqLinear. Input/output rotations stay in
    GptqLinear.forward. Dense quantized weights and GEMM caches are released.
    """
    from .gptq_layers import GptqLinear
    from .int4_storage import unpack_int4

    if version not in (1, 2, 3):
        raise ValueError("Expected backend version 1, 2 or 3")

    layers = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, GptqLinear)]
    prepared = []
    storage_bytes = 0
    for name, mod in layers:
        bits = mod._record_a_bits if mod._record_a_bits is not None else mod.cfg.act_bits
        if bits != 4 or mod.weight_bits != 4 or not mod._has_act_scale_table:
            raise ValueError(f"{name}: requires W4A4 and a calibrated scale table")
        if getattr(mod, "_fast", False) or mod._fused_enabled or mod._has_smooth_scale:
            raise ValueError(f"{name}: disable frozen/fused/SmoothQuant paths for reference rounding")
        if mod._has_svdquant and mod._lowrank_A.numel():
            raise ValueError(f"{name}: lowrank variant has not been validated")
        rec = records[name]["weight_res_q_int4"]
        # Match the dense loader exactly, including its cast to model dtype.
        expected = unpack_int4(rec).to(dtype=mod._weight_q.dtype)
        if not torch.equal(expected.to(mod._weight_q.device), mod._weight_q):
            raise ValueError(f"{name}: packed and reference weights differ")
        packed, table = weight_codebooks(rec, mod._weight_q.dtype)
        # Validate the actual codebook representation too, including the pack
        # dtype -> model dtype rounding sequence, before releasing dense W.
        group_size = int(rec["group_size"])
        groups = packed.reshape(-1, group_size // 2)
        expected_groups = expected.reshape(-1, group_size)
        for start in range(0, groups.shape[0], 8192):
            chunk = groups[start:start + 8192]
            codes = torch.stack((chunk & 15, chunk >> 4), -1).reshape(chunk.shape[0], group_size)
            decoded = table[start:start + 8192].gather(1, codes.long())
            if not torch.equal(decoded, expected_groups[start:start + 8192]):
                raise ValueError(f"{name}: codebook decoding changed a quantized weight")
        del expected, expected_groups, decoded, codes
        packed = packed.to(mod._weight_q.device)
        table = table.to(mod._weight_q.device) if version == 1 else table.new_empty(0).to(mod._weight_q.device)
        storage_bytes += packed.numel() * packed.element_size() + table.numel() * table.element_size()
        metadata = None
        if version in (2, 3):
            from .packed_w4a4_fast import symmetric_metadata

            scales, fallback_map, fallback_codebook, kind = symmetric_metadata(rec, mod._weight_q.dtype)
            metadata = (scales.to(mod._weight_q.device), fallback_map.to(mod._weight_q.device),
                        fallback_codebook.to(mod._weight_q.device), kind)
            storage_bytes += sum(t.numel() * t.element_size() for t in metadata[:3])
        prepared.append((mod, packed, table, group_size, metadata))
    if not prepared:
        raise ValueError("No GptqLinear layers found")
    for mod, packed, table, group_size, metadata in prepared:
        mod.register_buffer("_w4a4_packed_weight", packed, persistent=False)
        mod.register_buffer("_w4a4_codebook", table, persistent=False)
        mod._w4a4_group_size = group_size
        mod._packed_w4a4_version = version
        if metadata is not None:
            scales, fallback_map, fallback_codebook, kind = metadata
            mod.register_buffer("_w4a4_scales", scales, persistent=False)
            mod.register_buffer("_w4a4_fallback_map", fallback_map, persistent=False)
            mod.register_buffer("_w4a4_fallback_codebook", fallback_codebook, persistent=False)
            mod._w4a4_stored_kind = kind
        mod._packed_w4a4_enabled = True
        mod._strict_perstep = True
        mod._weight_q = mod._weight_q.new_empty(0)
        mod._w_promoted_cache = None
    return {"layers": len(prepared), "version": version, "weight_storage_bytes_with_codebooks": storage_bytes,
            "compute": "BF16 Tensor Core, FP32 accumulation", "activation_codes": "INT4 packed",
            "large_m_fallback": False, "weight_requantization": False,
            "gemm_strategy": "global decode + cuBLAS" if version == 3 else "on-chip decode + Triton dot",
            "decoded_scratch": "full BF16 A/W per call" if version == 3 else "on-chip tiles",
            "all_weight_values_bitwise_verified": True}
