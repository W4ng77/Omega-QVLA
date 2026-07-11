"""Fused DuQuant block-rotation + activation fake-quant (Triton).

Replaces the GptqLinear rotation-blocks + fake_quantize_sym sequence
(index_select, 2x transpose+contiguous, bmm, round/clamp/mul ~ 10 kernels)
with a single kernel:

    out[t, b*B+j] = fq( sum_i x[t, perm[b*B+i]] * R[b, i, j] )

where fq(v) = clamp(round(v / s_j), -maxq-1, maxq) * s_j, and s optionally
folds a SmoothQuant per-channel divide (s_eff = smooth * scale).

Compute in fp32, IO in x.dtype. Enabled via GR00T_GPTQ_FUSED=1.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

try:  # triton >= 3.0
    from triton.language.extra import libdevice as _tld
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice as _tld


@triton.jit
def _rot_quant_kernel(
    x_ptr, perm_ptr, r_ptr, scale_ptr, out_ptr,
    n_tokens, in_features,
    stride_xt, stride_ot,
    maxq,
    B: tl.constexpr,
    T_TILE: tl.constexpr,
    DO_QUANT: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)

    t_off = pid_t * T_TILE + tl.arange(0, T_TILE)
    t_mask = t_off < n_tokens
    cols = pid_b * B + tl.arange(0, B)

    # gather column indices for this block
    src_cols = tl.load(perm_ptr + cols)  # [B]

    # x_tile: [T_TILE, B] gathered
    x_ptrs = x_ptr + t_off[:, None] * stride_xt + src_cols[None, :]
    x_tile = tl.load(x_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float32)

    # R block: [B, B] row-major, R[b, i, j]
    i = tl.arange(0, B)
    j = tl.arange(0, B)
    r_ptrs = r_ptr + pid_b * B * B + i[:, None] * B + j[None, :]
    R = tl.load(r_ptrs).to(tl.float32)

    y = tl.dot(x_tile, R, allow_tf32=False)  # [T_TILE, B]

    if DO_QUANT:
        s = tl.load(scale_ptr + cols).to(tl.float32)  # [B] per rotated channel
        s = tl.maximum(s, 1e-8)
        q = y / s[None, :]
        q = _tld.rint(q)
        q = tl.minimum(tl.maximum(q, -maxq - 1.0), maxq)
        y = q * s[None, :]

    out_ptrs = out_ptr + t_off[:, None] * stride_ot + cols[None, :]
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=t_mask[:, None])


def rot_quant(
    x2d: torch.Tensor,          # [N, in] contiguous rows
    perm: torch.Tensor,         # [in] int
    blocks: torch.Tensor,       # [n_blocks, B, B] float (row-major)
    scale: torch.Tensor | None,  # [in] per rotated channel, or None (rotation only)
    bits: int,
) -> torch.Tensor:
    assert x2d.dim() == 2
    n_tokens, in_f = x2d.shape
    n_blocks, B, _ = blocks.shape
    assert n_blocks * B == in_f, (n_blocks, B, in_f)
    out = torch.empty_like(x2d)
    do_quant = scale is not None and bits and bits > 0
    if do_quant:
        maxq = float((1 << (bits - 1)) - 1)
        scale_t = scale.to(device=x2d.device, dtype=torch.float32).reshape(-1)
        if scale_t.numel() == 1:
            scale_t = scale_t.expand(in_f)
        elif scale_t.numel() != in_f:
            raise ValueError(f"act scale numel {scale_t.numel()} != in_features {in_f}")
        scale_t = scale_t.contiguous()
    else:
        maxq = 0.0
        scale_t = torch.empty(1, device=x2d.device, dtype=torch.float32)

    T_TILE = 32
    grid = (triton.cdiv(n_tokens, T_TILE), n_blocks)
    _rot_quant_kernel[grid](
        x2d, perm.to(torch.int32).contiguous(),
        blocks.to(torch.float32).contiguous(),
        scale_t, out,
        n_tokens, in_f,
        x2d.stride(0), out.stride(0),
        maxq,
        B=B, T_TILE=T_TILE, DO_QUANT=bool(do_quant),
    )
    return out


@torch.no_grad()
def try_decompose_rotation(R: torch.Tensor, block: int = 64):
    """Decompose a dense DuQuant rotation into (perm, blocks) if it has the
    (permutation ∘ block-diagonal) structure. Returns (perm[int64], blocks
    [n,B,B]) or (None, None). Pure torch; used at pack-load time so the
    runtime can take the 32x-cheaper block bmm path instead of a dense matmul.
    """
    R = R.detach().to(torch.float32)
    in_f = R.shape[0]
    if R.dim() != 2 or R.shape[0] != R.shape[1] or in_f % block != 0:
        return None, None
    n_b = in_f // block
    e = (R.reshape(in_f, n_b, block) ** 2).sum(-1)  # [in, n_b] row energy per col-block
    row_blk = e.argmax(dim=1)
    off_energy = float(e.sum() - e.gather(1, row_blk[:, None]).sum())
    counts = torch.bincount(row_blk, minlength=n_b)
    if off_energy > 1e-4 or not bool((counts == block).all()):
        return None, None
    order = torch.argsort(row_blk * in_f + torch.arange(in_f, device=R.device))
    rows = R.index_select(0, order)
    blocks = torch.stack(
        [rows[b * block:(b + 1) * block, b * block:(b + 1) * block] for b in range(n_b)]
    ).contiguous()
    rec = torch.zeros_like(R)
    for b in range(n_b):
        rec[order[b * block:(b + 1) * block], b * block:(b + 1) * block] = blocks[b]
    if float((rec - R).abs().max()) > 1e-5:
        return None, None
    return order.to(torch.int64).contiguous(), blocks


def reference_rot_quant(x2d, perm, blocks, scale, bits):
    """Eager reference identical to GptqLinear's unfused path (for tests)."""
    n_tokens, in_f = x2d.shape
    n_blocks, B, _ = blocks.shape
    x = x2d.index_select(-1, perm.to(x2d.device))
    x = x.reshape(-1, n_blocks, B)
    x = torch.bmm(x.transpose(0, 1).contiguous().to(torch.float32),
                  blocks.to(torch.float32))
    x = x.transpose(0, 1).contiguous().reshape(-1, in_f)
    if scale is not None and bits and bits > 0:
        maxq = float((1 << (bits - 1)) - 1)
        s = scale.to(torch.float32).clamp_min(1e-8)
        x = torch.clamp(torch.round(x / s), -maxq - 1, maxq) * s
    return x.to(x2d.dtype)
