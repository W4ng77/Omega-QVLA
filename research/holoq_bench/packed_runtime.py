"""Explicit packed backends; no dense-weight inference cache or F.linear path."""

from pathlib import Path

import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from research.holoq_bench import kernels


@triton.jit
def _prepare(
    X,
    SM,
    AS,
    OUT,
    SC,
    K: tl.constexpr,
    MODE: tl.constexpr,
    BITS: tl.constexpr,
    SMOOTH: tl.constexpr,
    INT8: tl.constexpr,
    QMIN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    kk = tl.arange(0, BK)
    x = tl.load(X + row * K + kk, kk < K, 0).to(tl.float32)
    if SMOOTH:
        x = tl.div_rn(x, tl.load(SM + kk, kk < K, 1))
    x = x.to(tl.float16).to(tl.float32)
    if BITS < 16:
        if MODE == 1:
            a = (tl.maximum(tl.max(tl.abs(x)), 1e-5) / ((1 << (BITS - 1)) - 1)).to(tl.float16).to(tl.float32)
        else:
            a = tl.load(AS + kk, kk < K, 1).to(tl.float32)
        q = libdevice.nearbyint(tl.div_rn(x, a))
        q = tl.minimum(tl.maximum(q, QMIN), (1 << (BITS - 1)) - 1)
        if INT8:
            y = q.to(tl.int8)
            tl.store(SC + row, a)
        else:
            y = q * a
    else:
        y = x
    tl.store(OUT + row * K + kk, y, kk < K)


@triton.jit
def _tile_fp16(
    A,
    P,
    S,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    mm = tl.program_id(0) * BM + tl.arange(0, BM)
    nn = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        aa = tl.load(A + mm[:, None] * K + k[None, :], (mm[:, None] < M) & (k[None, :] < K), 0)
        p = tl.load(P + nn[:, None] * tl.cdiv(K, 2) + k[None, :] // 2, (nn[:, None] < N) & (k[None, :] < K), 0).to(
            tl.int32
        )
        q = (p >> ((k[None, :] % 2) * 4)) & 15
        q = tl.where(q >= 8, q - 16, q).to(tl.float32)
        s = tl.load(S + nn[:, None] * tl.cdiv(K, G) + k[None, :] // G, (nn[:, None] < N) & (k[None, :] < K), 0)
        ww = (q * s).to(tl.float16)
        acc = tl.dot(aa, tl.trans(ww), acc)
    tl.store(OUT + mm[:, None] * N + nn[None, :], acc, (mm[:, None] < M) & (nn[None, :] < N))


def apply_blocks(x, blocks, perm=None):
    shape = x.shape
    xx = x.reshape(-1, shape[-1])
    if perm is not None and perm.numel():
        xx = xx.index_select(1, perm)
    nb, b, _ = blocks.shape
    y = torch.bmm(xx.reshape(-1, nb, b).transpose(0, 1).contiguous(), blocks)
    return y.transpose(0, 1).contiguous().reshape(shape)


class PackedLinear(nn.Module):
    def __init__(self, record, name):
        super().__init__()
        self.name = name
        self.in_features, self.out_features = int(record["K"]), int(record["N"])
        self.backend = record["backend"]
        self.act_bits = int(record["act_bits"])
        self.act_mode = record["act_mode"]
        self.act_qmin = int(record.get("act_qmin", -(2 ** (self.act_bits - 1))))
        self.group = int(record.get("group", self.in_features))
        self.calls = 0
        for key in ("B", "s", "z", "s2", "z2", "smooth", "a_scale", "rot_in", "rot_out", "perm", "bias"):
            value = record.get(key)
            self.register_buffer(key, value if value is not None else torch.empty(0), persistent=True)
        if self.backend == "marlin":
            self._marlin = kernels.marlin()
            self.register_buffer(
                "workspace", torch.zeros(self.out_features // 128 * 16, dtype=torch.int32), persistent=False
            )
        elif self.backend == "awq":
            self._awq = kernels.awq()
        elif self.backend == "qserve":
            if self.act_mode != "token" or self.act_bits > 8:
                raise ValueError("QServe requires matched per-token integer activation scales")
            self._qserve = kernels.qserve()
        elif self.backend == "native_s4":
            from gr00t.quantization.perstep_backends import PerStepBackend

            self.integer = PerStepBackend(record["integer"])
        elif self.backend != "tile_fp16":
            raise ValueError(f"Unsupported packed backend {self.backend}")

    def forward(self, x):
        self.calls += 1
        # Adaptive layer normalization can emit FP32 even in an FP16 policy.
        # Match an autocast FP16 Linear's operand conversion explicitly.
        if x.dtype != torch.float16:
            x = x.half()
        if self.rot_in.numel():
            x = apply_blocks(x, self.rot_in, self.perm)
        shape = x.shape
        xx = x.reshape(-1, self.in_features).contiguous()
        m, k = xx.shape
        n = self.out_features
        if self.act_mode == "step":
            from gr00t.quantization.dit_step_context import get_current_dit_step

            step = get_current_dit_step()
            if step is None or not 0 <= step < 8:
                raise RuntimeError(f"Missing denoising step for {self.name}")
            a = self.a_scale[step]
        else:
            a = self.a_scale
        if self.backend == "native_s4":
            if self.smooth.numel():
                xx = xx / self.smooth
            y = self.integer(xx, step, engine="native_s4").half()
        else:
            int8 = self.backend == "qserve"
            if self.act_bits == 16 and not self.smooth.numel():
                aa = xx
                asc = torch.empty(0, device=x.device, dtype=torch.float16)
            else:
                aa = torch.empty_like(xx, dtype=torch.int8 if int8 else torch.float16)
                asc = torch.empty(m, device=x.device, dtype=torch.float16)
                _prepare[(m,)](
                    xx,
                    self.smooth,
                    a,
                    aa,
                    asc,
                    k,
                    1 if self.act_mode == "token" else 2,
                    self.act_bits,
                    bool(self.smooth.numel()),
                    int8,
                    self.act_qmin,
                    triton.next_power_of_2(k),
                    num_warps=4 if k < 4096 else 8,
                )
            y = torch.empty((m, n), device=x.device, dtype=torch.float16)
            if self.backend == "marlin":
                self._marlin.mul(aa, self.B, y, self.s, self.workspace)
            elif self.backend == "awq":
                if m < 8:
                    y = self._awq.gemv_forward_cuda_new(aa, self.B, self.s, self.z, m, n, k, self.group)
                else:
                    y = self._awq.gemm_forward_cuda_new(aa, self.B, self.s, self.z)
            elif self.backend == "qserve":
                self._qserve.gemm_forward_cuda(aa, self.B, self.z2, self.s2, self.s, asc, y)
            else:
                _tile_fp16[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
                    aa, self.B, self.s, y, m, n, k, self.group, 32, 64, 64, num_warps=4, num_stages=3
                )
        y = y.reshape(*shape[:-1], n)
        if self.rot_out.numel():
            y = apply_blocks(y, self.rot_out)
        if self.bias.numel():
            y = y + self.bias
        return y


def enable(model, path):
    pack = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if pack.get("format") != "holoq_bench_packed_v1":
        raise ValueError("Not an explicit benchmark packed export")
    names = dict(model.named_modules())
    for name, record in pack["layers"].items():
        old = names.get(name)
        if not isinstance(old, nn.Linear) or old.in_features != record["K"] or old.out_features != record["N"]:
            raise ValueError(f"Packed replacement mismatch: {name}")
        parent_name, attr = name.rsplit(".", 1)
        setattr(names[parent_name], attr, PackedLinear(record, name))
    for name, entry in pack.get("atm", {}).items():
        attention = names[name]
        alpha = torch.tensor(entry["all"], dtype=torch.float16)
        attention.register_buffer("_atm_alpha_all", alpha, persistent=True)
        attention._ohb_beta_scalar = float(entry["beta"])
    expected = int(pack["meta"]["expected_layers"])
    if len(pack["layers"]) != expected:
        raise RuntimeError("Quantized layer-count mismatch")
    print(
        f"[HOLOQ_PACKED] method={pack['meta']['method']} layers={expected} backends="
        f"{sorted({r['backend'] for r in pack['layers'].values()})} source={Path(path).resolve()}",
        flush=True,
    )
    return pack["meta"]
