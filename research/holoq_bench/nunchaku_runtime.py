"""Explicit Nunchaku G64 adapter, never selected by shape or silent fallback."""

from pathlib import Path
import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from research.holoq_bench.nunchaku_probe import extension
from research.holoq_bench.packed_runtime import PackedLinear, apply_blocks


@triton.jit
def _pack_input(X, SM, TABLE, PACKED, SCALES, M: tl.constexpr, K: tl.constexpr,
                STEP: tl.constexpr, SMOOTH: tl.constexpr, QMIN: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    if STEP:
        scalar = tl.full((), 1, tl.float32)
    else:
        peak = tl.full((), 1e-5, tl.float32)
        for nibble in range(8):
            col = j * 8 + nibble
            value = tl.load(X + row * K + col, (row < M) & (col < K), 0).to(tl.float32)
            if SMOOTH:
                value = tl.div_rn(value, tl.load(SM + col, col < K, 1).to(tl.float32))
            value = value.to(tl.float16).to(tl.float32)
            peak = tl.maximum(peak, tl.max(tl.abs(value)))
        scalar = (peak / 7).to(tl.float16).to(tl.float32)
    word = tl.full((B,), 0, tl.int32)
    for nibble in range(8):
        col = j * 8 + nibble
        value = tl.load(X + row * K + col, (row < M) & (col < K), 0).to(tl.float32)
        if SMOOTH:
            value = tl.div_rn(value, tl.load(SM + col, col < K, 1).to(tl.float32))
        value = value.to(tl.float16).to(tl.float32)
        if STEP:
            scale = tl.load(TABLE + col, col < K, 1).to(tl.float32)
        else:
            scale = scalar
        q = libdevice.nearbyint(tl.div_rn(value, scale))
        q = tl.minimum(tl.maximum(q, QMIN), 7).to(tl.int32)
        word = word | ((q & 15) << (nibble * 4))
    kg = j // 8
    warp = row % 256 // 32
    tile = row % 32 // 16
    lane = row % 8 * 4 + j % 4
    reg = row % 16 // 8 + (j % 8 // 4) * 2
    offset = ((((row // 256 * (K // 64) + kg) * 8 + warp) * 2 + tile) * 32 + lane) * 4 + reg
    tl.store(PACKED + offset, word, j < K // 8)
    groups = tl.arange(0, B // 8)
    slane = row % 32 // 16 * 8 + row % 8
    sel = row % 16 // 8
    so = ((row // 256 * (K // 64) + groups) * 8 + warp) * 32 + slane * 2 + sel
    tl.store(SCALES + so, scalar, groups < K // 64)


class NunchakuLinear(PackedLinear):
    def __init__(self, record, name, fused_rotation):
        nn.Module.__init__(self)
        assert record["backend"] == "nunchaku_s4" and record["act_bits"] == 4
        assert record["act_mode"] in ("token", "step")
        self.name, self.backend = name, "nunchaku_s4"
        self.in_features, self.out_features = record["K"], record["N"]
        self.act_bits, self.group = 4, 64
        self.act_mode, self.act_qmin = record["act_mode"], record["act_qmin"]
        self.fused_rotation = fused_rotation
        self.calls, self.input_shapes = 0, set()
        for key in ("B", "s", "a_scale", "smooth", "rot_in", "rot_out", "perm", "bias"):
            value = record.get(key)
            self.register_buffer(key, value if value is not None else torch.empty(0))
        if self.act_mode == "step":
            assert self.a_scale.shape == (8, self.in_features) and self.B.shape[0] == 9
        assert self.B.dtype == torch.int32 and self.s.dtype == torch.float16
        self._extension = extension()

    def forward(self, x):
        self.calls += 1
        self.input_shapes.add(tuple(x.shape))
        x = x.half()
        if self.rot_in.numel():
            if self.fused_rotation:
                from research.holoq_bench.fused_rotation import rotate

                x = rotate(x, self.rot_in, self.perm)
            else:
                x = apply_blocks(x, self.rot_in, self.perm)
        shape = x.shape
        xx = x.reshape(-1, self.in_features).contiguous()
        m, k = xx.shape
        if self.act_mode == "step":
            from gr00t.quantization.dit_step_context import get_current_dit_step

            step = get_current_dit_step()
            if step is None or not 0 <= step < 8:
                raise RuntimeError(f"Missing explicit step for {self.name}")
            table, weight, scales = self.a_scale[step], self.B[step], self.s[step]
        else:
            table, weight, scales = self.a_scale, self.B, self.s
        padded = triton.cdiv(m, 256) * 256
        aa = torch.empty((padded, k // 8), device=x.device, dtype=torch.int32)
        asc = torch.empty((padded * k // 64,), device=x.device, dtype=torch.float16)
        _pack_input[(padded,)](
            xx, self.smooth, table, aa, asc, m, k, self.act_mode == "step", bool(self.smooth.numel()),
            self.act_qmin, triton.next_power_of_2(k // 8), num_warps=4 if k < 4096 else 8,
        )
        out = torch.empty((m, self.out_features), device=x.device, dtype=torch.float16)
        self._extension.gemm(aa, weight, asc, scales, out)
        y = out.reshape(*shape[:-1], self.out_features)
        if self.rot_out.numel():
            if self.fused_rotation:
                from research.holoq_bench.fused_rotation import rotate

                y = rotate(y, self.rot_out)
            else:
                y = apply_blocks(y, self.rot_out)
        return y + self.bias if self.bias.numel() else y


def enable(model, path):
    """Separate explicit factory leaves all frozen previous runtime bytes intact."""
    pack = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    assert pack["format"] == "holoq_bench_packed_v1" and pack["meta"]["runtime_plugin"] == "nunchaku_s4"
    fused = pack["meta"].get("rotation_plugin") == "nunchaku_and_triton_block64"
    names = dict(model.named_modules())
    assert not pack.get("nonquantized_overrides") and not pack.get("atm"), "These exports only support HoloQ/SQ"
    for name, record in pack["layers"].items():
        old = names.get(name)
        if not isinstance(old, nn.Linear) or old.in_features != record["K"] or old.out_features != record["N"]:
            raise ValueError(("Nunchaku replacement mismatch", name))
        if record["backend"] == "nunchaku_s4":
            replacement = NunchakuLinear(record, name, fused)
        else:
            replacement = PackedLinear(record, name)
            if fused and replacement.rot_in.numel():
                from research.holoq_bench.fused_rotation import FusedPackedLinear

                replacement.__class__ = FusedPackedLinear
                for src, dest in [("rot_in", "_fused_in"), ("rot_out", "_fused_out"), ("bias", "_fused_bias")]:
                    replacement.register_buffer(dest, getattr(replacement, src))
                    setattr(replacement, src, torch.empty(0))
        parent_name, attr = name.rsplit(".", 1)
        setattr(names[parent_name], attr, replacement)
    assert len(pack["layers"]) == pack["meta"]["expected_layers"] == 180
    print("EXPLICIT_NUNCHAKU_PACKED", Path(path).resolve(), "fused", fused, flush=True)
    return pack["meta"]
