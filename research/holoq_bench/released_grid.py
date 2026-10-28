"""Compatibility deployment of the released FP16-rounded W4 weight values.

Recover a signed nibble plus FP32 group coefficient whose FP16 reconstruction
equals EVERY released weight value; never round/requantize those values again.
Use the existing tile_fp16 packed kernel. Activations follow legacy FP16
division/round/clamp/multiply, with frozen original runtime scales.
"""
import torch
import triton
from research.holoq_bench.packed_runtime import PackedLinear, apply_blocks, _tile_fp16
from gr00t.quantization.dit_step_context import get_current_dit_step


def half_neighbors(w):
    """Exact FP16 adjacent values; Torch2.5 lacks CUDA half nextafter."""
    assert w.dtype == torch.float16
    bits = w.contiguous().view(torch.int16).int()
    negative = w < 0
    previous = torch.where(negative, bits + 1, bits - 1)
    following = torch.where(negative, bits - 1, bits + 1)
    previous = torch.where(w == 0, -32767, previous)  #negative min subnormal
    following = torch.where(w == 0, 1, following)
    return previous.short().view(torch.float16), following.short().view(torch.float16)


@torch.no_grad()
def recover_exact_grid(weight, group=128):
    assert weight.ndim == 2 and weight.dtype == torch.float16 and weight.shape[1] % group == 0
    assert torch.isfinite(weight).all()
    n, k = weight.shape
    w = weight.reshape(n, k // group, group)
    wf = w.float()
    nonzero = w != 0
    smallest = torch.where(nonzero, wf.abs(), float("inf")).amin(-1)
    empty = ~nonzero.any(-1)
    smallest = torch.where(empty, 1, smallest)
    previous, following = half_neighbors(w)
    lower = (wf + previous.float()) * .5
    upper = (wf + following.float()) * .5
    qout = torch.zeros_like(w, dtype=torch.int8)
    sout = torch.ones_like(smallest)
    resolved = empty.clone()
    for divisor in range(1, 9):
        candidate = smallest / divisor
        q = (wf / candidate[..., None]).round()
        safe = torch.where(q != 0, q, 1)
        lo = torch.where(q > 0, lower / safe, upper / safe)
        hi = torch.where(q > 0, upper / safe, lower / safe)
        lo = torch.where(q != 0, lo, 0).amax(-1)
        hi = torch.where(q != 0, hi, float("inf")).amin(-1)
        scale = (lo + hi) * .5
        valid = (~resolved) & (lo <= hi) & (q.amin(-1) >= -8) & (q.amax(-1) <= 7)
        valid &= ((q * scale[..., None]).half() == w).all(-1)
        sout = torch.where(valid, scale, sout)
        qout = torch.where(valid[..., None], q.to(torch.int8), qout)
        resolved |= valid
        if bool(resolved.all()):
            break
    if not bool(resolved.all()):
        raise ValueError(f"Released tensor is not exactly representable as W4/G{group}; unresolved={int((~resolved).sum())}")
    assert torch.equal((qout.float() * sout[..., None]).half(), w)
    return qout.reshape(n, k), sout


class ReleasedGridLinear(PackedLinear):
    def forward(self, x):
        self.calls += 1
        self.input_shapes.add(tuple(x.shape))
        x = x.half()
        if self.rot_in.numel():
            x = apply_blocks(x, self.rot_in, self.perm)
        if self.act_mode == "step":
            step = get_current_dit_step()
            if step is None or not 0 <= step < 8:
                raise RuntimeError("Released grid requires the original8-step scale row")
            scale = self.a_scale[step].half()
        else:
            scale = self.a_scale.half()
        # Keep original half division rounding BEFORE nearest-integer rounding.
        # This intentionally differs from the new paper rebuild's F32 prepare.
        shape = x.shape
        if self.backend == "native_s4":
            # Converted variant consumes original rotated, unquantized input;
            # its existing native engine applies the same frozen scale table.
            y = self.integer(x.reshape(-1, self.in_features).contiguous(), step, engine="native_s4").half()
        else:
            xq = ((x / scale).round().clamp(-8, 7) * scale).half()
            xx = xq.reshape(-1, self.in_features).contiguous()
            m, k = xx.shape
            n = self.out_features
            y = torch.empty((m, n), device=x.device, dtype=torch.float16)
            _tile_fp16[(triton.cdiv(m, 32), triton.cdiv(n, 64))](xx, self.B, self.s, y, m, n, k, self.group, 32, 64, 64, num_warps=4, num_stages=3)
        y = y.reshape(*shape[:-1], self.out_features)
        if self.rot_out.numel():
            y = apply_blocks(y, self.rot_out)
        return y + self.bias if self.bias.numel() else y


def install(model):
    count = 0
    for module in model.modules():
        if not isinstance(module, PackedLinear):
            continue
        if module.backend not in ("tile_fp16", "native_s4"):
            raise ValueError("Explicit released-grid deployment requires packed tile/native records")
        module.__class__ = ReleasedGridLinear
        if not bool((module.a_scale.half() > 0).all()):
            raise ValueError("Released FP16 activation scale underflows; do not silently change it")
        count += 1
    assert count == 180
    print("EXPLICIT_RELEASED_GRID_PRESERVED", count, flush=True)
