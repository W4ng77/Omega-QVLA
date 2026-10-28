"""Causal accuracy ablation: change ONLY released LLM A4 scale granularity.

Retain the released W/R,[-8,7] rounding, DiT8 scale rows and existing FP16-MMA
packed kernel. This is not a new original-method deployment or kernel search.
"""
import torch
import triton
from research.holoq_bench.released_grid import ReleasedGridLinear
from research.holoq_bench.packed_runtime import apply_blocks, _tile_fp16


class ReleasedTokenLinear(ReleasedGridLinear):
    def forward(self, x):
        self.calls += 1
        self.input_shapes.add(tuple(x.shape))
        assert not self.rot_out.numel() and self.backend == "tile_fp16"
        x = apply_blocks(x.half(), self.rot_in, self.perm)
        scale = (x.float().abs().amax(-1, keepdim=True).clamp_min(1e-5) / 7).half()
        xq = ((x / scale).round().clamp(-8, 7) * scale).half()
        shape = x.shape
        xx = xq.reshape(-1, self.in_features).contiguous()
        m, k, n = len(xx), self.in_features, self.out_features
        y = torch.empty((m, n), device=x.device, dtype=torch.float16)
        _tile_fp16[(triton.cdiv(m, 32), triton.cdiv(n, 64))](xx, self.B, self.s, y, m, n, k, self.group, 32, 64, 64, num_warps=4, num_stages=3)
        y = y.reshape(*shape[:-1], n)
        return y + self.bias if self.bias.numel() else y


def install(model):
    count = 0
    for name, module in model.named_modules():
        if "language_model" in name and isinstance(module, ReleasedGridLinear):
            assert module.act_mode == "token"
            module.__class__ = ReleasedTokenLinear
            count += 1
    assert count == 84
    print("EXPLICIT_RELEASE_TOKEN_SCALE_ACCURACY_ABLATION", count, flush=True)
