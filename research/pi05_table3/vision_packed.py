"""Execute the existing 162 Table3 vision RTN records without dense W caches.

Retains the record's asymmetric W4 grid and per-token asymmetric A4 arithmetic.
A and W are nibble packed; tiles decode to BF16/FP16 for MMA, FP32 accumulation.
This is not native INT4 MMA and is not a new quantization algorithm.
"""
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _vision_gemm(A, SA, ZA, W, SW, ZW, BIAS, Y,
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
                 W_STORED_BF16: tl.constexpr, W_STORED_FP16: tl.constexpr,
                 INPUT_BF16: tl.constexpr, HAS_BIAS: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    sa = tl.load(SA + m, m < M, 1)
    za = tl.load(ZA + m, m < M, 0).to(tl.float32)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        pa = tl.load(A + m[:, None] * (K // 2) + k[None, :] // 2,
                     (m[:, None] < M) & (k[None, :] < K), 0).to(tl.int32)
        qa = (pa >> ((k[None, :] % 2) * 4)) & 15
        av = (qa.to(tl.float32) - za[:, None]) * sa[:, None]
        pw = tl.load(W + n[:, None] * (K // 2) + k[None, :] // 2,
                     (n[:, None] < N) & (k[None, :] < K), 0).to(tl.int32)
        qw = (pw >> ((k[None, :] % 2) * 4)) & 15
        group = n[:, None] * (K // G) + k[None, :] // G
        mask = (n[:, None] < N) & (k[None, :] < K)
        sw = tl.load(SW + group, mask, 0)
        zw = tl.load(ZW + group, mask, 0).to(tl.float32)
        wv = (qw.to(tl.float32) - zw) * sw
        # Preserve offline stored dtype before OpenPI's model dtype cast.
        if W_STORED_BF16:
            wv = wv.to(tl.bfloat16).to(tl.float32)
        elif W_STORED_FP16:
            wv = wv.to(tl.float16).to(tl.float32)
        if INPUT_BF16:
            acc = tl.dot(av.to(tl.bfloat16), tl.trans(wv.to(tl.bfloat16)), acc)
        else:
            acc = tl.dot(av.to(tl.float16), tl.trans(wv.to(tl.float16)), acc)
    if HAS_BIAS:
        acc += tl.load(BIAS + n, n < N, 0).to(tl.float32)[None, :]
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


class PackedVision(nn.Module):
    def __init__(self, source, record, name):
        super().__init__()
        assert record["encoding"] == "int4_rtn_v1"
        self.name = name
        self.out_features, self.in_features = record["shape"]
        self.group_size = record["group_size"]
        self.stored_dtype = record["dtype"]
        self.compute_dtype = source.weight.dtype
        assert self.compute_dtype in (torch.bfloat16, torch.float16)
        device = source.weight.device
        # SigLIP reads q_proj.weight.dtype to cast its input. Expose dtype
        # without allocating or reconstructing a full floating weight tensor.
        self.register_buffer("_weight_dtype_marker", torch.empty(0, dtype=self.compute_dtype, device=device))
        self.register_buffer("B", record["packed"].reshape(self.out_features, self.in_features // 2).to(device).contiguous())
        self.register_buffer("sw", record["scales"].to(device).float().contiguous())
        self.register_buffer("zw", record["zero_points"].to(device).contiguous())
        self.register_buffer("bias", source.bias.detach().to(device) if source.bias is not None else torch.empty(0, device=device))
        self.calls = 0
        self.shapes = set()

    @property
    def weight(self):
        return self._weight_dtype_marker

    def forward(self, x):
        self.calls += 1
        self.shapes.add(tuple(x.shape))
        assert x.dtype == self.compute_dtype, (self.name, x.dtype, self.compute_dtype)
        shape = x.shape
        xx = x.reshape(-1, self.in_features).float()
        minimum, maximum = xx.amin(-1, keepdim=True), xx.amax(-1, keepdim=True)
        sa = (maximum - minimum).clamp_min(1e-8) / 15
        za = (-minimum / sa).round().clamp(0, 15)
        q = (xx / sa + za).round().clamp(0, 15).to(torch.uint8)
        aa = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
        m = len(xx)
        y = torch.empty((m, self.out_features), device=x.device, dtype=self.compute_dtype)
        _vision_gemm[(triton.cdiv(m, 32), triton.cdiv(self.out_features, 64))](
            aa, sa, za, self.B, self.sw, self.zw, self.bias, y,
            m, self.out_features, self.in_features, self.group_size,
            self.stored_dtype == "torch.bfloat16", self.stored_dtype == "torch.float16",
            self.compute_dtype == torch.bfloat16, bool(self.bias.numel()),
            32, 64, 64, num_warps=4, num_stages=2)
        return y.reshape(*shape[:-1], self.out_features)
