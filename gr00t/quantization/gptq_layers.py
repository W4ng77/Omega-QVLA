"""GPTQ runtime wrapper and solver.

Loads offline-built GPTQ packs at inference time and replaces matching
``nn.Linear`` layers with :class:`GptqLinear`. Supports:

  * the GPTQ-quantized dense weight (``baseline_q`` / legacy ``weight_q``),
  * the SVDQuant residual format (``weight_res_q`` + low-rank ``A``/``B``),
  * per-DiT-step activation scale tables (``act_scale_table``),
  * DuQuant input/output rotations (dense or block fast-path),
  * optional SmoothQuant per-channel scaling.

``gptq_quantize_weight`` is the blockwise GPTQ solver used by the offline pack
builders in ``tools/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from .duquant_layers import _parse_per_layer_wbits, select_targets
from .duquant_preprocess import (
    PercentileCalibrator,
    compute_mse_scales,
    fake_quantize_sym,
    qmax,
    sanitize_name,
)


_GPTQ_CACHE: Dict[str, Any] = {}


@dataclass
class GptqConfig:
    enabled: Optional[bool] = None
    path: Optional[str] = None
    act_bits: Optional[int] = None
    act_percentile: Optional[float] = None
    calib_batches: Optional[int] = None
    weight_bits: Optional[int] = None
    missing: Optional[str] = None

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = os.environ.get("GR00T_GPTQ", "0") not in ("0", "false", "False")
        if self.path is None:
            self.path = os.environ.get("GR00T_GPTQ_PATH")
        if self.act_bits is None:
            self.act_bits = int(os.environ.get("GR00T_GPTQ_ABITS", 8))
        if self.act_percentile is None:
            self.act_percentile = float(os.environ.get("GR00T_GPTQ_ACT_PCT", 99.9))
        if self.calib_batches is None:
            self.calib_batches = int(os.environ.get("GR00T_GPTQ_CALIB_STEPS", 32))
        if self.weight_bits is None:
            self.weight_bits = int(os.environ.get("GR00T_GPTQ_WBITS_DEFAULT", 4))
        if self.missing is None:
            self.missing = os.environ.get("GR00T_GPTQ_MISSING", "error").lower()
        if self.enabled and not self.path:
            raise ValueError("GR00T_GPTQ=1 requires GR00T_GPTQ_PATH")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_quantized_container(path: Path) -> Any:
    cache_key = str(path.resolve())
    if cache_key in _GPTQ_CACHE:
        return _GPTQ_CACHE[cache_key]
    if path.suffix not in (".pt", ".pth"):
        raise ValueError(f"Unsupported GPTQ weight file: {path}")
    value = _torch_load(path)
    _GPTQ_CACHE[cache_key] = value
    return value


def _record_get(record: Any, keys: Tuple[str, ...]) -> Optional[Any]:
    if isinstance(record, dict):
        for key in keys:
            if key in record:
                return record[key]
    return None


def _load_quantized_record(layer_name: str, weights_path: str) -> Optional[Any]:
    path = Path(weights_path)
    if path.is_dir():
        safe = sanitize_name(layer_name)
        for suffix in (".pt", ".pth"):
            candidate = path / f"{safe}{suffix}"
            if candidate.exists():
                return _load_quantized_container(candidate)
        return None

    container = _load_quantized_container(path)
    if not isinstance(container, dict):
        return container
    if any(key in container for key in ("baseline_q", "weight_q", "W_q", "quant_weight")):
        return container
    for key in (layer_name, sanitize_name(layer_name)):
        if key in container:
            return container[key]
    return None


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value)


def _quantize_vector_sym(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    max_q = qmax(bits)
    return torch.clamp(torch.round(x / scale), -max_q - 1, max_q) * scale


def _prepare_hessian(
    H: torch.Tensor,
    damp_percent: float,
    reg_lambda: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize H, mark dead diag, add damping + absolute regularization.

    Two diagonal additions:
      - damp = damp_percent * mean(diag(H))   (relative; classic GPTQ)
      - reg_lambda                            (absolute; adds λI for L2 reg on
                                               (W-Q), i.e. solves min (W-Q)^T (H + λI)(W-Q))
    """
    H = 0.5 * (H + H.t())
    diag = torch.diag(H).clone()
    dead = diag <= 0
    if dead.any():
        H = H.clone()
        H[dead, dead] = 1.0
    mean_diag = float(diag[~dead].mean().item()) if (~dead).any() else 1.0
    damp = max(float(damp_percent), 0.0) * max(mean_diag, 1e-8)
    total_diag_add = damp + max(float(reg_lambda), 0.0)
    if total_diag_add > 0:
        idx = torch.arange(H.shape[0], device=H.device)
        H = H.clone()
        H[idx, idx] += total_diag_add
    return H, dead


def gptq_quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: int,
    block_size: int = 128,
    damp_percent: float = 0.01,
    mse_alphas: Optional[Sequence[float]] = None,
    reg_lambda: float = 0.0,
    err_comp_gamma: float = 1.0,
    group_size: int = 0,
) -> torch.Tensor:
    """A practical GPTQ-style solver on the input Gram matrix H.

    We quantize each input block independently. This is exact when block_size
    covers the full input dimension and otherwise acts as a standard blockwise
    GPTQ approximation.

    ``mse_alphas`` overrides the per-row MSE candidate grid. Wider grids (e.g.
    up to 5.0) let rows with heavy outliers automatically choose aggressive
    clipping when it strictly reduces row-MSE — this is the principled
    AWQ-style replacement for ad-hoc per-row scale factors. Default falls back
    to ``compute_mse_scales``' built-in narrow grid.

    ``group_size`` (P1.3): if > 0 and < in_features, scales are recomputed every
    ``group_size`` input columns instead of once per block. The output stays a
    dense FP weight (the per-group scale information is "baked in" via the
    rounding), so the runtime needs no change.
    """
    if bits <= 0:
        return W.detach().clone()

    W = W.detach().to(dtype=torch.float32)
    h_dtype = torch.float32 if W.is_cuda else torch.float64
    H = H.detach().to(device=W.device, dtype=h_dtype)
    out_features, in_features = W.shape
    if H.shape != (in_features, in_features):
        raise ValueError(f"H has shape {tuple(H.shape)}, expected {(in_features, in_features)}")

    if block_size <= 0 or block_size >= in_features:
        block_size = in_features

    use_groups = 0 < int(group_size) < in_features
    group_scales: Dict[int, torch.Tensor] = {}
    if use_groups:
        # Precompute per-group per-row scales from the original W. Using the
        # original (not partially-quantized) W matches AWQ/GPTQ-grouped
        # convention and keeps the scale grid stable.
        for g_start in range(0, in_features, int(group_size)):
            g_end = min(g_start + int(group_size), in_features)
            group_scales[g_start] = compute_mse_scales(
                W[:, g_start:g_end], bits, alphas=mse_alphas
            ).to(dtype=W.dtype, device=W.device)

    Q = W.clone()
    gamma = float(err_comp_gamma)
    for start in range(0, in_features, block_size):
        end = min(start + block_size, in_features)
        W_block = W[:, start:end].clone()
        H_block = H[start:end, start:end].clone()
        H_block, dead = _prepare_hessian(H_block, damp_percent, reg_lambda=reg_lambda)
        if dead.any():
            W_block[:, dead] = 0

        try:
            chol = torch.linalg.cholesky(H_block)
            Hinv = torch.cholesky_inverse(chol).to(dtype=torch.float32)
        except RuntimeError:
            Hinv = torch.linalg.pinv(H_block).to(dtype=torch.float32)

        block_q = torch.zeros_like(W_block)
        if not use_groups:
            row_scales = compute_mse_scales(W_block, bits, alphas=mse_alphas).to(
                dtype=W_block.dtype, device=W_block.device
            )

        for col in range(W_block.shape[1]):
            denom = float(Hinv[col, col].item())
            if not torch.isfinite(Hinv[col, col]) or abs(denom) < 1e-12:
                denom = 1.0
            w_col = W_block[:, col]
            if use_groups:
                global_col = start + col
                g_start = (global_col // int(group_size)) * int(group_size)
                row_scales_g = group_scales[g_start]
            else:
                row_scales_g = row_scales
            q_col = _quantize_vector_sym(w_col, row_scales_g, bits)
            block_q[:, col] = q_col
            err = (w_col - q_col) / denom
            # err_comp_gamma scales the off-diagonal propagation; gamma=0
            # disables error compensation (equivalent to per-block-MSE RTN).
            if gamma > 0 and col + 1 < W_block.shape[1]:
                W_block[:, col + 1:] -= gamma * err[:, None] * Hinv[col, col + 1:][None, :]

        Q[:, start:end] = block_q
    return Q


class GptqLinear(nn.Module):
    """Runtime wrapper for offline-built GPTQ quantized weights."""

    def __init__(self, base: nn.Linear, name: str, cfg: GptqConfig, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.cfg = cfg
        self.weight_bits = int(weight_bits if weight_bits is not None else (cfg.weight_bits or 0))
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        # Drop the duplicate FP weight buffer when quantization is available — only
        # allocate it as a fallback if the record is missing. Set
        # GR00T_GPTQ_KEEP_FP=1 to opt back in.
        keep_fp_env = os.environ.get("GR00T_GPTQ_KEEP_FP", "0") not in ("0", "false", "False")
        self._debug_enabled = os.environ.get("GR00T_GPTQ_DEBUG", "0") not in ("0", "false", "False")

        self._act_scale: Optional[torch.Tensor] = None
        self._act_scale_initialized = False
        self.calibrator = PercentileCalibrator(
            percentile=float(cfg.act_percentile or 99.9),
            max_batches=int(cfg.calib_batches or 32),
        )

        record = _load_quantized_record(name, cfg.path or "")
        # SmoothQuant per-input-channel rescale (loaded from pack if present).
        # When set, runtime applies x = x / smooth_scale before A8 fake_quant;
        # the stored W is already W·diag(s).
        self._has_smooth_scale = False
        # Per-DiT-step activation scale table (loaded from pack if present).
        # Shape (num_steps, in_features) — looked up by current_dit_step at
        # forward time. When unset, falls back to single ``_act_scale`` (the
        # legacy step-agnostic A8 calibration).
        self._has_act_scale_table = False
        # SVDQuant low-rank residual branch (loaded from pack if record format
        # == "dit_svdquant_v1"). When set, forward computes
        #   y = linear(x_q, weight_res_q) + (x_s @ B) @ A.T + bias
        # where x_s = x / smooth_scale and x_q = fake_quant(x_s, table[t]).
        self._has_svdquant = False
        # Per-record a_bits override (set by SVDQuant records). Allows mixed
        # precision: LLM records use cfg.act_bits, DiT SVDQuant records use
        # the value baked into the pack at build time.
        self._record_a_bits: Optional[int] = None
        # Init optional-feature flags up-front so fallback path is forward-safe.
        self._has_duquant_rotation = False
        self._has_duquant_rotation_out = False
        # Block-fast path for input/output rotation: when a pack stores the
        # block-diagonal structure explicitly (R_in_blocks: [n_blocks, B, B],
        # perm: [in_features]) alongside the dense R, runtime uses bmm over
        # blocks instead of a full in×in matmul. Critical for pi0.5
        # paligemma down_proj where in=16384 dense matmul = ~1GB/layer.
        self._has_duquant_rotation_blocks = False
        self._has_duquant_rotation_out_blocks = False
        # Fused Triton rot+act-quant path (GR00T_GPTQ_FUSED=1) + promoted-weight cache.
        self._fused_enabled = os.environ.get("GR00T_GPTQ_FUSED", "0") not in ("0", "false", "False")
        self._fused_min_tokens = int(os.environ.get("GR00T_GPTQ_FUSED_MIN_TOKENS", "64"))
        self._w_promoted_cache: Optional[torch.Tensor] = None
        if record is None:
            if cfg.missing == "error":
                raise FileNotFoundError(f"No GPTQ record found for layer '{name}' in {cfg.path}")
            # No quant available — keep FP weight as the live tensor.
            self.register_buffer("_weight_fp", base.weight.detach().clone(), persistent=False)
            self.register_buffer("_weight_q", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            self._quant_available = False
            self.register_buffer("_smooth_scale", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            self.register_buffer("_act_scale_table", torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            self._has_smooth_scale = False
            self._has_act_scale_table = False
            self._has_svdquant = False
        else:
            # Build the quantized dense weight; only keep _weight_q.
            weight_q_t = self._build_weight_from_record(record, name, base.weight)
            self.register_buffer("_weight_q", weight_q_t, persistent=False)
            if keep_fp_env:
                self.register_buffer("_weight_fp", base.weight.detach().clone(), persistent=False)
            else:
                # FP fallback not needed; allocate a zero-element placeholder so
                # attribute access remains valid for the `weight` property.
                self.register_buffer("_weight_fp", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            self._quant_available = True
            # Optional SmoothQuant scale: load if present in the record.
            ss_raw = _record_get(record, ("smooth_scale",))
            if ss_raw is not None:
                ss_t = _to_tensor(ss_raw).to(dtype=torch.float32)
                if ss_t.numel() == self.in_features:
                    self.register_buffer("_smooth_scale", ss_t, persistent=False)
                    self._has_smooth_scale = True
                else:
                    self.register_buffer("_smooth_scale", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            else:
                self.register_buffer("_smooth_scale", torch.empty(0, dtype=base.weight.dtype, device=base.weight.device), persistent=False)
            # Optional per-DiT-step activation scale table (only present for
            # DiT layers; LLM layers leave this unset and use _act_scale).
            ast_raw = _record_get(record, ("act_scale_table",))
            if ast_raw is not None:
                ast_t = _to_tensor(ast_raw).to(dtype=torch.float32)
                # Guard against FP16-underflow zeros in act_scale_table:
                # build computes scale = q99.9(|x|) / qmax(a_bits); for channels
                # with very low activation magnitude (q99.9 < ~1e-6) the result
                # is ~8e-9 which underflows FP16's subnormal range (~6e-8) when
                # save_dtype=float16 was used → stored as exact zero → runtime
                # fake_quantize_sym divides by zero → NaN cascade. Clamp on
                # load so existing packs are usable without rebuild.
                ast_t = ast_t.clamp_min(1e-8)
                if ast_t.dim() == 2 and ast_t.shape[1] == self.in_features:
                    self.register_buffer("_act_scale_table", ast_t, persistent=False)
                    # Pre-compute the step-mean fallback buffer at init time so
                    # the forward() fallback path doesn't allocate a fresh
                    # tensor inside a torch.compile + CUDAGraph capture (which
                    # would cause "accessing tensor output of CUDAGraphs that
                    # has been overwritten" RuntimeErrors on pi0.5 inference).
                    self.register_buffer(
                        "_act_scale_table_mean_buf",
                        ast_t.mean(dim=0).contiguous().clone(),
                        persistent=False,
                    )
                    self._has_act_scale_table = True
                else:
                    self.register_buffer(
                        "_act_scale_table",
                        torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device),
                        persistent=False,
                    )
            else:
                self.register_buffer(
                    "_act_scale_table",
                    torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device),
                    persistent=False,
                )
            # SVDQuant lowrank branch — only present for "dit_svdquant_v1"
            # records. lowrank_A is [out, r], lowrank_B is [in, r].
            fmt = _record_get(record, ("format",))
            fmt_str = str(fmt) if fmt is not None else ""
            if fmt_str == "dit_svdquant_v1":
                A_raw = _record_get(record, ("lowrank_A",))
                B_raw = _record_get(record, ("lowrank_B",))
                if A_raw is not None and B_raw is not None:
                    A_t = _to_tensor(A_raw).to(dtype=torch.float32)
                    B_t = _to_tensor(B_raw).to(dtype=torch.float32)
                    if (
                        A_t.dim() == 2
                        and B_t.dim() == 2
                        and A_t.shape[0] == self.out_features
                        and B_t.shape[0] == self.in_features
                        and A_t.shape[1] == B_t.shape[1]
                    ):
                        self.register_buffer("_lowrank_A", A_t, persistent=False)
                        self.register_buffer("_lowrank_B", B_t, persistent=False)
                        self._has_svdquant = True
                # Pull baked-in a_bits so DiT SVDQuant records use their own
                # bit-width regardless of cfg.act_bits (which may be 8 for LLM).
                ab_raw = _record_get(record, ("a_bits",))
                if ab_raw is not None:
                    try:
                        self._record_a_bits = int(ab_raw)
                    except Exception:
                        pass
                    else:
                        print(
                            f"[GPTQ][WARN] {name}: dit_svdquant_v1 record has "
                            f"bad lowrank shapes A={tuple(A_t.shape)} B={tuple(B_t.shape)} "
                            f"expected ({self.out_features}, r) and ({self.in_features}, r)",
                            flush=True,
                        )
            if not self._has_svdquant:
                self.register_buffer(
                    "_lowrank_A",
                    torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device),
                    persistent=False,
                )
                self.register_buffer(
                    "_lowrank_B",
                    torch.empty(0, 0, dtype=base.weight.dtype, device=base.weight.device),
                    persistent=False,
                )
            # Optional DuQuant input rotation: applied as x' = x @ R BEFORE
            # the standard GPTQ forward path. Used by the DuQuant+GPTQ hybrid
            # pack format built with --duquant-rotation.
            #
            # Fast path: when the pack stores `duquant_rotation_blocks` ([n,B,B])
            # + `duquant_rotation_perm` ([in_features] int), runtime uses bmm
            # over blocks: y = x[:, perm].view(T, n, B) @ R_b. This is
            # ~in/B × cheaper than the dense in×in matmul. For pi0.5
            # paligemma down_proj (in=16384, B=64): 256× cheaper.
            blocks_raw = _record_get(record, ("duquant_rotation_blocks",))
            perm_raw = _record_get(record, ("duquant_rotation_perm",))
            if blocks_raw is not None and perm_raw is not None:
                blocks_t = _to_tensor(blocks_raw).to(dtype=torch.float32)
                perm_t = _to_tensor(perm_raw).to(dtype=torch.int64)
                if (
                    blocks_t.dim() == 3
                    and blocks_t.shape[1] == blocks_t.shape[2]
                    and blocks_t.shape[0] * blocks_t.shape[1] == self.in_features
                    and perm_t.dim() == 1
                    and perm_t.numel() == self.in_features
                ):
                    self.register_buffer("_duquant_rotation_blocks", blocks_t, persistent=False)
                    self.register_buffer("_duquant_rotation_perm", perm_t, persistent=False)
                    self._has_duquant_rotation_blocks = True
            # Dense fallback (legacy packs): full in×in R. Slow on large in.
            # DuQuant rotations are (perm ∘ block-diag) by construction, so
            # first try to decompose the dense matrix and take the 32x-cheaper
            # block bmm fast path at runtime.
            rot_raw = _record_get(record, ("duquant_rotation",))
            if rot_raw is not None and not self._has_duquant_rotation_blocks:
                rot_t = _to_tensor(rot_raw).to(dtype=torch.float32)
                if rot_t.dim() == 2 and rot_t.shape[0] == self.in_features and rot_t.shape[1] == self.in_features:
                    if os.environ.get("GR00T_GPTQ_DENSE_ROT", "0") in ("0", "false", "False"):
                        from gr00t.quantization.fused_rot_quant import try_decompose_rotation

                        perm_d, blocks_d = try_decompose_rotation(rot_t, block=64)
                    else:
                        perm_d, blocks_d = None, None
                    if blocks_d is not None:
                        self.register_buffer("_duquant_rotation_blocks", blocks_d, persistent=False)
                        self.register_buffer("_duquant_rotation_perm", perm_d, persistent=False)
                        self._has_duquant_rotation_blocks = True
                    else:
                        self.register_buffer("_duquant_rotation", rot_t, persistent=False)
                        self._has_duquant_rotation = True
            # Optional DuQuant output rotation: applied as y_restored = y @ R_out
            # AFTER matmul, BEFORE bias.
            #
            # Fast path: `duquant_rotation_out_blocks` ([n_out, B_out, B_out])
            # — output side has no permutation in our builders, so just bmm.
            blocks_out_raw = _record_get(record, ("duquant_rotation_out_blocks",))
            if blocks_out_raw is not None:
                blocks_out_t = _to_tensor(blocks_out_raw).to(dtype=torch.float32)
                if (
                    blocks_out_t.dim() == 3
                    and blocks_out_t.shape[1] == blocks_out_t.shape[2]
                    and blocks_out_t.shape[0] * blocks_out_t.shape[1] == self.out_features
                ):
                    self.register_buffer("_duquant_rotation_out_blocks", blocks_out_t, persistent=False)
                    self._has_duquant_rotation_out_blocks = True
            # Dense fallback (legacy packs)
            rot_out_raw = _record_get(record, ("duquant_rotation_out",))
            if rot_out_raw is not None and not self._has_duquant_rotation_out_blocks:
                rot_out_t = _to_tensor(rot_out_raw).to(dtype=torch.float32)
                if rot_out_t.dim() == 2 and rot_out_t.shape[0] == self.out_features and rot_out_t.shape[1] == self.out_features:
                    self.register_buffer("_duquant_rotation_out", rot_out_t, persistent=False)
                    self._has_duquant_rotation_out = True
            # Free the now-unneeded base.weight reference; the parent module no
            # longer owns it after Module.replace.
            del weight_q_t

    @staticmethod
    def _build_weight_from_record(
        record: Any, name: str, ref_weight: torch.Tensor
    ) -> torch.Tensor:
        """Materialize the dense quantized weight from an offline record.

        Supports the SVDQuant residual format (weight_res_q + lowrank_A/B),
        the GPTQ baseline format (baseline_q), and the legacy fused format
        (weight_q).
        """
        # SVDQuant residual format: dense W_res_q stored at "weight_res_q".
        # The lowrank branch (A, B) is loaded as separate buffers in __init__.
        weight_res_q = _record_get(record, ("weight_res_q",))
        if weight_res_q is not None:
            wrq_t = _to_tensor(weight_res_q).to(dtype=ref_weight.dtype)
            if tuple(wrq_t.shape) != tuple(ref_weight.shape):
                raise ValueError(
                    f"GPTQ weight_res_q for '{name}' has shape {tuple(wrq_t.shape)}, "
                    f"expected {tuple(ref_weight.shape)}"
                )
            return wrq_t
        # GPTQ baseline format: dense GPTQ-quantized weight stored at "baseline_q".
        baseline_q = _record_get(record, ("baseline_q",))
        if baseline_q is not None:
            baseline_t = _to_tensor(baseline_q).to(dtype=torch.float32)
            if tuple(baseline_t.shape) != tuple(ref_weight.shape):
                raise ValueError(
                    f"GPTQ baseline_q for '{name}' has shape {tuple(baseline_t.shape)}, "
                    f"expected {tuple(ref_weight.shape)}"
                )
            return baseline_t.to(dtype=ref_weight.dtype)

        # Legacy fused format.
        weight_q = _record_get(record, ("weight_q", "W_q", "quant_weight", "weight"))
        if weight_q is None:
            raise ValueError(f"GPTQ record for '{name}' is missing quantized weight data")
        weight_q_t = _to_tensor(weight_q).to(dtype=ref_weight.dtype)
        if tuple(weight_q_t.shape) != tuple(ref_weight.shape):
            raise ValueError(
                f"GPTQ weight for '{name}' has shape {tuple(weight_q_t.shape)}, "
                f"expected {tuple(ref_weight.shape)}"
            )
        return weight_q_t

    @property
    def weight(self) -> torch.Tensor:
        return self._weight_q if self._quant_available else self._weight_fp

    @torch.no_grad()
    def freeze_fast_path(self, compute_dtype: torch.dtype = torch.bfloat16) -> bool:
        """Fold all runtime state into buffers and enable a branch-free,
        torch.compile-friendly forward. Returns False if act calibration is
        still pending (call again after more warmup forwards).

        Numerics match the eager path's steady state, with one deliberate
        choice: layers with a per-step act_scale_table use the step-MEAN scale
        (identical to the eager fallback used when no step context is set,
        which is the case for the pi0.5 openpi loop).
        """
        if getattr(self, "_fast", False):
            return True
        eff_bits = (
            int(self._record_a_bits)
            if self._record_a_bits is not None
            else int(self.cfg.act_bits or 0)
        )
        scale = None
        if eff_bits > 0:
            if self._has_act_scale_table:
                scale = self._act_scale_table.float().mean(dim=0).reshape(-1)
            elif self._act_scale_initialized and self._act_scale is not None:
                scale = self._act_scale.float().reshape(-1)
            else:
                return False  # calibration pending
            if scale.numel() == 1:
                scale = scale.expand(self.in_features)
            mq = qmax(eff_bits)
            self._fr_qmin, self._fr_qmax = float(-mq - 1), float(mq)
        self._fr_bits = eff_bits
        dev = (self._weight_q if self._quant_available else self._weight_fp).device
        if scale is not None:
            self.register_buffer(
                "_fr_scale", scale.to(device=dev, dtype=torch.float32).contiguous(), persistent=False
            )
        else:
            self._fr_scale = None

        self._fr_rot_mode = 0  # 0=none 1=blocks 2=dense
        if self._has_duquant_rotation_blocks:
            blk = self._duquant_rotation_blocks.to(device=dev, dtype=compute_dtype).contiguous()
            self.register_buffer("_fr_blocks", blk, persistent=False)
            self.register_buffer(
                "_fr_perm",
                self._duquant_rotation_perm.to(device=dev, dtype=torch.int64).contiguous(),
                persistent=False,
            )
            self._fr_nb, self._fr_B = int(blk.shape[0]), int(blk.shape[1])
            self._fr_rot_mode = 1
        elif self._has_duquant_rotation:
            self.register_buffer(
                "_fr_R",
                self._duquant_rotation.to(device=dev, dtype=compute_dtype).contiguous(),
                persistent=False,
            )
            self._fr_rot_mode = 2

        w = self._weight_q if self._quant_available else self._weight_fp
        # Rotation folding (GR00T_GPTQ_FOLD_ROT=1): reconstruct the effective
        # unrotated linear map W_eq = R_out^T · W · R^T so the runtime carries
        # ZERO rotation cost. Activations are then quantized per-token by the
        # real-quant kernels (QServe int4/int8 small-M, FP8 large-M).
        self._fr_fold_rot = False
        if (
            os.environ.get("GR00T_GPTQ_FOLD_ROT", "0") not in ("0", "false", "False")
            and os.environ.get("GR00T_GPTQ_QSERVE", "0") not in ("0", "false", "False")
            and (
                os.environ.get("GR00T_GPTQ_FP8", "0") not in ("0", "false", "False")
                or os.environ.get("GR00T_GPTQ_A16_LARGE_M", "0") not in ("0", "false", "False")
            )
        ):
            wf = w.detach().float()
            # Fold input rotation blockwise: W_eq = W · R^T with R = P∘blockdiag.
            # (x@R)[j-th block col] mixes x[perm[block j]] — so W·R^T means:
            # W_eq[:, perm[b*B+i]] = sum_j W[:, b*B+j] * blk[b][i, j].
            if self._has_duquant_rotation_blocks:
                blk = self._duquant_rotation_blocks.float().to(wf.device)   # [nb,B,B]
                perm = self._duquant_rotation_perm.to(torch.int64).to(wf.device)
                nb, B, _ = blk.shape
                wcols = wf.reshape(self.out_features, nb, B)                # W in rotated-in basis
                mixed = torch.einsum("onb,nib->oni", wcols, blk)            # W[:,bB+j]·blk[b,i,j]
                weq = torch.empty_like(wf)
                weq[:, perm] = mixed.reshape(self.out_features, nb * B)
                wf = weq
            elif self._has_duquant_rotation:
                wf = wf @ self._duquant_rotation.float().to(wf.device).t()
            # Fold output rotation blockwise: W_eq = R_out^T · W (block-diagonal, no perm)
            if self._has_duquant_rotation_out_blocks:
                ob = self._duquant_rotation_out_blocks.float().to(wf.device)  # [nbo,Bo,Bo]
                nbo, Bo, _ = ob.shape
                wrows = wf.reshape(nbo, Bo, self.in_features)
                wf = torch.einsum("nji,njk->nik", ob, wrows).reshape(self.out_features, self.in_features)
            elif self._has_duquant_rotation_out:
                wf = self._duquant_rotation_out.float().to(wf.device).t() @ wf
            w = wf.to(w.dtype)
            self._fr_fold_rot = True
            self._fr_rot_mode = 0  # rotations are folded into the weight
        self.register_buffer("_fr_weight", w.to(dtype=compute_dtype).contiguous(), persistent=False)
        # Real W4A8 via QServe (GR00T_GPTQ_QSERVE=1): int4-packed weights +
        # per-token dynamic int8 activations on INT8 tensor cores. Wins at
        # small M (expert/DiT calls); gated by GR00T_GPTQ_QSERVE_MAX_M.
        self._fr_qserve = False
        if os.environ.get("GR00T_GPTQ_QSERVE", "0") not in ("0", "false", "False"):
            try:
                import omniserve_backend.qgemm_w4a8_per_chn  # noqa: F401

                wq = w.detach().float()
                mx, mn = wq.amax(1), wq.amin(1)
                s1 = ((mx - mn) / 15.0).clamp_min(1e-6)
                zr = torch.clamp(torch.round(-mn / s1), 0, 15)
                qg = torch.clamp(torch.round(wq / s1[:, None]) + zr[:, None], 0, 15).to(torch.int8)
                # QServe tensor-core packing (per-channel branch of from_linear)
                oc, ic = wq.shape
                r = (
                    qg.reshape(oc // 32, 2, 2, 8, ic // 32, 2, 4, 4)
                    .permute(0, 4, 3, 6, 1, 5, 2, 7)
                    .contiguous()
                    .permute(0, 1, 2, 3, 5, 6, 7, 4)
                    .contiguous()
                )
                packed = ((r[..., 1] << 4) + r[..., 0]).reshape(oc, ic // 2)
                self.register_buffer("_qs_qweight", packed.to(device=dev).contiguous(), persistent=False)
                self.register_buffer("_qs_s1", s1.half().to(dev).contiguous(), persistent=False)
                self.register_buffer("_qs_szeros", (zr * s1).half().to(dev).contiguous(), persistent=False)
                self._qs_max_m = int(os.environ.get("GR00T_GPTQ_QSERVE_MAX_M", "128"))
                self._fr_qserve = oc % 32 == 0 and ic % 32 == 0
            except Exception as e:  # pragma: no cover
                print(f"[GPTQ] qserve pack failed for {self.name}: {e}")
        # FP8 path for large-M calls (GR00T_GPTQ_FP8=1): per-row fp8 weights +
        # per-token fp8 activations via torch._scaled_mm (H100 native, ~2x
        # bf16 on compute-bound GEMMs). Used when M exceeds the QServe gate.
        self._fr_fp8 = False
        if os.environ.get("GR00T_GPTQ_FP8", "0") not in ("0", "false", "False"):
            try:
                wf8 = w.detach().float()
                s_wr = (wf8.abs().amax(dim=1) / 448.0).clamp_min(1e-12)  # e4m3 max
                w_fp8 = (wf8 / s_wr[:, None]).to(torch.float8_e4m3fn).contiguous()
                # [K, N] column-major view — the layout _scaled_mm requires for mat2
                self.register_buffer("_fp8_w_t", w_fp8.t(), persistent=False)
                self.register_buffer("_fp8_w_scale", s_wr.to(torch.float32).reshape(1, -1).contiguous(), persistent=False)
                self._fr_fp8 = True
            except Exception as e:  # pragma: no cover
                print(f"[GPTQ] fp8 pack failed for {self.name}: {e}")
        # Real-INT8 path (GR00T_GPTQ_REAL_INT8=1): fold the static per-channel
        # act scale into the weight, re-quantize rows to int8, and run the
        # main GEMM on INT8 tensor cores via torch._int_mm. Requires a static
        # act scale (frozen above).
        self._fr_int8 = False
        if (
            os.environ.get("GR00T_GPTQ_REAL_INT8", "0") not in ("0", "false", "False")
            and scale is not None
        ):
            wf = w.detach().float() * scale.to(w.device).reshape(1, -1)  # fold s_act
            s_row = (wf.abs().amax(dim=1) / 127.0).clamp_min(1e-12)      # [out]
            w_i8 = torch.clamp(torch.round(wf / s_row[:, None]), -127, 127).to(torch.int8)
            self.register_buffer("_fr_w_int8_t", w_i8.t().contiguous(), persistent=False)  # [in, out]
            self.register_buffer("_fr_w_row_scale", s_row.to(torch.float32), persistent=False)
            aq = qmax(eff_bits)
            self._fr_a_qmax = float(aq)
            self._fr_int8 = True
        if self.bias is not None:
            self.register_buffer(
                "_fr_bias", self.bias.to(dtype=compute_dtype, device=dev).contiguous(), persistent=False
            )
        else:
            self._fr_bias = None
        # SmoothQuant fold: x/s before quant == scale_eff = s*scale on divide,
        # multiply-back stays `scale`. Not supported in fast path — bail out.
        if self._has_smooth_scale and self._smooth_scale.numel() == self.in_features:
            return False
        if self._has_svdquant and self._lowrank_A.numel() > 0:
            return False
        self._fr_out_mode = 0
        if self._fr_fold_rot:
            pass  # output rotation folded into the weight
        elif self._has_duquant_rotation_out_blocks:
            ob = self._duquant_rotation_out_blocks.to(device=dev, dtype=compute_dtype).contiguous()
            self.register_buffer("_fr_out_blocks", ob, persistent=False)
            self._fr_nb_out, self._fr_B_out = int(ob.shape[0]), int(ob.shape[1])
            self._fr_out_mode = 1
        elif self._has_duquant_rotation_out:
            self.register_buffer(
                "_fr_R_out",
                self._duquant_rotation_out.to(device=dev, dtype=compute_dtype).contiguous(),
                persistent=False,
            )
            self._fr_out_mode = 2
        # A16 fallback for large-M calls (GR00T_GPTQ_A16_LARGE_M=1): skip
        # activation quantization when the QServe small-M gate doesn't apply.
        # Weights stay W4 (baked); prefill GEMM runs at native bf16 speed and
        # accuracy is strictly better than A8 there.
        self._fr_a16_large = os.environ.get("GR00T_GPTQ_A16_LARGE_M", "0") not in ("0", "false", "False")
        self._fast = True
        return True

    def _forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        w = self._fr_weight
        if self._fr_rot_mode == 1:
            sh = x.shape
            x2 = x.reshape(-1, sh[-1]).index_select(-1, self._fr_perm)
            x2 = x2.reshape(-1, self._fr_nb, self._fr_B).transpose(0, 1)
            x2 = torch.bmm(x2.to(w.dtype), self._fr_blocks)
            x = x2.transpose(0, 1).reshape(sh)
        elif self._fr_rot_mode == 2:
            x = torch.matmul(x.to(w.dtype), self._fr_R)
        if self._fr_qserve and x.numel() // x.shape[-1] <= self._qs_max_m:
            from gr00t.quantization.qserve_op import qserve_linear

            sh = x.shape
            y16 = qserve_linear(x.reshape(-1, sh[-1]), self._qs_qweight, self._qs_s1, self._qs_szeros)
            y = y16.reshape(*sh[:-1], y16.shape[-1]).to(w.dtype)
        elif getattr(self, "_fr_a16_large", False):
            y = torch.nn.functional.linear(x.to(w.dtype), w)
        elif self._fr_fp8:
            sh = x.shape
            x2 = x.reshape(-1, sh[-1]).float()
            s_a = (x2.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
            a8 = (x2 / s_a).to(torch.float8_e4m3fn)
            from gr00t.quantization.qserve_op import fp8_rowwise_mm

            y = fp8_rowwise_mm(a8, self._fp8_w_t, s_a.contiguous(), self._fp8_w_scale)
            y = y.reshape(*sh[:-1], y.shape[-1]).to(w.dtype)
        elif self._fr_int8:
            sh = x.shape
            a = torch.clamp(
                torch.round(x.reshape(-1, sh[-1]).float() / self._fr_scale),
                -self._fr_a_qmax - 1, self._fr_a_qmax,
            ).to(torch.int8)
            m = a.shape[0]
            if m < 17:  # cuBLAS int8 GEMM requires M > 16
                a = torch.nn.functional.pad(a, (0, 0, 0, 17 - m))
            y = torch._int_mm(a, self._fr_w_int8_t)[:m]  # [M, out] int32, INT8 tensor cores
            y = y.to(torch.float32) * self._fr_w_row_scale
            y = y.reshape(*sh[:-1], y.shape[-1]).to(w.dtype)
        else:
            if self._fr_scale is not None:
                s = self._fr_scale
                x = torch.clamp(torch.round(x.float() / s), self._fr_qmin, self._fr_qmax) * s
            y = torch.nn.functional.linear(x.to(w.dtype), w)
        if self._fr_out_mode == 1:
            sh_y = y.shape
            y2 = y.reshape(-1, self._fr_nb_out, self._fr_B_out).transpose(0, 1)
            y2 = torch.bmm(y2.to(self._fr_out_blocks.dtype), self._fr_out_blocks)
            y = y2.transpose(0, 1).reshape(sh_y)
        elif self._fr_out_mode == 2:
            y = torch.matmul(y.to(self._fr_R_out.dtype), self._fr_R_out)
        if self._fr_bias is not None:
            y = y + self._fr_bias
        return y.to(orig_dtype)

    @torch._dynamo.disable
    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act_bits <= 0:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

        if self._act_scale_initialized and self._act_scale is not None:
            return self._act_scale

        with torch.no_grad():
            if self.calibrator is not None and not self.calibrator.is_full():
                self.calibrator.observe(x)
                if self.calibrator.is_full():
                    p_vec = self.calibrator.finalize()
                    max_q = qmax(int(self.cfg.act_bits or 8))
                    scale = torch.clamp(p_vec / max_q, min=1e-6)
                    scale = scale.to(dtype=x.dtype, device=x.device).clone()
                    self._act_scale = scale
                    self._act_scale_initialized = True
                    self._dump_act_stats(p_vec)

            if not self._act_scale_initialized:
                x_abs = torch.abs(x.detach().to(torch.float32))
                x2d = x_abs.reshape(-1, x_abs.shape[-1])
                p_vec = torch.quantile(x2d, float(self.cfg.act_percentile or 99.9) / 100.0, dim=0)
                max_q = qmax(int(self.cfg.act_bits or 8))
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                scale = scale.to(dtype=x.dtype, device=x.device).clone()
                self._act_scale = scale
                self._act_scale_initialized = True
                self._dump_act_stats(p_vec)

        return self._act_scale if self._act_scale is not None else torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

    def _dump_act_stats(self, p_vec: torch.Tensor) -> None:
        """Dump per-channel A8 percentile vector to a JSON file when
        GR00T_ACT_STATS_DIR is set (one file per layer per process).
        """
        out_dir = os.environ.get("GR00T_ACT_STATS_DIR")
        if not out_dir:
            return
        try:
            import json
            from pathlib import Path
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            safe_name = self.name.replace("/", "_")
            (d / f"{safe_name}.json").write_text(json.dumps({
                "layer": self.name,
                "method": "gptq",
                "act_bits": int(self.cfg.act_bits or 0),
                "percentile": float(self.cfg.act_percentile or 99.9),
                "calib_batches": int(self.cfg.calib_batches or 0),
                "n_channels": int(p_vec.numel()),
                "p_vec": p_vec.detach().cpu().tolist(),
            }))
        except Exception:
            pass

    def _resolve_fused_act_scale(self, x: torch.Tensor):
        """Static per-channel act scale (rotated space) for the fused path.

        Returns (scale_or_None, eff_act_bits). scale None with bits>0 means
        no static scale available yet (calibration pending) -> use eager path.
        """
        eff = (
            int(self._record_a_bits)
            if self._record_a_bits is not None
            else int(self.cfg.act_bits or 0)
        )
        if eff <= 0:
            return None, 0
        if self._has_act_scale_table:
            from gr00t.quantization.dit_step_context import get_current_dit_step

            cur = get_current_dit_step()
            if cur is not None and 0 <= cur < self._act_scale_table.shape[0]:
                return self._act_scale_table[cur], eff
            if not hasattr(self, "_act_scale_table_mean_buf"):
                self.register_buffer(
                    "_act_scale_table_mean_buf",
                    self._act_scale_table.mean(dim=0),
                    persistent=False,
                )
            return self._act_scale_table_mean_buf, eff
        if self._act_scale_initialized and self._act_scale is not None:
            return self._act_scale, eff
        return None, eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_fast", False):
            return self._forward_fast(x)
        did_fused = False
        # Fused Triton path: permute + block-rotation + act fake-quant in one
        # kernel. Only when no SmoothQuant scale / SVDQuant branch is involved
        # and a static act scale is available (or act quant disabled).
        if (
            self._fused_enabled
            and x.numel() // x.shape[-1] >= self._fused_min_tokens
            and (self._has_duquant_rotation_blocks or self._has_duquant_rotation)
            and (not self._has_svdquant or self._lowrank_A.numel() == 0)
            and not (self._has_smooth_scale and self._smooth_scale.numel() == x.shape[-1])
        ):
            f_scale, f_bits = self._resolve_fused_act_scale(x)
            if f_bits == 0 or f_scale is not None:
                from gr00t.quantization.fused_rot_quant import rot_quant

                if not hasattr(self, "_rot_blocks_f32"):
                    if self._has_duquant_rotation_blocks:
                        blocks = self._duquant_rotation_blocks.to(
                            device=x.device, dtype=torch.float32
                        ).contiguous()
                        perm = self._duquant_rotation_perm.to(
                            device=x.device, dtype=torch.int32
                        ).contiguous()
                    else:
                        # Dense R from the pack. DuQuant rotations are
                        # (permutation ∘ block-diagonal) with B=64: every ROW
                        # of R has support inside exactly one 64-column block.
                        # Recover perm+blocks; verify by reconstruction; fall
                        # back to eager permanently if the structure is absent.
                        R = self._duquant_rotation.to(device=x.device, dtype=torch.float32)
                        Bsz = 64
                        in_f = R.shape[0]
                        blocks = None
                        if in_f % Bsz == 0:
                            n_b = in_f // Bsz
                            # energy of each row within each column-block
                            e = (R.reshape(in_f, n_b, Bsz) ** 2).sum(-1)  # [in, n_b]
                            row_blk = e.argmax(dim=1)                      # [in]
                            off_energy = float(e.sum() - e.gather(1, row_blk[:, None]).sum())
                            counts = torch.bincount(row_blk, minlength=n_b)
                            if off_energy < 1e-4 and bool((counts == Bsz).all()):
                                order = torch.argsort(row_blk * in_f + torch.arange(in_f, device=R.device))
                                perm = order.to(torch.int32).contiguous()
                                rows = R.index_select(0, order)            # [in, in]
                                blocks = torch.stack(
                                    [rows[b * Bsz:(b + 1) * Bsz, b * Bsz:(b + 1) * Bsz] for b in range(n_b)]
                                ).contiguous()
                                # algebraic verification: rebuild dense R from
                                # (perm, blocks) and compare exactly.
                                rec = torch.zeros_like(R)
                                for b in range(n_b):
                                    rec[order[b * Bsz:(b + 1) * Bsz], b * Bsz:(b + 1) * Bsz] = blocks[b]
                                if float((rec - R).abs().max()) > 1e-5:
                                    blocks = None
                        if blocks is None:
                            self._fused_enabled = False
                    if blocks is not None:
                        self.register_buffer("_rot_blocks_f32", blocks, persistent=False)
                        self.register_buffer("_rot_perm_i32", perm, persistent=False)
                if hasattr(self, "_rot_blocks_f32"):
                    orig_shape = x.shape
                    x2 = x.reshape(-1, orig_shape[-1])
                    if not x2.is_contiguous():
                        x2 = x2.contiguous()
                    y2 = rot_quant(x2, self._rot_perm_i32, self._rot_blocks_f32, f_scale, f_bits)
                    x = y2.reshape(*orig_shape)
                    did_fused = True
                    self._fused_calls = getattr(self, "_fused_calls", 0) + 1
        # DuQuant rotation preprocessing (if present): x' = x @ R
        # Applied BEFORE SmoothQuant scaling because R is on raw input space;
        # the pack's smooth_scale was computed on rotated activations.
        if not did_fused and self._has_duquant_rotation_blocks:
            # Fast path: x[:, perm].view(..., n_blocks, B) @ R_blocks (bmm).
            # Equivalent to x @ (P @ block_diag(R_b)) where P is perm matrix.
            blocks = self._duquant_rotation_blocks.to(dtype=x.dtype, device=x.device)  # [n, B, B]
            perm = self._duquant_rotation_perm.to(device=x.device)
            in_f = blocks.shape[0] * blocks.shape[1]
            B = blocks.shape[1]
            n_blocks = blocks.shape[0]
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1])           # (N, in)
            x2 = x2.index_select(dim=-1, index=perm)     # (N, in)  permuted
            x2 = x2.reshape(-1, n_blocks, B)             # (N, n, B)
            x2 = torch.bmm(x2.transpose(0, 1).contiguous(), blocks)  # (n, N, B)
            x2 = x2.transpose(0, 1).contiguous().reshape(-1, in_f)
            x = x2.reshape(*orig_shape[:-1], in_f)
        elif self._has_duquant_rotation:
            R = getattr(self, "_rot_R_cast", None)
            if R is None or R.dtype != x.dtype or R.device != x.device:
                R = self._duquant_rotation.to(dtype=x.dtype, device=x.device)
                self._rot_R_cast = R
            x = torch.matmul(x, R)
        # SmoothQuant: divide input by per-channel smooth scale so that the
        # forward sees activations matched to the smoothed-W stored in pack.
        if self._has_smooth_scale and self._smooth_scale.numel() == x.shape[-1]:
            x = x / self._smooth_scale.to(dtype=x.dtype, device=x.device)
        # SVDQuant lowrank branch consumes the smoothed (but un-A-quantized)
        # activation x_s. We compute it before any A4/A8 fake-quant on the
        # residual path so the low-rank correction stays at full precision.
        x_s_for_lowrank: Optional[torch.Tensor] = (
            x if self._has_svdquant else None
        )
        x_q = x
        # Per-record a_bits override: SVDQuant records store their own a_bits
        # (e.g. 4 for W4A4 DiT pack) so that mixed-precision packs can use
        # cfg.act_bits for non-record layers (e.g. LLM A8) and the record's
        # value for SVDQuant DiT layers (A4). Falls back to cfg.act_bits.
        eff_act_bits = (
            int(self._record_a_bits)
            if self._record_a_bits is not None
            else int(self.cfg.act_bits or 0)
        )
        if eff_act_bits > 0 and not did_fused:
            # Per-DiT-step scale dispatch: if this layer has an act_scale_table
            # AND a denoising-step context is active AND the step is in range,
            # use that step's row.
            step_scale: Optional[torch.Tensor] = None
            if self._has_act_scale_table:
                from gr00t.quantization.dit_step_context import get_current_dit_step
                cur = get_current_dit_step()
                if cur is not None and 0 <= cur < self._act_scale_table.shape[0]:
                    tbl = getattr(self, "_act_table_cast", None)
                    if tbl is None or tbl.dtype != x.dtype or tbl.device != x.device:
                        tbl = self._act_scale_table.to(dtype=x.dtype, device=x.device)
                        self._act_table_cast = tbl
                    step_scale = tbl[cur]
                elif cur is not None and not getattr(self, "_act_table_oob_warned", False):
                    print(
                        f"[GPTQ][WARN] {self.name}: act_scale_table has "
                        f"{self._act_scale_table.shape[0]} steps but current_dit_step={cur} — "
                        f"falling back to step-mean scale. (warned once per layer)",
                        flush=True,
                    )
                    self._act_table_oob_warned = True
            if step_scale is not None:
                x_q = fake_quantize_sym(x, step_scale, eff_act_bits, label="gptq_activation_per_step")
            elif self._has_act_scale_table:
                # Deterministic fallback for DiT layers that have a table but
                # the step context is missing/oob: use the step-mean scale.
                # Do NOT fall through to the running PercentileCalibrator,
                # which would otherwise silently corrupt the rollout with a
                # mismatched dynamic estimate.
                if not hasattr(self, "_act_scale_table_mean_buf"):
                    mean_scale = self._act_scale_table.mean(dim=0)
                    self.register_buffer("_act_scale_table_mean_buf", mean_scale, persistent=False)
                fb = self._act_scale_table_mean_buf.to(dtype=x.dtype, device=x.device)
                x_q = fake_quantize_sym(x, fb, eff_act_bits, label="gptq_activation_table_fallback")
            else:
                s_a = self._get_act_scale(x)
                x_q = fake_quantize_sym(x, s_a, eff_act_bits, label="gptq_activation")
        weight = self._weight_q if self._quant_available else self._weight_fp
        # Promote main matmul to avoid FP16 overflow: q99.9 × W4_max ×
        # in_features can exceed 65504 (FP16 max), causing catastrophic
        # NaN cascades on SVDQuant W4A8 + LLM-FP16. Real INT4 GEMM (cutlass)
        # uses INT32 accumulator and avoids this; our fake-quant emulation
        # needs explicit promote.
        # Use BF16 (range = FP32, BF16 native on A100/H100, ~zero overhead)
        # over FP32 (~10% slower). Falls back to FP32 on older GPUs.
        out_dtype = x_q.dtype
        promote_dtype = (
            torch.bfloat16
            if (x_q.is_cuda and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        x_q_p = x_q.to(promote_dtype)
        w_p = self._w_promoted_cache
        if w_p is None or w_p.dtype != promote_dtype or w_p.device != x_q.device:
            w_p = weight.to(dtype=promote_dtype, device=x_q.device)
            self._w_promoted_cache = w_p
        y = torch.nn.functional.linear(x_q_p, w_p, None)
        # SVDQuant: add lowrank correction y += (x_s @ B) @ A.T (also promoted)
        if self._has_svdquant and x_s_for_lowrank is not None:
            B = self._lowrank_B.to(dtype=promote_dtype, device=x_q.device)
            A = self._lowrank_A.to(dtype=promote_dtype, device=x_q.device)
            x_s_p = x_s_for_lowrank.to(promote_dtype)
            z = torch.nn.functional.linear(x_s_p, B.t())   # [..., r]
            y_lr = torch.nn.functional.linear(z, A)         # [..., out]
            y = y + y_lr
        # Optional DuQuant output rotation restore: y_restored = y @ R_out.
        # Applied BEFORE bias (bias lives in the un-rotated output basis).
        if self._has_duquant_rotation_out_blocks:
            # Fast path: bmm over n_out blocks of size B_out.
            blocks_out = self._duquant_rotation_out_blocks.to(dtype=y.dtype, device=y.device)
            out_f = blocks_out.shape[0] * blocks_out.shape[1]
            B_out = blocks_out.shape[1]
            n_out = blocks_out.shape[0]
            orig_shape = y.shape
            y2 = y.reshape(-1, orig_shape[-1])
            y2 = y2.reshape(-1, n_out, B_out)
            y2 = torch.bmm(y2.transpose(0, 1).contiguous(), blocks_out)
            y2 = y2.transpose(0, 1).contiguous().reshape(-1, out_f)
            y = y2.reshape(*orig_shape[:-1], out_f)
        elif self._has_duquant_rotation_out:
            R_out = self._duquant_rotation_out.to(dtype=y.dtype, device=y.device)
            y = torch.matmul(y, R_out)
        # Cast back to original dtype
        y = y.to(dtype=out_dtype)
        if self.bias is not None:
            b = getattr(self, "_bias_cast", None)
            if b is None or b.dtype != y.dtype or b.device != y.device:
                b = self.bias.to(dtype=y.dtype, device=y.device)
                self._bias_cast = b
            y = y + b
        if self._debug_enabled and not hasattr(self, "_debug_forward_logged"):
            print(
                f"[GR00T-GPTQ][FORWARD] {self.name} input={tuple(x.shape)} output={tuple(y.shape)} "
                f"W{self.weight_bits} A{self.cfg.act_bits} quant={int(self._quant_available)}",
                flush=True,
            )
            self._debug_forward_logged = True
        return y


@torch.no_grad()
def freeze_gptq_fast_paths(model: nn.Module, compute_dtype: torch.dtype = torch.bfloat16):
    """Freeze all GptqLinear layers into the compile-friendly fast path.
    Call after enough warmup forwards for act-scale calibration to finish.
    Returns (frozen, pending) counts."""
    frozen = pending = 0
    for m in model.modules():
        if isinstance(m, GptqLinear):
            if m.freeze_fast_path(compute_dtype=compute_dtype):
                frozen += 1
            else:
                pending += 1
    print(f"[GPTQ] fast-path frozen: {frozen}, pending: {pending}")
    return frozen, pending


def wrap_gptq(
    model: nn.Module,
    layer_names: Iterable[str],
    cfg: GptqConfig,
    per_layer_wbits: Optional[Dict[str, int]] = None,
    dry_run: bool = False,
) -> int:
    per_layer_wbits = per_layer_wbits or {}
    replaced = 0
    listed = 0
    for name in layer_names:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        wbits = per_layer_wbits.get(name, cfg.weight_bits)
        if dry_run:
            print(
                f"[GR00T-GPTQ][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{cfg.act_bits}"
            )
            listed += 1
            continue
        wrapped = GptqLinear(mod, name=name, cfg=cfg, weight_bits=wbits)
        setattr(parent, attr, wrapped)
        print(
            f"[GR00T-GPTQ][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) "
            f"-> GptqLinear W{wbits} A{cfg.act_bits} quant={int(wrapped._quant_available)}"
        )
        replaced += 1
    if dry_run:
        print(f"[GR00T-GPTQ] Dry-run total layers listed: {listed}")
        return listed
    print(f"[GR00T-GPTQ] Total layers replaced: {replaced}")
    return replaced


def enable_gptq_if_configured(model: nn.Module) -> bool:
    env = os.environ
    activate = env.get("GR00T_GPTQ", "0") not in ("0", "false", "False")
    if not activate:
        return False

    scope = env.get("GR00T_GPTQ_SCOPE", "")
    whitelist = env.get("GR00T_GPTQ_LAYERS")
    whitelist_list = [x.strip() for x in whitelist.split(",") if x.strip()] if whitelist else None
    inc = env.get(
        "GR00T_GPTQ_INCLUDE",
        (
            r".*(?:"
            r"backbone\.eagle_model\.language_model\..*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            r"|"
            r"action_head\.model\..*(?:attn1\.to_(?:q|k|v)|attn1\.to_out\.0|ff\.net\.(?:0\.proj|2))"
            r").*"
        ),
    )
    exc = env.get(
        "GR00T_GPTQ_EXCLUDE",
        (
            r"(?:^|\.)"
            r"(?:vision_model|vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|future_tokens|vl_self_attention)"
            r"(?:\.|$)"
        ),
    )
    per_layer_wbits = _parse_per_layer_wbits(env.get("GR00T_GPTQ_WBITS"))
    dry_run = env.get("GR00T_GPTQ_DRYRUN", "0") not in ("0", "false", "False")

    cfg = GptqConfig()
    targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope if scope else None,
        whitelist=whitelist_list,
        blacklist=None,
    )
    layer_names = [n for n, _ in targets]
    print(f"[GR00T-GPTQ] SCOPE filter: '{scope}'")
    print(f"[GR00T-GPTQ] Matched Linear layers: {len(layer_names)}")
    wrap_gptq(model, layer_names, cfg, per_layer_wbits, dry_run=dry_run)
    return True
