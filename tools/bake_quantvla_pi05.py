#!/usr/bin/env python
"""Bake a QuantVLA-real checkpoint for pi0.5.

QuantVLA method = runtime DuQuant rotation (svd_hadamard, 64-block) + RTN
weight quantization. For a real-quant deployment we compute, per target layer:
    R    = duquant rotation from W          (their compute_duquant_rotation_only)
    W'   = W @ R                            (weight in rotated-input basis)
    W'_q = RTN groupwise int4(W')           (their scheme, group 64)
    W_eq = W'_q @ R^T                       (fold rotation back out)
and bake W_eq into a checkpoint copy. Deployment then packs W_eq with the
same QServe/A16 pipeline used for all methods. Run in the openpi venv.
"""

import re
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gr00t.quantization.duquant_preprocess import compute_duquant_rotation_only  # noqa: E402
from gr00t.quantization.rtn_layers import _quant_dequant_weight_groupwise  # noqa: E402

SRC = "/work/sicheng/models/pi05_libero_pytorch"
OUT = "/work/sicheng/models/quantvla-pi05-w4"
INCLUDE = re.compile(
    r".*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)"
    r"\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
)


def main():
    from safetensors.torch import load_file, save_file

    sd = load_file(f"{SRC}/model.safetensors")
    n = 0
    for k in list(sd.keys()):
        if not INCLUDE.match(k):
            continue
        W = sd[k].float().cuda()
        R, _ = compute_duquant_rotation_only(W, block_size=64, enable_permute=False, rot_mode="svd_hadamard")
        R = R.to(dtype=torch.float32, device=W.device)
        Wr = W @ R
        in_f = Wr.shape[1]
        gs = 64 if in_f % 64 == 0 else in_f
        Wq = _quant_dequant_weight_groupwise(Wr, gs, 4)
        Weq = (Wq @ R.t()).to(sd[k].dtype).cpu()
        sd[k] = Weq
        n += 1
        if n % 50 == 0:
            print(f"[quantvla-bake] {n} layers done")
    print(f"[quantvla-bake] total baked: {n}")

    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    for item in Path(SRC).iterdir():
        if item.name == "model.safetensors":
            continue
        if item.is_dir():
            shutil.copytree(item, out / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, out / item.name)
    save_file(sd, str(out / "model.safetensors"))
    print(f"[quantvla-bake] saved -> {out}")


if __name__ == "__main__":
    main()
