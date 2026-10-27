"""Find the first nonfinite operator without replacing any inference path."""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import tree
from research.holoq_bench.collect_calibration import seed


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--deployment", required=True, type=Path)
    p.add_argument("--observations", required=True, type=Path)
    a = p.parse_args()
    os.environ.update(HOLOQ_DEPLOY=str(a.deployment.resolve()), HOLOQ_COMPUTE_DTYPE="float16")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import load_data_config
    from research.holoq_bench.packed_runtime import PackedLinear

    cfg = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path=str(a.deployment.resolve()),
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
    )

    def check(name, kind, values):
        for value in tree.flatten(values):
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                print(
                    "FIRST_NONFINITE",
                    name,
                    kind,
                    "shape",
                    list(value.shape),
                    "dtype",
                    value.dtype,
                    "nan",
                    value.isnan().sum().item(),
                    "inf",
                    value.isinf().sum().item(),
                    flush=True,
                )
                raise RuntimeError("Nonfinite operator " + name)

    def hook(name):
        def output(module, args, result):
            if isinstance(module, PackedLinear) and not torch.isfinite(result).all():
                from research.holoq_bench.packed_runtime import apply_blocks
                from research.holoq_bench.kernels import pack_marlin

                pack = torch.load(a.deployment / "packed.pt", mmap=True, weights_only=False)
                rr = dict(pack["layers"][name])
                method = "quantvla" if "quantvla" in a.deployment.name else "duquant"
                root = Path("results/holoq_object/packs_v2/layer_cache") / method
                entry = next(
                    torch.load(f, weights_only=False)
                    for f in root.glob("*.pt")
                    if torch.load(f, weights_only=False)["name"] == name
                )
                c = entry["canonical"]
                xx = args[0].half()
                xx = apply_blocks(xx, module.rot_in, module.perm).reshape(-1, module.in_features)
                aa = (xx.float() / module.a_scale).round().clamp(module.act_qmin, 127) * module.a_scale
                ww = c["q"].float() * c["scales"].float()
                with torch.autocast("cuda", enabled=False):
                    oracle = aa.float() @ ww.cuda().t()
                print(
                    "NUMERICAL_ORACLE",
                    "input",
                    args[0].abs().max().item(),
                    "rotated",
                    xx.abs().max().item(),
                    "activation",
                    aa.abs().max().item(),
                    "weight_scale",
                    c["scales"].min().item(),
                    c["scales"].max().item(),
                    "oracle_finite",
                    torch.isfinite(oracle).all().item(),
                    "oracle_peak",
                    oracle.abs().max().item(),
                    flush=True,
                )
                rr.update(pack_marlin(c["q"], c["scales"].repeat(1, module.in_features // 128)))
                test = PackedLinear(rr, "exact_same_grid_g128").cuda()
                alternative = test(args[0])
                print(
                    "G128_REENCODE",
                    torch.isfinite(alternative).all().item(),
                    alternative.abs().max().item(),
                    flush=True,
                )
            check(name, "output", result)

        return output

    def prehook(name):
        def inputs(module, args):
            check(name, "input", args)

        return inputs

    for name, m in policy.model.named_modules():
        if isinstance(m, (PackedLinear, torch.nn.Linear, torch.nn.LayerNorm)):
            m.register_forward_pre_hook(prehook(name))
            m.register_forward_hook(hook(name))
    observations = torch.load(a.observations, weights_only=False)
    for i in range(10):
        seed(1000 + i)
        out = policy.get_action(observations[i * 10]["obs"])
        print("SAMPLE", i, "finite", all(np.isfinite(v).all() for v in out.values()), flush=True)


if __name__ == "__main__":
    main()
