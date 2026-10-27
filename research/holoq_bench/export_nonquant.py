"""Store the exact necessary non-quantized model parameters for byte accounting."""

import argparse
import json
import os
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--capture", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    os.environ["HOLOQ_COMPUTE_DTYPE"] = "float16"
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")
    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=config.modality_config(),
        modality_transform=config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=8,
        device="cpu",
    )
    state = {k: v.detach().cpu() for k, v in policy.model.state_dict().items()}
    # Keep one common FP16 state archive. Deployment accounting excludes exactly
    # the selected quantized weights/biases for each individual configuration.
    output = args.capture / "fp16_model_state.pt"
    if output.exists():
        raise FileExistsError(output)
    torch.save(state, output)
    metadata = {
        "tensors": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype), "bytes": v.numel() * v.element_size()}
            for k, v in state.items()
        },
        "archive_bytes": output.stat().st_size,
    }
    (args.capture / "fp16_model_state.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Exported", output, output.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
