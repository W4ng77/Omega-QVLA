#!/usr/bin/env bash
# Pin only existing required implementations; preserve every existing checkout.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$task_root/third_party"
while read -r task_name task_url task_revision; do
  task_dir="$task_root/third_party/$task_name"
  if [ -e "$task_dir/.git" ]; then
    task_actual="$(git -C "$task_dir" rev-parse HEAD)"
    if [ "$task_actual" != "$task_revision" ]; then
      echo "Existing $task_name revision differs; preserve it and use a fresh source directory." >&2
      exit 1
    fi
    if [ -n "$(git -C "$task_dir" status --porcelain --untracked-files=no)" ]; then
      echo "Existing $task_name has changes; refusing to overwrite them." >&2
      exit 1
    fi
  else
    git clone "$task_url" "$task_dir"
    git -C "$task_dir" checkout --detach "$task_revision"
    git -C "$task_dir" submodule update --init --recursive
  fi
done <<'PINS'
gptq https://github.com/IST-DASLab/gptq.git 2d65066eeb06a5c9ff5184d8cebdf33662c67faf
marlin https://github.com/IST-DASLab/marlin.git 1f25790bdd49fba53106164a24666dade68d7c90
omniserve https://github.com/mit-han-lab/omniserve.git 02b2925aa6fa3b92b06316a1524b7f38922cd9c8
QuaRot https://github.com/spcl/QuaRot.git 5008669b08c1f11f9b64d52d16fddd47ca754c5a
llm-awq https://github.com/mit-han-lab/llm-awq.git d6e797a42b9ef7778de8ee2352116e0f48a78d61
PINS
