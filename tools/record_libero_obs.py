#!/usr/bin/env python
"""Record LIBERO observations in openpi element format for pi0.5 calibration.

Runs in the LIBERO-capable env (omega_qvla). One obs per task by default.
Output: torch .pt file with {"samples": [element, ...]}.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch


def quat2axisangle(quat):
    # ported from openpi examples/libero/main.py (robosuite convention)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    p.add_argument("--out-path", required=True)
    p.add_argument("--per-task", type=int, default=1)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--resize", type=int, default=224)
    args = p.parse_args()

    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools

    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    samples = []
    bdict = benchmark.get_benchmark_dict()

    for suite_name in args.task_suites.split(","):
        suite = bdict[suite_name.strip()]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            task_bddl = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
            env.seed(7)
            for ep in range(args.per_task):
                env.reset()
                obs = env.set_init_state(init_states[ep])
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize, args.resize)
                )
                wrist = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist, args.resize, args.resize)
                )
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"].copy()),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task.language),
                }
                samples.append(element)
            env.close()
            print(f"[record] {suite_name} task {task_id}: {task.language}")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save({"samples": samples}, args.out_path)
    print(f"[record] saved {len(samples)} samples -> {args.out_path}")


if __name__ == "__main__":
    main()
