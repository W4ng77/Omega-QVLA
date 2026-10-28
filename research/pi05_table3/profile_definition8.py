"""Eager phase trace for the already-validated, frozen8-step deployment."""
import dataclasses
import fcntl
import json
from pathlib import Path
import numpy as np
import torch
from research.pi05_table3.measure import sha, write


def main():
    guard=open('/tmp/holoq-vla-policy-timing.lock','a')
    fcntl.flock(guard,fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from openpi.training import config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from research.pi05_table3.definition_runtime import load_policy
    from research.pi05_table3.measure_scopes import profile_policy
    root=Path(__file__).resolve().parents[2]
    data=root/'results/pi05_table3_414_20261028'
    frozen=json.loads((data/'definition8_measurement/report.json').read_text())
    assert frozen['validation_passed'] and frozen['status']=='complete'
    cfg=config.get_config('pi05_libero')
    cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    directory=data/'definition8_deployment_fp64'
    assert sha(directory/'deployment.json')==frozen['deployment_manifest_sha256']
    policy=load_policy(directory,cfg,'packed')
    policy._model.eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False
    install_pi05_step_context(policy._model,fixed_loop=True)
    policy._sample_actions=policy._model.sample_actions
    sample=torch.load('/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt',map_location='cpu',weights_only=False)['samples'][10]
    noise=np.random.default_rng(42).standard_normal((cfg.model.action_horizon,cfg.model.action_dim)).astype(np.float32)
    output=data/'definition8_profile'
    output.mkdir(exist_ok=False)
    with torch.inference_mode():
        for _ in range(32):
            policy.infer(sample,noise=noise)
        torch.cuda.synchronize()
        profile_policy(policy,sample,noise,output,raw_only=True)
    write(output/'provenance.json',{'deployment_manifest_sha256':sha(directory/'deployment.json'),
        'timing_report_sha256':sha(data/'definition8_measurement/report.json'),
        'source_sha256':sha(Path(__file__)),'profile_helper_sha256':sha(root/'research/pi05_table3/measure_scopes.py'),
        'steps':8,'calls':5,'note':'Diagnostic eager profiling, not formal wall latency. Only raw activity attribution with valid kernel counts may be used.'})


if __name__=='__main__':
    main()
