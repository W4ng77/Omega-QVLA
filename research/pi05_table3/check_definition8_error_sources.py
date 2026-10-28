"""Untimed component substitution on identical frozen quantization grids."""
import dataclasses
import fcntl
import gc
from pathlib import Path
import numpy as np
import torch
from research.pi05_table3.measure import sha, write
from research.pi05_table3.measure_definition8 import difference


def main():
    guard=open('/tmp/holoq-vla-policy-timing.lock','a')
    fcntl.flock(guard,fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from openpi.training import config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from research.pi05_table3.definition_runtime import load_policy,DefinitionLinear,unpack_codes
    root=Path(__file__).resolve().parents[2]
    data=root/'results/pi05_table3_414_20261028'
    cfg=config.get_config('pi05_libero')
    cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    samples=torch.load('/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt',map_location='cpu',weights_only=False)['samples'][10:20]
    reference=np.load(data/'definition8_measurement/reference_outputs.npz')
    result={'purpose':'Isolate backend arithmetic differences, not quantization error or task success',
        'source_sha256':sha(Path(__file__)),
        'deployment_manifest_sha256':sha(data/'definition8_deployment_fp64/deployment.json'),'variants':{}}
    with torch.inference_mode():
        for variant in ['packed_core_reference_vision','reference_core_packed_vision']:
            initial='reference' if variant.startswith('packed_core') else 'packed'
            policy=load_policy(data/'definition8_deployment_fp64',cfg,initial)
            policy._model.eval().requires_grad_(False)
            torch.backends.cuda.matmul.allow_tf32=False
            for module in policy._model.modules():
                if isinstance(module,DefinitionLinear):
                    if initial=='reference':
                        module.set_implementation('packed')
                        del module.reference_codes
                    else:
                        module.set_implementation('reference')
                        module.register_buffer('reference_codes',unpack_codes(module.B))
            install_pi05_step_context(policy._model,fixed_loop=True)
            policy._sample_actions=policy._model.sample_actions
            values=[policy.infer(s,noise=reference['noise'])['actions'].copy() for s in samples]
            rows=[difference(v,r) for v,r in zip(values,reference['actions'])]
            result['variants'][variant]={'errors':rows,'max_nmse':max(x['nmse'] for x in rows)}
            np.savez(data/(variant+'_outputs.npz'),actions=np.stack(values))
            write(data/'definition8_error_sources.json',result)
            del policy
            gc.collect()
            torch.cuda.empty_cache()
    print('ERROR_SOURCE_DIAGNOSTIC',result,flush=True)


if __name__=='__main__':
    main()
