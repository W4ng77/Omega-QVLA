"""Untimed full-deployment head allocation window and resident tensor audit."""
from collections import Counter,defaultdict
import dataclasses
import fcntl
from pathlib import Path
import numpy as np
import torch
from research.pi05_table3.measure import write,sha,memory
from research.pi05_table3.audit_definition_storage import phase


def main():
    guard=open('/tmp/holoq-vla-policy-timing.lock','a')
    fcntl.flock(guard,fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from openpi.training import config
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from research.pi05_table3.definition_runtime import load_policy,DefinitionLinear
    from research.pi05_table3.vision_packed import PackedVision
    root=Path(__file__).resolve().parents[2]
    data=root/'results/pi05_table3_414_20261028'
    cfg=config.get_config('pi05_libero')
    cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    deployment=data/'definition8_deployment_fp64'
    policy=load_policy(deployment,cfg,'packed')
    model=policy._model.eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False
    install_pi05_step_context(model,fixed_loop=True)
    policy._sample_actions=model.sample_actions
    sample=torch.load('/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt',map_location='cpu',weights_only=False)['samples'][10]
    noise=np.random.default_rng(42).standard_normal((cfg.model.action_horizon,cfg.model.action_dim)).astype(np.float32)
    ledger=[]
    seen=set()
    by_phase=defaultdict(Counter)
    for name,tensor in list(model.named_parameters())+list(model.named_buffers()):
        if tensor.numel()==0 or tensor.untyped_storage().data_ptr() in seen:
            continue
        seen.add(tensor.untyped_storage().data_ptr())
        row={'name':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'storage_bytes':tensor.untyped_storage().nbytes(),'phase':phase(name)}
        ledger.append(row)
        by_phase[row['phase']][row['dtype']]+=row['storage_bytes']
    selected=[]
    for name,module in model.named_modules():
        if isinstance(module,(DefinitionLinear,PackedVision)):
            assert module.B.dtype==torch.uint8 and module.B.numel()==module.in_features*module.out_features//2
            assert module.weight.numel()==0
            assert not hasattr(module,'reference_codes')
            selected.append({'name':name,'packed_bytes':module.B.numel(),'shape':[module.out_features,module.in_features]})
    assert len(selected)==414
    with torch.inference_mode():
        for _ in range(32):
            policy.infer(sample,noise=noise)
        torch.cuda.synchronize()
        head={}
        calls=[0]
        original=model.denoise_step
        def denoise(*args,**kwargs):
            if calls[0]==0:
                torch.cuda.synchronize()
                head['entry_allocated_bytes']=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            calls[0]+=1
            return original(*args,**kwargs)
        model.denoise_step=denoise
        policy.infer(sample,noise=noise)
        torch.cuda.synchronize()
        head.update(steps=calls[0],through_policy_completion=memory())
        head['incremental_peak_bytes']=head['through_policy_completion']['peak_allocated_bytes']-head['entry_allocated_bytes']
        assert calls[0]==8
    result={'deployment_manifest_sha256':sha(deployment/'deployment.json'),'source_sha256':sha(Path(__file__)),
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'selected_quantized_targets':selected,'resident_registered_tensor_storage_bytes_by_phase_dtype':dict(by_phase),
        'resident_tensor_ledger':ledger,'head_stage_memory':head,
        'note':'Untimed diagnostic, same complete414 deployment, eager after32warmup. Model tensors exclude CUDA context, unregistered scratch and Graph pools. Head increment starts after prefix computation, above already-resident model+prefix; not totalheadmemory. Unused expert vocabulary weight retained equally in all methods.'}
    write(data/'definition8_memory_inspection.json',result)
    print('MEMORY_INSPECTION',head,flush=True)


if __name__=='__main__':
    main()
