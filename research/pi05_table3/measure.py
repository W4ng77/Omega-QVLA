"""Complete Pi0.5 policy evidence for exactly Table3's 414 weight targets.

Historical frozen grid, not a claimed submission-method/accuracy reproduction.
252 LM/expert +162 vision use packed W4/A4 and floating MMA. No dense fallback.
Reference mode is correctness-only. It is never an optimized speed baseline.
"""
import argparse
from collections import Counter
import dataclasses
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def memory():
    return {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def sample_gpu(stop, path):
    with path.open('a', buffering=1) as file:
        while not stop.is_set():
            row = {'time': time.time()}
            for key, query in [('gpus', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm'),
                               ('processes', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name')]:
                result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                row[key] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
            file.write(json.dumps(row) + '\n')
            stop.wait(5)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('/home/xinyu/models/pi05_libero_pytorch'))
    p.add_argument('--packed-root', type=Path, default=Path('/home/xinyu/models/omega-qvla-pi05-414-w4a4-int4'))
    p.add_argument('--suite', default='object', choices=['object','goal','spatial','long'])
    p.add_argument('--observations', type=Path, default=Path('/home/xinyu/omegaqvla/deploy_extras/pi05_calib_obs.pt'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=32)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    guard = open('/tmp/holoq-vla-policy-timing.lock', 'a')
    fcntl.flock(guard, fcntl.LOCK_EX)
    a.output.mkdir(parents=True, exist_ok=False)
    for key in list(os.environ):
        if key.startswith(('GR00T_', 'OPENPI_GPTQ', 'QVLA_ACT')):
            os.environ.pop(key)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    from openpi.policies import policy_config
    from openpi.training import config
    from gr00t.quantization.gptq_layers import GptqConfig, GptqLinear, wrap_gptq
    from gr00t.quantization.int4_storage import unpack_int4
    from gr00t.quantization.rtn_layers import RTNLinear
    from gr00t.quantization.packed_w4a4 import enable_packed_w4a4
    from gr00t.quantization.pi05_step_context import install_pi05_step_context
    from gr00t.quantization.pi05_cuda_graph import Pi05CudaGraphSampler
    from gr00t.quantization.dit_step_context import get_current_dit_step
    from research.pi05_table3.vision_packed import PackedVision

    cfg = config.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    path = a.packed_root / f'pi05_{a.suite}/quantized.pt'
    records = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    vision = torch.load(a.packed_root / 'vision_int4.pt', map_location='cpu', mmap=True, weights_only=False)
    manifest = json.loads((a.packed_root / 'manifest.json').read_text())
    expected = set(vision) | set(manifest['lm_shapes'])
    assert len(expected) == 414 and len(vision) == 162
    lm_names = [n for n in records if n != '__meta__']
    assert len(lm_names) == 252 and {n+'.weight' for n in lm_names} == set(manifest['lm_shapes'])
    step_counts = {int(records[n]['act_scale_table'].shape[0]) for n in lm_names if 'gemma_expert' in n}
    assert len(step_counts) == 1
    steps = next(iter(step_counts))
    report = {'status': 'building', 'suite': a.suite, 'scope': {'vision': 162, 'language': 126, 'expert': 126, 'total': 414},
              'source_pack': str(path), 'source_pack_sha256': sha(path), 'source_vision_sha256': sha(a.packed_root/'vision_int4.pt'),
              'checkpoint': str(a.checkpoint), 'checkpoint_sha256': sha(a.checkpoint/'model.safetensors'),
              'observations_sha256': sha(a.observations), 'denoising_steps': steps, 'submission_steps': 8,
              'step_note': 'Match ALL stored scale rows. Existing historical expert pack has10 rows; do not label as paper8-step reproduction.',
              'method_note': 'Historical Table3-scope pack: static channel language scales, damping.05 and output rotations; not automatically the submission-defined method.',
              'execution': 'REAL nibble-packed A4/W4; decode on chip toBF16 MMA,FP32 accumulation. No native INT4xINT4 claim, no full floating weight cache or fallback.',
              'source_files_sha256': {str(f.relative_to(ROOT)): sha(f) for folder in ['research/pi05_table3','gr00t/quantization'] for f in (ROOT/folder).glob('*.py')},
              'openpi_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd='/home/xinyu/openpi',text=True).strip(),
              'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'command': __import__('sys').argv, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'started': time.time(), 'modes': {}}
    write(a.output/'report.json', report)
    policy = policy_config.create_trained_policy(cfg, a.checkpoint, pytorch_device='cuda', sample_kwargs={'num_steps': steps})
    model = policy._model
    model.eval().requires_grad_(False)
    modules = dict(model.named_modules())
    assert all(isinstance(modules[n[:-7]], torch.nn.Linear) for n in expected)
    for weight_name, record in vision.items():
        name = weight_name[:-7]
        source = modules[name]
        source.weight.data.copy_(unpack_int4(record).to(device='cuda',dtype=source.weight.dtype))
        rtn = RTNLinear(source,name=name,group_size=record['group_size'],wbits=4,abits=4,prequantized=True)
        parent, attr = name.rsplit('.',1)
        setattr(modules[parent], attr, rtn)
    del modules, source, rtn
    gcfg = GptqConfig(enabled=True,path=str(path),act_bits=4,weight_bits=4,missing='error')
    assert wrap_gptq(model,lm_names,gcfg) == 252
    model.to('cuda')
    install_pi05_step_context(model, fixed_loop=True)
    policy._sample_actions = model.sample_actions
    all_data = torch.load(a.observations,map_location='cpu',weights_only=False)['samples']
    start = {'spatial':0,'object':10,'goal':20,'long':30}[a.suite]
    samples = all_data[start:start+10]
    noise = np.random.default_rng(42).standard_normal((cfg.model.action_horizon,cfg.model.action_dim)).astype(np.float32)
    report.update(batch=1,action_horizon=cfg.model.action_horizon,action_dim=cfg.model.action_dim,observation_indices=list(range(start,start+10)),warmup=a.warmup,reference_role='Correctness only. NEVER use its runtime as an optimized baseline.')

    def coverage():
        counts, shapes, handles = {}, {}, []
        for name, module in model.named_modules():
            if isinstance(module,(GptqLinear,RTNLinear,PackedVision)):
                counts[name] = Counter()
                shapes[name] = set()
                def hook(m, inputs, name=name):
                    step = get_current_dit_step()
                    counts[name]['prefix' if step is None else str(step)] += 1
                    shapes[name].add((tuple(inputs[0].shape),str(inputs[0].dtype)))
                handles.append(module.register_forward_pre_hook(hook))
        try:
            policy.infer(samples[0],noise=noise)
            torch.cuda.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        assert {name+'.weight' for name in counts} == expected and len(counts) == 414
        for name,counter in counts.items():
            if 'gemma_expert' in name:
                assert dict(counter) == {str(i):1 for i in range(steps)}, (name,counter)
            else:
                assert set(counter) == {'prefix'} and counter['prefix'] > 0, (name,counter)
        return {'layer_counts': {n:dict(c) for n,c in counts.items()},'shapes':{n:sorted(s) for n,s in shapes.items()},'distinct_active_layers':len(counts)}

    with torch.inference_mode():
        report['reference_coverage'] = coverage()
        reference = [policy.infer(obs,noise=noise)['actions'].copy() for obs in samples]
        report['status'] = 'reference_414_validated'
        write(a.output/'report.json', report)
        print('REFERENCE_414_COMPLETE',flush=True)
        report['packed_lm_setup'] = enable_packed_w4a4(model,records,version=2)
        modules = dict(model.named_modules())
        for weight_name,record in vision.items():
            name = weight_name[:-7]
            parent,attr = name.rsplit('.',1)
            setattr(modules[parent],attr,PackedVision(modules[name],record,name))
        del modules
        gc.collect()
        torch.cuda.empty_cache()
        report['packed_coverage'] = coverage()
        report['status'] = 'packed_414_active'
        write(a.output/'report.json', report)
        differences = []
        for obs,ref in zip(samples,reference):
            output = policy.infer(obs,noise=noise)['actions']
            diff = output.astype(float)-ref.astype(float)
            assert np.isfinite(output).all()
            differences.append({'nmse':float(np.square(diff).sum()/max(np.square(ref.astype(float)).sum(),1e-30)), 'max_abs':float(np.abs(diff).max())})
        report['reference_packed_action_errors'] = differences
        report['correctness_gate'] = {'limit_action_nmse':.01,'max_action_nmse':max(d['nmse'] for d in differences)}
        report['correctness_gate']['passed'] = report['correctness_gate']['max_action_nmse'] <= .01
        write(a.output/'report.json',report)
        assert report['correctness_gate']['passed'], 'Preserve failed check; no timing before diagnosis'
        assert all(m._weight_q.numel()==0 and m._w_promoted_cache is None for m in model.modules() if isinstance(m,GptqLinear))
        assert len([m for m in model.modules() if isinstance(m,PackedVision)]) == 162
        print('PACKED_414_VALIDATED',report['correctness_gate'],flush=True)
        stop = threading.Event()
        sampler = threading.Thread(target=sample_gpu,args=(stop,a.output/'gpu_samples.jsonl'),daemon=True)
        sampler.start()
        eager = []
        try:
            for mode in ['eager','sample_graph']:
                if mode == 'sample_graph':
                    policy._sample_actions = Pi05CudaGraphSampler(model.sample_actions)
                for i in range(a.warmup):
                    policy.infer(samples[i%10],noise=noise)
                torch.cuda.synchronize()
                if mode == 'sample_graph':
                    checks = []
                    for i,obs in enumerate(samples):
                        actual = policy.infer(obs,noise=noise)['actions']
                        checks.append(float(np.max(np.abs(actual-eager[i]))))
                    assert max(checks) <= 1e-6, checks
                    report['graph_eager_action_max_abs'] = checks
                else:
                    eager = [policy.infer(obs,noise=noise)['actions'].copy() for obs in samples]
                stable = memory()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                with (a.output/(mode+'_calls.jsonl')).open('w') as file:
                    for repeat in range(a.repeats):
                        for i,obs in enumerate(samples):
                            torch.cuda.synchronize()
                            start_time = time.perf_counter()
                            output = policy.infer(obs,noise=noise)
                            torch.cuda.synchronize()
                            elapsed = (time.perf_counter()-start_time)*1000
                            assert np.isfinite(output['actions']).all()
                            timings.append(elapsed)
                            file.write(json.dumps({'repeat':repeat,'observation_index':start+i,'ms':elapsed})+'\n')
                report['modes'][mode] = {'calls':len(timings),'mean_ms':float(np.mean(timings)),'p50_ms':float(np.percentile(timings,50)), 'p95_ms':float(np.percentile(timings,95)), 'policy_calls_per_s':1000/float(np.mean(timings)),'stable_memory_after_warmup':stable,'inference_memory':memory(), 'boundary':'synchronize;perf_counter;COMPLETE policy.infer; synchronize;perf_counter; includesCPU transforms,H2D,vision162,language126,all expert126 x10steps,rotations,A4 packing,packed GEMM,scales,D2H and output transforms. ExcludesRPC/simulator/model loading/cold graph capture.'}
                write(a.output/'report.json',report)
                print('TIMED_414',mode,report['modes'][mode],flush=True)
            if a.profile:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
                    for _ in range(5):
                        policy.infer(samples[0],noise=noise)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(a.output/'policy_trace.json'))
                (a.output/'top_self_cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40)+'\n')
                events = Counter(e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
                write(a.output/'cuda_kernel_counts.json',dict(events))
        finally:
            stop.set()
            sampler.join(timeout=25)
        report.update(status='complete',finished=time.time(),task_success='NOT MEASURED: no new LIBERO success-rate claim')
        write(a.output/'report.json',report)


if __name__ == '__main__':
    main()
