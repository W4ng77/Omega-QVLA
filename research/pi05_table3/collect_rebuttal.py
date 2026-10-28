"""Build reviewer tables from raw reports, preserving scope and missing evidence."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/pi05_table3_414_20261028'
DOC = ROOT / 'docs/pi05_table3_rebuttal_20261028'


def read(path):
    return json.loads(path.read_text())


def save_csv(name, rows):
    with (DOC/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    DOC.mkdir(exist_ok=True,parents=True)
    cases=[('HoloQ historical414','scope_controls/both/report.json','holoq_deployments/both/deployment.json','4/4','162vision+126language+126expert; historical grids'),
           ('HoloQ definition8 core + released vision RTN','definition8_measurement/report.json','definition8_deployment_fp64/deployment.json','4/4','162visionRTN+126language+126expert; signed15core'),
           ('Historical backbone-only','scope_controls/backbone/report.json','holoq_deployments/backbone/deployment.json','4/4','162vision+126language'),
           ('Historical head-only','scope_controls/head/report.json','holoq_deployments/head/deployment.json','4/4','126expert'),
           ('Definition8 backbone-only','definition8_scope_controls/backbone/report.json','definition8_scope_deployments/backbone/deployment.json','4/4','162visionRTN+126language'),
           ('Definition8 head-only','definition8_scope_controls/head/report.json','definition8_scope_deployments/head/deployment.json','4/4','126expert')]
    for steps,group in [(10,'controls'),(8,'controls8')]:
        for name,kind,bits,scope in [
            ('FP16 control','fp16','16/16','No quantized targets; sensitiveFP32 params retained'),
            ('BF16 source control','bf16','16/16','No quantized targets; sensitiveFP32 params retained'),
            ('GPTQ / Marlin','gptq_marlin252','4/16','126language+126expert; visionBF16; Wgroup128'),
            ('SmoothQuant-W4 extension / QServe','smoothquant_qserve414','4/8','162vision+126language+126expert; per-outputW4')]:
            artifact=None if kind in ['fp16','bf16'] else 'baseline_deployments/'+kind+'/deployment.json'
            cases.append((name,group+'/'+kind+'/report.json',artifact,bits,scope))
    rows=[]
    incomplete=[]
    for name,report_path,artifact,bits,scope in cases:
        path=DATA/report_path
        if not path.exists() or read(path).get('status')!='complete':
            incomplete.append(report_path)
            continue
        d=read(path)
        artifact_bytes=read(DATA/artifact)['model_artifact_bytes'] if artifact else 7233515040
        for mode,v in d['modes'].items():
            mem=v['inference_memory']
            row={'method':name,'mode':mode,'steps':d['steps'],'W_A_bits':bits,
                 'quantized_layers':d.get('quantized_layers',414 if '414' in name else ''),'actual_scope':scope,
                 'success_rate':'not measured for this deployment','mean_ms':v['mean_ms'],'p50_ms':v['p50_ms'],
                 'p95_ms':v['p95_ms'],'policy_calls_per_s':v['policy_calls_per_s'],'calls':v['count'],
                 'steady_allocated_bytes':v['stable_memory']['allocated_bytes'],
                 'steady_reserved_bytes':v['stable_memory']['reserved_bytes'],
                 'peak_allocated_bytes':mem['peak_allocated_bytes'],'peak_reserved_bytes':mem['peak_reserved_bytes'],
                 'model_artifact_bytes':artifact_bytes,
                 'storage_scope':'Tensor payload only for FP16/BF16; serialized self-contained model files+metadata+assets for packed rows; shared tokenizer separate',
                 'report':str(path.relative_to(ROOT))}
            assert row['calls']==100, row
            rows.append(row)
    save_csv('main_results.csv',rows)
    ratios=[]
    for h in rows:
        if h['method'] not in ['HoloQ historical414','HoloQ definition8 core + released vision RTN']:
            continue
        for b in rows:
            if b['method'] in ['GPTQ / Marlin','SmoothQuant-W4 extension / QServe'] and b['steps']==h['steps'] and b['mode']==h['mode']:
                ratios.append({'holoq':h['method'],'baseline':b['method'],'steps':h['steps'],'mode':h['mode'],
                    'baseline_ms_div_holoq_ms':b['mean_ms']/h['mean_ms'],
                    'holoq_ms_div_baseline_ms':h['mean_ms']/b['mean_ms'],
                    'precision_scope_caveat':'Deployment comparison; differing A precision, GPTQ scope, quantizer grids and calibration teacher steps are disclosed; no success equivalence claim'})
    if ratios:
        save_csv('speed_ratios.csv',ratios)
    stages=[]
    for variant,path in [('historical414','scope_controls/both/verified_eager_attribution.json'),('definition8','definition8_profile/verified_eager_attribution.json')]:
        if (DATA/path).exists():
            d=read(DATA/path)
            assert not d['missing_launch_correlations']
            for k,v in d['groups'].items():
                stages.append({'variant':variant,'phase_category':k,**v,'source':str((DATA/path).relative_to(ROOT)),
                    'note':'Cumulative rawGPU activities; not formal wall time; use interval union when overlap exists'})
    if stages:
        save_csv('profile_breakdown.csv',stages)
    sections=['Pi0.5 / Table3 deployment evidence','==================================','',
        'This is an implementation/performance packet, not a new task-success experiment. HoloQ-VLA and Omega-QVLA are the same method.',
        'The submitted-method source is the user-verified eight-page description. The separately available arXiv appendix is not used to fill missing submission details.','',
        'Scope: 162 vision + 126 language + 126 expert = 414 selected projections. Full module names and shapes are in the captured manifests and per-deployment reports.',
        'The released-grid historical variant and the definition-aligned8-step core with released visionRTN remain separate configurations. See scope_and_claims.txt and method_alignment.csv.','',
        'Complete-policy measurement','---------------------------','',
        'RTX4080SUPER16GB, batch1, horizon10, 224x224 supplied cameras, identical fixed observations/noise,32warmup and100calls per mode. Eager and full-sampler CUDA Graph are available to every measured method; compile is disabled. Includes preprocessing,H2D,vision/language/all Euler steps,transforms,quantization/packing,GEMM,D2H and output transforms. Excludes RPC,simulator,loading and cold graph capture.',
        'Allocated/reserved statistics refer to warmed inference, reset immediately before formal calls. Reserved memory includes allocator history; it is not a minimum required VRAM claim. Model static files and shared tokenizer bytes are separate.','',
        '.. csv-table:: Matched whole-policy timings and memory',
        '   :file: main_results.csv','   :header-rows: 1','',
        'The speed ratios are computed only within matching step counts and execution modes. A ratio baseline_ms/HoloQ_ms below1 means HoloQ is slower; see speed_ratios.csv. No inherent W4A4 speed advantage or superiority to unmeasured VLA-specific implementations is claimed.','',
        'Correctness and execution','-------------------------','']
    p=DATA/'definition8_measurement/report.json'
    if p.exists():
        d=read(p)
        if d.get('validation_passed'):
            sections.extend([f"Definition8 local same-input oracle checks: {d['local_checks']['count']}, maximum NMSE {d['local_checks']['max_nmse']:.9g}. Ten-observation final action maximum NMSE {d['max_action_nmse']:.9g}, max absolute discrepancy {max(x['max_abs'] for x in d['reference_packed_action']):.9g}. These are backend arithmetic comparisons on the SAME frozen grid, not quantization-to-FP16 errors or success rates.",
                'Graph/eager actions agree in the recorded fixed-input checks. The engineering action bound0.01 was fixed before timing; passing it does not prove equal task success.',''])
    sections.extend(['Language: packed signed15 W4/A4, existing CUTLASS nativeS4 products/INT32 accumulation andFP32 scale epilogue. Expert: packed signed15 W4/A4, Sa[t,k] insideK, on-chip FP16 operands/FP32 accumulation. Vision: released asymmetricRTN packedW4/A4, BF16MMA/FP32 accumulation. Outside selected targets, floating attention,normalization,embeddings,auxiliary projections and image patch computation remain. No dense selected-weight cache or floating F.linear fallback is used by the timed packed targets.','',
        'Correct expert equation: Y[m,n] = Sw[n] * sum_k(QX[m,k] * Sa[layer,t,k] * QW[n,k]) + bias[n]. One W4 matrix per layer; no step_weight banks or weight requantization.','',
        'Storage, attribution and limitations','------------------------------------','',
        'Table3 displayed5.41GB matches5,408,612,352bytes of selected414 weights. The full source checkpoint has7,233,515,040tensor bytes. Required metadata must be included; see complete_storage_audit.json and definition_storage_audit.json in the evidence archive. The current deployment does not establish the submitted1.39GB/74.2% number when all necessary metadata is included. Static target saving, complete model storage and runtime memory are different quantities.','',
        'Profiling uses raw GPU activity durations linked by CUDA correlation IDs to enclosing CPU scopes. Rejected FunctionEvent association sums are preserved but not cited. GPU overlap is handled by interval union. Nsight confirms the actual historical CUDA Graph deployment; CUDA API memcpy host time contains GPU completion waits and must not be added to GPU kernel work. See profile_breakdown.csv and nsight_verified_summary.json.','',
        'Calibration is10 initial Object snapshots/all tokens, not the submission10 trajectories. The definition8 builder retains signed15,full-K per-outputSw,input-onlyUDH64,8-step per-channelSa,DiTRTN and GPTQprocessing128/damp.01. Zigzag mapping,Dseed andq99.9 are explicitly disclosed choices where the body is incomplete. FirstFP32 factorization failure and separateFP64Gram/factorization retry are preserved. Baseline static weights were calibrated with10-step teacher inputs;8-step deployment timing does not make this an equal-calibration accuracy comparison.','',
        'No new LIBERO success rates or real-rollout timings are established here. Do not attach submitted success rates to these newly measured deployment rows. No historicalH100 native kernel trace was recovered. Optimized QuantVLA/QVLA/ActQuant superiority is not established. Those claims must be omitted or narrowed, not filled from unrelated local ports.','',
        'Reviewer replies and reproducibility','------------------------------------','',
        'READY_TO_PASTE.txt gives the short combined response. REVIEWER_REPLIES.txt gives topic-specific replies with evidence references. reviewer_evidence.csv lists answer status and remaining unsupported claims. Reproduction commands, package/source versions, pack hashes and raw records accompany this packet.',''])
    if incomplete:
        sections.extend(['Incomplete cases at collection time','-----------------------------------','',*incomplete,''])
    (DOC/'REPORT.rst').write_text('\n'.join(sections)+'\n')
    write={'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           'incomplete_reports':incomplete,'result_rows':len(rows),
           'report_sha256':{str((DATA/p).relative_to(ROOT)):hashlib.sha256((DATA/p).read_bytes()).hexdigest() for _,p,_,_,_ in cases if (DATA/p).exists()}}
    (DOC/'collection_manifest.json').write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'incomplete':incomplete}))


if __name__=='__main__':
    main()
