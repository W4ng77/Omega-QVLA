"""All metadata, complete policy storage, and complete action-head scope."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open
from research.pi05_table3.measure import write,sha
from research.pi05_table3.audit_storage import size


def phase(name):
    if 'gemma_expert' in name or name.startswith(('action_','time_mlp','state_proj')):
        return 'action_head'
    if 'vision_tower' in name or 'multi_modal_projector' in name:
        return 'vision'
    if 'paligemma' in name:
        return 'language'
    return 'other'


def main():
    root=Path(__file__).resolve().parents[2]
    data=root/'results/pi05_table3_414_20261028'
    source=Path('/home/xinyu/models/pi05_libero_pytorch')
    deployment=data/'definition8_deployment_fp64'
    core=torch.load(deployment/'packed_core.pt',map_location='cpu',mmap=True,weights_only=False)['layers']
    vision=torch.load(deployment/'vision_int4.pt',map_location='cpu',mmap=True,weights_only=False)
    selected={n+'.weight' for n in core}|set(vision)
    assert len(selected)==414
    raw=defaultdict(lambda:Counter())
    excluded=[]
    dtypes={'BF16':2,'F16':2,'F32':4,'F64':8,'I64':8,'I32':4,'BOOL':1}
    with safe_open(source/'model.safetensors',framework='pt',device='cpu') as file:
        for key in file.keys():
            s=file.get_slice(key)
            shape=s.get_shape()
            nbytes=math.prod(shape)*dtypes[s.get_dtype()]
            bucket=raw[phase(key)]
            bucket['all_original_tensor_bytes']+=nbytes
            bucket['selected_weight_bytes' if key in selected else 'remaining_original_tensor_bytes']+=nbytes
            if key not in selected:
                excluded.append({'name':key,'shape':shape,'dtype':s.get_dtype(),'bytes':nbytes,'phase':phase(key)})
    packed=defaultdict(lambda:Counter())
    for name,record in core.items():
        for field,value in record.items():
            packed[phase(name)][field]+=size(value)
    for name,record in vision.items():
        for field,value in record.items():
            packed[phase(name)][field]+=size(value)
    target_raw=sum(v['selected_weight_bytes'] for v in raw.values())
    target_packed=sum(sum(v.values()) for v in packed.values())
    original=sum(v['all_original_tensor_bytes'] for v in raw.values())
    nq=sum(v['remaining_original_tensor_bytes'] for v in raw.values())
    tok=Path('/home/xinyu/.cache/openpi/big_vision/paligemma_tokenizer.model')
    result={'source_checkpoint_sha256':sha(source/'model.safetensors'),'packed_core_sha256':sha(deployment/'packed_core.pt'),
        'deployment_manifest_sha256':sha(deployment/'deployment.json'),'raw_by_phase':dict(raw),
        'packed_metadata_by_phase':dict(packed),'unquantized_tensors':excluded,
        'original414_target_bytes':target_raw,'packed414_target_including_metadata_bytes':target_packed,
        'target_saving_fraction':1-target_packed/target_raw,'original_complete_tensor_bytes':original,
        'complete_deployed_tensor_bytes':target_packed+nq,'complete_tensor_saving_fraction':1-(target_packed+nq)/original,
        'serialized_model_artifact_bytes':json.loads((deployment/'deployment.json').read_text())['model_artifact_bytes'],
        'shared_tokenizer':{'path':str(tok),'bytes':tok.stat().st_size,'sha256':sha(tok)},
        'source_model_files_bytes':sum(f.stat().st_size for f in [source/'model.safetensors',source/'config.json',*list((source/'assets').rglob('*.json'))]),
        'head_scope':'All gemma_expert parameters plus action_in/out,time_mlp and state_proj if present; includes auxiliary/norm projections outside126 quantized expert weights. Peak head-stage incremental allocations are measured separately; serialized tensor bytes are not VRAM.',
        'step_weight_banks':1,'scope_note':'414 selected projections, not every parameter or operation. Vision uses releasedRTN, core follows disclosed submitted-definition reconstruction. Static bytes do not prove runtime savings.'}
    write(data/'definition_storage_audit.json',result)
    print('DEFINITION_STORAGE',target_raw,target_packed,target_packed+nq,result['serialized_model_artifact_bytes'])


if __name__=='__main__':
    main()
