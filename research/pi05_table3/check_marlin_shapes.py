"""Investigate the failed large-M oracle check without changing its guard."""
import fcntl
import json
from pathlib import Path
import torch
from research.pi05_table3.baseline_runtime import BaselineLinear
from research.pi05_table3.measure import write


def main():
    guard=open('/tmp/holoq-vla-policy-timing.lock','a')
    fcntl.flock(guard,fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    data=Path(__file__).resolve().parents[2]/'results/pi05_table3_414_20261028'
    manifest=json.loads((data/'baseline_capture/manifest.json').read_text())
    item=next(x for x in manifest['layers'] if 'language_model' in x['name'])
    entry=torch.load(data/'baseline_build/gptq_marlin252/layers'/f"{item['index']:03d}.pt",map_location='cpu',mmap=True,weights_only=False)
    layer=BaselineLinear(entry['record'],item['name']).cuda()
    w=(entry['canonical_codes'].cuda().float()*entry['canonical_scales'].cuda().float().repeat_interleave(128,1)).half()
    rows=[]
    with torch.inference_mode():
        for m in [7,16,32,64,65,128,152,160,161,176,192,968,1024]:
            torch.manual_seed(m)
            x=torch.randn(m,item['K'],device='cuda',dtype=torch.bfloat16)
            expected=torch.nn.functional.linear(x.half(),w).to(layer.external_dtype)
            for mode in ['default','maxpar1','zero_output']:
                if mode=='default':
                    actual=layer(x)
                else:
                    out=torch.zeros(m,layer.n,device='cuda',dtype=torch.float16) if mode=='zero_output' else torch.empty(m,layer.n,device='cuda',dtype=torch.float16)
                    layer._kernel.mul(x.half(),layer.B,out,layer.s,layer.workspace,max_par=1 if mode=='maxpar1' else 16)
                    actual=out.to(layer.external_dtype)
                diff=(actual.float()-expected.float()).square().sum(1)
                norm=expected.float().square().sum(1)
                rowerr=(diff/norm.clamp_min(1e-30))
                rows.append({'m':m,'mode':mode,'nmse':float(diff.sum()/norm.sum()),'row_nmse':rowerr.cpu().tolist(),
                    'worst_row':int(rowerr.argmax()),'max_abs':float((actual.float()-expected.float()).abs().max()),
                    'nonzero_workspace_elements':int(torch.count_nonzero(layer.workspace))})
                print(m,mode,rows[-1]['nmse'],rows[-1]['worst_row'],flush=True)
            write(data/'marlin_shape_diagnostic.json',{'layer':item,'rows':rows,'note':'Diagnostic only. Initial guard1e-4 unchanged; failed baseline timings not treated as validated until resolved.'})


if __name__=='__main__':
    main()
