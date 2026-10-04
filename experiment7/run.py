"""E7 registered-grid runner; preserve complete raw outcomes and validity checks."""
import argparse,hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from experiment6.data import encode_case,answer_ids,VALUES
from experiment6.model_adapter import GPT2Adapter
from experiment7.model_adapter import GPT2DiagnosticAdapter,ResidualPatch
from experiment7.conditions import CONDITIONS,masks,layer_masks

ROOT=Path(__file__).resolve().parents[1]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_hash(items):
    h=hashlib.sha256()
    for name,x in items:
        h.update(name.encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def main():
    a=argparse.ArgumentParser();a.add_argument('--registration',required=True)
    a.add_argument('--manifest',default='outputs/experiment7/discovery_manifest.json')
    a.add_argument('--root',default='outputs/experiment7/discovery');args=a.parse_args()
    mp=ROOT/args.manifest;mb=mp.read_bytes();m=json.loads(mb)
    assert mb==subprocess.check_output(['git','show',f'{args.registration}:{args.manifest}'],cwd=ROOT)
    for name,h in m['files'].items():
        b=(ROOT/name).read_bytes()
        assert hashlib.sha256(b).hexdigest()==h,name
        assert b==subprocess.check_output(['git','show',f'{args.registration}:{name}'],cwd=ROOT),name
    for name,r in json.loads((ROOT/'outputs/experiment6/model_manifest.json').read_text())['files'].items():
        assert digest(ROOT/'work/models/gpt2'/name)==r['sha256'],name
    out=ROOT/args.root
    if out.exists() and any(out.iterdir()): raise RuntimeError('refuse to overwrite or restart nonempty run')
    out.mkdir(parents=True,exist_ok=True);(out/'batches').mkdir()
    cases=json.loads((ROOT/m['inputs_path']).read_text()); assert len(cases)==m['n']
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'work/models/gpt2',local_files_only=True)
    enc=[encode_case(tokenizer,c,'one_demo') for c in cases];e=enc[0]
    assert all(x.chunks==e.chunks and x.prefix==e.prefix and x.suffix==e.suffix for x in enc)
    ids=torch.stack([x.ids for x in enc]);p=e.permutation(True);inv=torch.argsort(p)
    avail=masks(e);candidates=torch.tensor(answer_ids(tokenizer));n=len(cases)
    regions={'prefix':e.prefix,'keys':[t for c in e.chunks[::2] for t in c],
             'values':[t for c in e.chunks[1::2] for t in c],'query':e.suffix}
    legacy=GPT2Adapter.from_local(ROOT/'work/models/gpt2');legacy.model.double()
    adapter=GPT2DiagnosticAdapter(legacy.model)
    assert all(x.dtype==torch.float64 for x in adapter.model.parameters())
    before=tensor_hash(adapter.model.named_parameters());buffers=tensor_hash(adapter.model.named_buffers())
    assert before=='9727c8b37eeaef2943a17936a984ee4688ff3566e575fe6486fbcdf8722779b6'
    start=time.perf_counter()
    write(out/'started.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'registration':args.registration,
         'manifest':args.manifest,'phase':m['phase'],'n':n,'conditions':CONDITIONS,'dtype':'float64',
         'parameters_sha256':before,'buffers_sha256':buffers,'threads':torch.get_num_threads(),
         'model_forwards_per_batch':len(CONDITIONS)+2})
    raw={c['id']:{'candidate_probs':[],'candidate_logits':[],'top_token_ids':[],'residual_mse':[]} for c in CONDITIONS}
    errors={k:0. for k in ['legacy_native_logits','legacy_guard_logits','noop_probability',
        'guard_value_state','guard_query_state','patched_full_first_block',
        'transport_all_logits','keypatch_transport_logits']}
    ledger=[]
    for begin in range(0,n,8):
        batch=ids[begin:begin+8];chunk={};native=None;grouped_probs=None
        for c in CONDITIONS:
            name=c['id'];perm=p if c['grouped'] else torch.arange(len(p))
            patches={}
            if c['patch']:
                patches={0:ResidualPatch(inv[regions['keys']],native.post_block[0][:,regions['keys']].clone())}
            result=adapter(batch[:,perm],position_ids=perm.expand(len(batch),-1),
                layer_masks=layer_masks(c,avail),allow_future=c['mask'] in ('add','transport','transport_all'),
                residual_patches=patches,capture_layers='all')
            assert result.used_future_edges==(c['mask'] in ('add','transport','transport_all'))
            prob=result.logits.softmax(-1)
            assert bool(torch.isfinite(prob).all()) and torch.allclose(prob.sum(-1),torch.ones(len(batch),dtype=torch.float64),atol=1e-6,rtol=0)
            if name=='native':
                native=result
                check=legacy(batch).logits
                errors['legacy_native_logits']=max(errors['legacy_native_logits'],float((result.logits-check).abs().max()))
            if name=='grouped': grouped_probs=prob.clone()
            if name=='grouped_noop': errors['noop_probability']=max(errors['noop_probability'],float((prob-grouped_probs).abs().max()))
            state=result.post_block[0][:,torch.argsort(perm)]
            if c['guard']:
                for region,key in [('values','guard_value_state'),('query','guard_query_state')]:
                    errors[key]=max(errors[key],float((state[:,regions[region]]-native.post_block[0][:,regions[region]]).abs().max()))
            if c['patch']: errors['patched_full_first_block']=max(errors['patched_full_first_block'],float((state-native.post_block[0]).abs().max()))
            if name=='guard_physical':
                check=legacy(batch[:,p],position_ids=p.expand(len(batch),-1),layer_masks={0:avail['guard']}).logits
                errors['legacy_guard_logits']=max(errors['legacy_guard_logits'],float((result.logits-check).abs().max()))
            if name in ('transport_all','keypatch_transport'):
                errors[name+'_logits']=max(errors[name+'_logits'],float((result.logits-native.logits).abs().max()))
                assert torch.allclose(prob,native.logits.softmax(-1),atol=1e-6,rtol=0),name
            mse=[]
            for layer in range(12):
                restored=result.post_block[layer][:,torch.argsort(perm)]
                assert bool(torch.isfinite(restored).all())
                mse.append(torch.stack([(restored[:,rows]-native.post_block[layer][:,rows]).square().mean((1,2)) for rows in regions.values()],1))
            vals={'candidate_probs':prob[:,candidates].numpy().copy(),'candidate_logits':result.logits[:,candidates].numpy().copy(),'top_token_ids':result.logits.argmax(-1).numpy().copy(),
                  'residual_mse':torch.stack(mse,1).numpy().copy()}
            for field,value in vals.items():raw[name][field].append(value);chunk[name+'__'+field]=value
        chunk['case_indices']=np.arange(begin,min(n,begin+8))
        bp=out/'batches'/f'batch_{begin:04d}.npz';np.savez_compressed(bp,**chunk)
        ledger.append({'file':str(bp.relative_to(out)),'sha256':digest(bp),'begin':begin,'end':min(n,begin+8)})
        write(out/'ledger.json',ledger)
        if begin%32==0: print(json.dumps({'completed_cases':begin+len(batch),'n':n,'seconds':time.perf_counter()-start}),flush=True)
    targets=np.array([VALUES.index(c['values'][c['query_pair']]) for c in cases])
    for name,fields in raw.items():
        np.savez_compressed(out/(name+'.npz'),**{k:np.concatenate(v) for k,v in fields.items()},
            target_indices=targets,target_token_ids=candidates.numpy()[targets],query_pair=np.array([c['query_pair'] for c in cases]))
    after=tensor_hash(adapter.model.named_parameters());after_buffers=tensor_hash(adapter.model.named_buffers())
    checks={k:v<=(1e-6 if k=='noop_probability' else 1e-5) for k,v in errors.items()}
    checks.update(parameters_unchanged=before==after,buffers_unchanged=buffers==after_buffers,complete_finite_outputs=True)
    validity={'all_pass':all(checks.values()),'checks':checks,'max_errors':errors,'parameters_before':before,
              'parameters_after':after,'buffers_before':buffers,'buffers_after':after_buffers}
    write(out/'validity.json',validity)
    write(out/'finished.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start,
          'validity_pass':validity['all_pass'],'files':{p.name:digest(p) for p in sorted(out.glob('*.npz'))}})
    print(json.dumps({'finished':True,'validity':validity}),flush=True)
if __name__=='__main__': main()
