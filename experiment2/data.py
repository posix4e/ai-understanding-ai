"""Construct all E2 input splits without model evaluation, rejecting whole bundles."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import ModelConfig,generate_batch
from prepare_holdout import forbidden_inputs,identity
from audit_holdout import encode

ROOT=Path('outputs/experiment2')
SPLITS={'calibration':(5100001,512),'discovery':(5200001,1024),'confirmatory':(5300001,2048)}

def make_bundle(tokens,targets,meta,generator):
    config=ModelConfig()
    donor,_,_=generate_batch(len(tokens),generator,config)
    for i in range(len(tokens)):
        q=int(meta['query_pair'][i]);query=int(tokens[i,-1]); target=int(targets[i])+16
        keys=donor[i,:-1:2]
        found=(keys==query).nonzero().flatten()
        if len(found):
            j=int(found[0]); keys[q],keys[j]=keys[j].clone(),keys[q].clone()
        else:
            keys[q]=query
        donor[i,-1]=query
        if int(donor[i,2*q+1])==target:
            j=(q+1)%4
            donor[i,2*q+1],donor[i,2*j+1]=donor[i,2*j+1].clone(),donor[i,2*q+1].clone()
    matched=donor.clone()
    for i in range(len(tokens)):
        q=int(meta['query_pair'][i]); target=int(targets[i])+16
        values=matched[i,1::2]
        found=(values==target).nonzero().flatten()
        if len(found):
            j=int(found[0]);values[q],values[j]=values[j].clone(),values[q].clone()
        else:
            values[q]=target
    return {'tokens':tokens.numpy(),'targets':targets.numpy(),'query_pair':meta['query_pair'].numpy(),
            'donor_tokens':donor.numpy(),'matched_donor_tokens':matched.numpy(),
            'donor_targets':(donor[torch.arange(len(tokens)),2*meta['query_pair']+1]-16).numpy()}

def base_forbidden():
    protocol=json.loads(Path('protocol.json').read_text())
    forbidden=forbidden_inputs(protocol)
    # New model training uses unchanged E1 generator and fixed batch128.
    for seed in [3,4,5]:
        generator=torch.Generator().manual_seed(1000+seed)
        for _ in range(2000):
            tokens,_,_=generate_batch(128,generator)
            forbidden.update(map(int,encode(tokens)))
    # Include every published E1 confirmation input and donor, including aborted candidates.
    frozen=np.load('outputs/holdout_v2.npz')
    for name in frozen.files:
        if frozen[name].ndim==2:
            forbidden.update(map(int,encode(torch.from_numpy(frozen[name]))))
    from evaluate import make_donors
    generator=torch.Generator().manual_seed(3000001)
    donor_gen=torch.Generator().manual_seed(13000001)
    for _ in range(17):
        tokens,_,meta=generate_batch(128,generator)
        donors,_,_=make_donors(tokens,meta,donor_gen)
        cross=donors['key'][0].clone();cross[:,1::2]=donors['value'][0][:,1::2]
        for batch in [tokens,cross,*[x[0] for x in donors.values()]]:
            forbidden.update(map(int,encode(batch)))
    return forbidden

def main():
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True,exist_ok=True)
    forbidden=base_forbidden()
    metadata={'base_forbidden_count':len(forbidden),'base_forbidden_sha256':identity(forbidden),'splits':{}}
    for split,(seed,n) in SPLITS.items():
        path=ROOT/f'{split}_inputs.npz'
        if path.exists():raise RuntimeError('Refusing to replace '+str(path))
        generator=torch.Generator().manual_seed(seed)
        donor_gen=torch.Generator().manual_seed(seed+10000000)
        seen=set(); rows=[]; rejected=[]; candidate=0
        before=identity(forbidden)
        while len(rows)<n:
            tokens,targets,meta=generate_batch(128,generator)
            bundle=make_bundle(tokens,targets,meta,donor_gen)
            codes={key:encode(torch.from_numpy(bundle[key])) for key in ['tokens','donor_tokens','matched_donor_tokens']}
            for i in range(len(tokens)):
                reasons=[key for key,values in codes.items() if int(values[i]) in forbidden or int(values[i]) in seen]
                if len({int(v[i]) for v in codes.values()})!=3:reasons.append('within_bundle_duplicate')
                if reasons:rejected.append({'candidate':candidate,'reasons':reasons})
                else:
                    rows.append({key:value[i] for key,value in bundle.items()})
                    seen.update(int(v[i]) for v in codes.values())
                candidate+=1
                if len(rows)==n:break
        arrays={key:np.stack([row[key] for row in rows]) for key in rows[0]}
        np.savez_compressed(path,**arrays)
        metadata['splits'][split]={'seed':seed,'donor_seed':seed+10000000,'n':n,'candidates':candidate,
            'rejected':rejected,'forbidden_before_sha256':before,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        forbidden.update(seen)
    (ROOT/'input_audit.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
