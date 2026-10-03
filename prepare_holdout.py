"""V2 recovery: freeze exact disjoint inputs before registration, no model forwards."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model import ModelConfig,generate_batch
from evaluate import make_donors,FAMILIES
from audit_holdout import encode

def forbidden_inputs(protocol):
    config=ModelConfig()
    forbidden=set()
    for seed,steps in protocol['training_audit_steps'].items():
        generator=torch.Generator().manual_seed(1000+int(seed))
        for _ in range(steps):
            tokens,_,_=generate_batch(protocol['training_batch_size'],generator,config)
            forbidden.update(map(int,encode(tokens)))
    generator=torch.Generator().manual_seed(protocol['training_validation_seed'])
    for _ in range(protocol['training_validation_batches']):
        tokens,_,_=generate_batch(protocol['training_validation_batch_size'],generator,config)
        forbidden.update(map(int,encode(tokens)))
    generator=torch.Generator().manual_seed(protocol['discovery']['input_seed'])
    donor_generator=torch.Generator().manual_seed(protocol['discovery']['input_seed']+10000000)
    for start in range(0,protocol['discovery']['n'],protocol['batch_size']):
        tokens,_,meta=generate_batch(min(protocol['batch_size'],protocol['discovery']['n']-start),generator,config)
        donors,_,_=make_donors(tokens,meta,donor_generator)
        forbidden.update(map(int,encode(tokens)))
        for donor,_,_ in donors.values():
            forbidden.update(map(int,encode(donor)))
    tokens,_,_=generate_batch(8,torch.Generator().manual_seed(47821),config)
    forbidden.update(map(int,encode(tokens)))
    return forbidden

def identity(values):
    return hashlib.sha256(np.array(sorted(values),dtype='<u8').tobytes()).hexdigest()

def audit_prepared(protocol):
    forbidden=forbidden_inputs(protocol)
    data=np.load(protocol['confirmatory']['input_file'])
    collisions={name:sum(int(x) in forbidden for x in encode(torch.from_numpy(data[name])))
                for name in ['tokens',*[f'donor_{f}' for f in FAMILIES],'donor_cross']}
    codes=encode(torch.from_numpy(data['tokens']))
    collisions['duplicate_recipients']=len(codes)-len(set(codes))
    metadata=json.loads(Path('outputs/holdout_v2.json').read_text())
    actual_hash=hashlib.sha256(Path(protocol['confirmatory']['input_file']).read_bytes()).hexdigest()
    return {'pass':not any(collisions.values()) and len(codes)==protocol['confirmatory']['n']
                    and actual_hash==metadata['input_file_sha256'] and identity(forbidden)==metadata['forbidden_set_sha256'],
            'collisions':collisions,'n':len(codes),'forbidden_distinct_inputs':len(forbidden),
            'forbidden_set_sha256':identity(forbidden),'input_file_sha256':actual_hash,
            'method':'Reconstruct exact forbidden set; check frozen recipient and five donor arrays before any model forward'}

def main():
    torch.set_num_threads(4)
    protocol=json.loads(Path('protocol.json').read_text())
    path=Path(protocol['confirmatory']['input_file'])
    if path.exists():
        raise RuntimeError('Never overwrite frozen input file')
    forbidden=forbidden_inputs(protocol)
    generator=torch.Generator().manual_seed(protocol['confirmatory']['input_seed'])
    donor_generator=torch.Generator().manual_seed(protocol['confirmatory']['input_seed']+10000000)
    config=ModelConfig()
    rows=[]; selected=set(); rejected=[]; candidates=0
    while len(rows)<protocol['confirmatory']['n']:
        tokens,original,meta=generate_batch(protocol['batch_size'],generator,config)
        donors,other,_=make_donors(tokens,meta,donor_generator)
        cross=donors['key'][0].clone(); cross[:,1::2]=donors['value'][0][:,1::2]
        batch={'tokens':tokens.numpy(),'original':original.numpy(),'query_pair':meta['query_pair'].numpy(),
               'other':other.numpy(),'alternative':meta['values'][torch.arange(len(tokens)),other].numpy(),
               **{'donor_'+f:donor.numpy() for f,(donor,_,_) in donors.items()},'donor_cross':cross.numpy()}
        codes={name:encode(torch.from_numpy(value)) for name,value in batch.items() if value.ndim==2}
        for i in range(len(tokens)):
            reasons=[name for name,code in codes.items() if int(code[i]) in forbidden]
            if int(codes['tokens'][i]) in selected:
                reasons.append('duplicate_recipient')
            if reasons:
                rejected.append({'candidate_index':candidates,'reasons':reasons})
            else:
                selected.add(int(codes['tokens'][i])); rows.append({k:v[i] for k,v in batch.items()})
            candidates+=1
            if len(rows)==protocol['confirmatory']['n']:
                break
    arrays={key:np.stack([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(path,**arrays)
    metadata={'revision':'v2; v1 stopped at overlap gate before any confirmation forward',
              'candidate_seed':protocol['confirmatory']['input_seed'],'donor_seed':protocol['confirmatory']['input_seed']+10000000,
              'candidate_batch_size':protocol['batch_size'],'candidates_considered':candidates,'accepted':len(rows),
              'rejected':rejected,'forbidden_distinct_inputs':len(forbidden),'forbidden_set_sha256':identity(forbidden),
              'input_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'policy':'Reject entire recipient and all donors on any exact forbidden-input match; reject duplicate accepted recipients; no model outputs used'}
    Path('outputs/holdout_v2.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    main()
