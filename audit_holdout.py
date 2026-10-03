"""Exact 45-bit input identity audit, no model forward passes."""
import json
from pathlib import Path
import numpy as np
import torch
from model import ModelConfig, generate_batch

def encode(tokens):
    array = tokens.numpy().astype(np.uint64)
    return (array << (np.arange(9, dtype=np.uint64)*5)).sum(axis=1)

def audit(protocol):
    if protocol['confirmatory'].get('input_file'):
        from prepare_holdout import audit_prepared
        return audit_prepared(protocol)
    from evaluate import make_donors
    config=ModelConfig()
    p=protocol['confirmatory']
    generator=torch.Generator().manual_seed(p['input_seed'])
    donor_generator=torch.Generator().manual_seed(p['input_seed']+10000000)
    candidates=set()
    recipients=[]
    for start in range(0,p['n'],protocol['batch_size']):
        tokens,_,meta=generate_batch(min(protocol['batch_size'],p['n']-start),generator,config)
        recipients.extend(map(int,encode(tokens)))
        donors,_,_=make_donors(tokens,meta,donor_generator)
        candidates.update(map(int,encode(tokens)))
        for donor,_,_ in donors.values():
            candidates.update(map(int,encode(donor)))
        cross=donors['key'][0].clone()
        cross[:,1::2]=donors['value'][0][:,1::2]
        candidates.update(map(int,encode(cross)))
    collisions={}
    for seed,steps in protocol['training_audit_steps'].items():
        gen=torch.Generator().manual_seed(1000+int(seed))
        count=0
        for _ in range(steps):
            tokens,_,_=generate_batch(protocol['training_batch_size'],gen,config)
            count+=sum(int(x) in candidates for x in encode(tokens))
        collisions[f'training_seed_{seed}']=count
    gen=torch.Generator().manual_seed(protocol['training_validation_seed'])
    validation=[]
    for _ in range(protocol['training_validation_batches']):
        tokens,_,_=generate_batch(protocol['training_validation_batch_size'],gen,config)
        validation.extend(map(int,encode(tokens)))
    collisions['training_validation']=len(candidates.intersection(validation))
    smoke,_,_=generate_batch(8,torch.Generator().manual_seed(47821),config)
    collisions['trained_model_smoke_inputs']=len(candidates.intersection(map(int,encode(smoke))))
    gen=torch.Generator().manual_seed(protocol['discovery']['input_seed'])
    donor_gen=torch.Generator().manual_seed(protocol['discovery']['input_seed']+10000000)
    discovery=set()
    for start in range(0,protocol['discovery']['n'],protocol['batch_size']):
        tokens,_,meta=generate_batch(min(protocol['batch_size'],protocol['discovery']['n']-start),gen,config)
        donors,_,_=make_donors(tokens,meta,donor_gen)
        discovery.update(map(int,encode(tokens)))
        for donor,_,_ in donors.values():
            discovery.update(map(int,encode(donor)))
    collisions['discovery_and_donors']=len(candidates.intersection(discovery))
    collisions['duplicate_confirmatory_recipients']=len(recipients)-len(set(recipients))
    result={'method':'exact injective base32 encoding of all9 tokens; includes all donor inputs',
            'distinct_confirmatory_inputs':len(candidates),'collisions':collisions,
            'pass':not any(collisions.values())}
    return result
