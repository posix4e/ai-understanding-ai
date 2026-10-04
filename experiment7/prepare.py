"""Prepare excluded discovery inputs and freeze source bytes; no model inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
from transformers import AutoTokenizer
from experiment6.data import generate_cases,identity,encode_case
from experiment7.conditions import CONDITIONS,masks

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiment7'

def main():
    exclusions=[]
    for f in ('calibration_inputs.json','confirmation_inputs.json'):
        exclusions+=json.loads((ROOT/'outputs/experiment6'/f).read_text())
    cases=generate_cases(128,10700001,exclusions)
    p=OUT/'discovery_inputs.json'
    assert not p.exists(),'never overwrite prepared inputs'
    p.write_text(json.dumps(cases,indent=2)+'\n')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'work/models/gpt2',local_files_only=True)
    encoded=[encode_case(tokenizer,c,'one_demo') for c in cases]
    for e in encoded: masks(e)
    assert len({tuple(map(tuple,e.chunks)) for e in encoded})==1
    assert len({identity(c) for c in cases})==128
    assert not ({identity(c) for c in cases}&{identity(c) for c in exclusions})
    audit={'n':128,'seed':10700001,'unique_association_maps':128,'excluded_prior_maps':len(exclusions),
           'overlap':0,'query_counts':[sum(c['query_pair']==q for c in cases) for q in range(4)],
           'tokens':len(encoded[0].ids),'conditions':CONDITIONS,'pretrained_E7_forwards':0}
    (OUT/'input_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    files=list((ROOT/'experiment7').glob('*.py'))+list((ROOT/'experiment7').glob('*.md'))
    files+=list((ROOT/'experiment6').glob('*.py'))
    files += [p,OUT/'input_audit.json',OUT/'NOVELTY_REVIEW.md',ROOT/'outputs/experiment6/model_manifest.json',
        ROOT/'outputs/experiment6/environment.json',ROOT/'experiment6/requirements.txt',ROOT/'outputs/experiment6/calibration_inputs.json',
        ROOT/'outputs/experiment6/confirmation_inputs.json']
    manifest={'phase':'exploratory discovery','inputs_path':'outputs/experiment7/discovery_inputs.json','n':128,'prepared_utc':datetime.now(timezone.utc).isoformat(),
        'pretrained_E7_forwards':0,'files':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}}
    (OUT/'discovery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(audit))

if __name__=='__main__': main()
