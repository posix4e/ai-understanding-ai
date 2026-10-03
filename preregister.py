"""Prepare hashes, then verify public commit/tree/blob content before confirmation."""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime,timezone

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def api(endpoint):
    return json.loads(subprocess.check_output(['gh','api',endpoint]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--verify',metavar='COMMIT')
    args=parser.parse_args()
    if Path('outputs/confirmatory_v2').exists():
        raise RuntimeError('Confirmation directory already exists; no retrospective registration')
    if args.prepare:
        paths=[Path(x) for x in ['model.py','train.py','evaluate.py','audit_holdout.py','score.py','preregister.py',
            'protocol.json','predictions.json','PREDICTION.md','PROTOCOL.md','requirements.txt','tests/preflight.py',
            'prepare_holdout.py','outputs/holdout_v2.npz','outputs/holdout_v2.json','outputs/PREDICTION_REAFFIRMATION_V2.md']]
        for pattern in ['outputs/trained/*','outputs/discovery/*','outputs/machine.json',
                        'outputs/training_recovery.md','outputs/review_preflight.md','outputs/preflight_runtime.json',
                        'outputs/preflight_final.json','outputs/review_design.md','outputs/smoke_checks.json']:
            paths.extend(Path('.').glob(pattern))
        for pattern in ['outputs/preflight_v2.json','outputs/review_v2_preflight.md','outputs/holdout_v2_precheck.json','outputs/confirmatory/*',
                        'outputs/remote_lock.json','preregistration_manifest.json']:
            paths.extend(Path('.').glob(pattern))
        files={str(p):sha(p) for p in sorted(set(paths)) if p.is_file()}
        Path('preregistration_manifest_v2.json').write_text(json.dumps({'files':files},indent=2)+'\n')
        print(json.dumps({'files':len(files),'manifest_sha256':sha('preregistration_manifest_v2.json')}))
    elif args.verify:
        commit=api(f'repos/posix4e/ai-understanding-ai/git/commits/{args.verify}')
        if commit['sha']!=args.verify:
            raise RuntimeError('Commit mismatch')
        content=api(f'repos/posix4e/ai-understanding-ai/contents/preregistration_manifest_v2.json?ref={args.verify}')
        remote_manifest=base64.b64decode(content['content'])
        if remote_manifest!=Path('preregistration_manifest_v2.json').read_bytes():
            raise RuntimeError('Remote manifest differs')
        tree=api(f"repos/posix4e/ai-understanding-ai/git/trees/{commit['tree']['sha']}?recursive=1")
        if tree.get('truncated'):
            raise RuntimeError('Truncated remote tree')
        remote_blobs={x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'}
        files=json.loads(remote_manifest)['files']
        files['preregistration_manifest_v2.json']=sha('preregistration_manifest_v2.json')
        for path,expected in files.items():
            if sha(path)!=expected:
                raise RuntimeError(f'Local hash mismatch: {path}')
            data=Path(path).read_bytes()
            blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if remote_blobs.get(path)!=blob:
                raise RuntimeError(f'Remote blob mismatch: {path}')
        lock={'commit':commit['sha'],'verified_at_utc':datetime.now(timezone.utc).isoformat(),
              'remote_tree_sha':commit['tree']['sha'],'verified_blob_count':len(files),
              'verification':'GitHub API commit+manifest readback+recursive tree; every local blob identity checked',
              'files':files}
        Path('outputs/remote_lock_v2.json').write_text(json.dumps(lock,indent=2)+'\n')
        print(json.dumps({k:v for k,v in lock.items() if k!='files'},indent=2))
    else:
        parser.error('Choose --prepare or --verify COMMIT')

if __name__=='__main__':
    main()
