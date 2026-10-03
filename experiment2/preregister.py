"""E2 hash manifest and exact remote-content verification, before test forwards."""
import argparse
import base64
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('outputs/experiment2')
MANIFEST=Path('experiment2/manifest.json')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def api(path):return json.loads(subprocess.check_output(['gh','api',path]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--verify')
    args=p.parse_args()
    if (ROOT/'confirmatory').exists():raise RuntimeError('No retrospective registration')
    if args.prepare:
        required=['experiment2/predictions.json','experiment2/PREDICTION.md','experiment2/PROTOCOL.md',
                  'outputs/experiment2/PREFLIGHT_REVIEW.md','outputs/experiment2/preflight_data_verification.json']
        for path in required:
            if not Path(path).exists():raise RuntimeError('Missing prerequisite '+path)
        files=[x for x in Path('experiment2').rglob('*') if x.suffix in ['.py','.json','.md'] and x!=MANIFEST]
        files += [x for x in ROOT.rglob('*') if x.is_file() and x.name!='remote_lock.json']
        files += [Path(x) for x in ['model.py','train.py','evaluate.py','prepare_holdout.py','audit_holdout.py',
                  'protocol.json','requirements.txt','outputs/machine.json','outputs/holdout_v2.npz','outputs/holdout_v2.json']]
        files += [Path(f'outputs/trained/seed_{s}.pt') for s in range(3)]
        hashes={str(path):digest(path) for path in sorted(set(files))}
        MANIFEST.write_text(json.dumps({'experiment':2,'files':hashes},indent=2)+'\n')
        print(json.dumps({'files':len(hashes),'manifest_sha256':digest(MANIFEST)}))
    elif args.verify:
        commit=api(f'repos/posix4e/ai-understanding-ai/git/commits/{args.verify}')
        if commit['sha']!=args.verify:raise RuntimeError('Commit mismatch')
        remote=api(f'repos/posix4e/ai-understanding-ai/contents/experiment2/manifest.json?ref={args.verify}')
        data=base64.b64decode(remote['content'])
        if data!=MANIFEST.read_bytes():raise RuntimeError('Manifest readback mismatch')
        tree=api(f"repos/posix4e/ai-understanding-ai/git/trees/{commit['tree']['sha']}?recursive=1")
        if tree.get('truncated'):raise RuntimeError('Truncated Git tree')
        blobs={x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'}
        hashes=json.loads(data)['files'];hashes[str(MANIFEST)]=digest(MANIFEST)
        for path,expected in hashes.items():
            data=Path(path).read_bytes()
            if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('Local mismatch '+path)
            blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if blobs.get(path)!=blob:raise RuntimeError('Remote blob mismatch '+path)
        lock={'experiment':2,'commit':args.verify,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
              'remote_tree_sha':commit['tree']['sha'],'verified_blob_count':len(hashes),
              'verification':'Remote commit+manifest+recursive Git tree; every local blob identity checked','files':hashes}
        (ROOT/'remote_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
        print(json.dumps({k:v for k,v in lock.items() if k!='files'},indent=2))
    else:p.error('Choose --prepare or --verify COMMIT')

if __name__=='__main__':main()
