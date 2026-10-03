"""E4 manifest and exact public Git blob verification before trained forwards."""
import argparse
import base64
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('outputs/experiment4')
MANIFEST = Path('experiment4/manifest.json')
REPO = 'repos/posix4e/ai-understanding-ai'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path]))


def checkpoint(seed):
    parent = 'outputs/trained' if seed < 3 else 'outputs/experiment2/checkpoints'
    return Path(parent) / f'seed_{seed}.pt'


def verify_local_lock():
    lock = json.loads((ROOT / 'remote_lock.json').read_text())
    for path, expected in lock['files'].items():
        if digest(path) != expected:
            raise RuntimeError('Frozen file changed: ' + path)
    remote = api(f"{REPO}/git/commits/{lock['commit']}")
    if remote['sha'] != lock['commit']:
        raise RuntimeError('Remote commit mismatch')
    return lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--verify')
    args = parser.parse_args()
    if (ROOT / 'confirmatory').exists():
        raise RuntimeError('No retrospective registration')
    if args.prepare:
        if MANIFEST.exists():
            raise RuntimeError('Refusing to replace an existing manifest')
        for name in ['experiment4/PREDICTION.md', 'experiment4/predictions.json',
                     'experiment4/PROTOCOL.md', 'experiment4/protocol.json',
                     'outputs/experiment4/PREFLIGHT_REVIEW.md',
                     'outputs/experiment4/preflight_data_verification.json']:
            if not Path(name).exists():
                raise RuntimeError('Missing prerequisite ' + name)
        files = [x for x in Path('experiment4').rglob('*')
                 if x.suffix in ['.py', '.md', '.json'] and x != MANIFEST]
        files += [x for x in ROOT.rglob('*') if x.is_file() and x.name != 'remote_lock.json']
        files += [Path(x) for x in [
            'model.py', 'train.py', 'evaluate.py', 'prepare_holdout.py',
            'audit_holdout.py', 'protocol.json', 'requirements.txt', 'outputs/machine.json',
            'experiment2/data.py', 'experiment3/data.py', 'outputs/experiment3/inputs.npz',
            'outputs/experiment3/confirmatory/summary.json', 'outputs/holdout_v2.npz', 'outputs/holdout_v2.json',
            'outputs/experiment2/calibration_inputs.npz',
            'outputs/experiment2/discovery_inputs.npz',
            'outputs/experiment2/confirmatory_inputs.npz']]
        files += [checkpoint(seed) for seed in range(6)]
        hashes = {str(x): digest(x) for x in sorted(set(files))}
        MANIFEST.write_text(json.dumps({'experiment': 4, 'files': hashes}, indent=2) + '\n')
        print(json.dumps({'files': len(hashes), 'manifest_sha256': digest(MANIFEST)}))
    elif args.verify:
        if (ROOT / 'remote_lock.json').exists():
            raise RuntimeError('Refusing to replace an existing public-registration lock')
        commit = api(f'{REPO}/git/commits/{args.verify}')
        if commit['sha'] != args.verify:
            raise RuntimeError('Commit mismatch')
        remote = api(f'{REPO}/contents/{MANIFEST}?ref={args.verify}')
        content = base64.b64decode(remote['content'])
        if content != MANIFEST.read_bytes():
            raise RuntimeError('Manifest readback mismatch')
        tree = api(f"{REPO}/git/trees/{commit['tree']['sha']}?recursive=1")
        if tree.get('truncated'):
            raise RuntimeError('Truncated Git tree')
        blobs = {x['path']: x['sha'] for x in tree['tree'] if x['type'] == 'blob'}
        hashes = json.loads(content)['files']
        hashes[str(MANIFEST)] = digest(MANIFEST)
        for path, expected in hashes.items():
            data = Path(path).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Local mismatch ' + path)
            git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if blobs.get(path) != git_hash:
                raise RuntimeError('Remote blob mismatch ' + path)
        lock = {'experiment': 4, 'commit': args.verify,
                'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                'remote_tree_sha': commit['tree']['sha'],
                'verified_blob_count': len(hashes), 'files': hashes,
                'verification': 'Remote commit, manifest and recursive Git tree; all local blobs checked'}
        (ROOT / 'remote_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
        print(json.dumps({k: v for k, v in lock.items() if k != 'files'}, indent=2))
    else:
        parser.error('Choose --prepare or --verify COMMIT')


if __name__ == '__main__':
    main()
