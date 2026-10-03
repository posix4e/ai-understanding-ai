"""Freeze E5 scientific dependencies and verify the exact public Git objects."""
import argparse
import base64
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('outputs/experiment5')
MANIFEST = Path('experiment5/manifest.json')
REPO = 'repos/posix4e/ai-understanding-ai'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path]))


def checkpoint(seed):
    return ROOT / f'checkpoints/seed_{seed}.pt'


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
            raise RuntimeError('Refusing to replace a manifest')
        required = ['experiment5/PREDICTION.md', 'experiment5/predictions.json',
                    'experiment5/PROTOCOL.md', 'experiment5/protocol.json',
                    'experiment5/conditions.json', 'experiment5/scorer.py',
                    'experiment5/report.py', 'experiment5/test_score.py',
                    'outputs/experiment5/PREFLIGHT_REVIEW.md',
                    'outputs/experiment5/preflight_data_verification.json']
        for name in required:
            if not Path(name).is_file():
                raise RuntimeError('Missing prerequisite ' + name)
        files = {Path(p) for p in json.loads(Path('experiment4/manifest.json').read_text())['files']}
        files.update(p for p in Path('experiment5').rglob('*')
                     if p.suffix in ['.py', '.md', '.json'] and p != MANIFEST)
        files.update(p for p in ROOT.rglob('*') if p.is_file() and p.name != 'remote_lock.json')
        files.update(Path(p) for p in ['experiment4/data.py', 'outputs/experiment4/inputs.npz',
                     'outputs/experiment4/RESULTS.md', 'outputs/experiment4/SKEPTICAL_REVIEW.md',
                     'outputs/experiment4/confirmatory/summary.json',
                     'outputs/PUBLICATION_GAP_REVIEW.md', 'outputs/PUBLICATION_PLAN.md'])
        hashes = {str(p): digest(p) for p in sorted(files)}
        MANIFEST.write_text(json.dumps({'experiment': 5, 'files': hashes}, indent=2) + '\n')
        print(json.dumps({'files': len(hashes), 'manifest_sha256': digest(MANIFEST)}))
    elif args.verify:
        if (ROOT / 'remote_lock.json').exists():
            raise RuntimeError('Refusing to replace a public lock')
        commit = api(f'{REPO}/git/commits/{args.verify}')
        if commit['sha'] != args.verify:
            raise RuntimeError('Commit mismatch')
        remote = api(f'{REPO}/contents/{MANIFEST}?ref={args.verify}')
        content = base64.b64decode(remote['content'])
        if content != MANIFEST.read_bytes():
            raise RuntimeError('Remote manifest differs')
        tree = api(f"{REPO}/git/trees/{commit['tree']['sha']}?recursive=1")
        if tree.get('truncated'):
            raise RuntimeError('Truncated tree')
        blobs = {x['path']: x['sha'] for x in tree['tree'] if x['type'] == 'blob'}
        hashes = json.loads(content)['files']
        hashes[str(MANIFEST)] = digest(MANIFEST)
        for path, expected in hashes.items():
            data = Path(path).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Local frozen-file mismatch: ' + path)
            if blobs.get(path) != hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest():
                raise RuntimeError('Remote blob mismatch: ' + path)
        lock = {'experiment': 5, 'commit': args.verify,
                'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                'remote_tree_sha': commit['tree']['sha'], 'verified_blob_count': len(hashes),
                'files': hashes, 'verification': 'Exact public commit, manifest and recursive Git blob identities'}
        (ROOT / 'remote_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
        print(json.dumps({k: v for k, v in lock.items() if k != 'files'}, indent=2))
    else:
        parser.error('Choose --prepare or --verify COMMIT')


if __name__ == '__main__':
    main()
