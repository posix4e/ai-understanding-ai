"""External phase watchdog, no retries or replacement runs."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time

def launch(out, phase):
    out = Path(out)
    receipt = out/f'{phase}_WATCHDOG.json'
    if (out/phase).exists() or receipt.exists():
        raise FileExistsError('Refuse repeat phase')
    reason = None
    def stop(signum, frame):
        nonlocal reason
        reason = f'external_signal_{signum}'
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).with_name('run.py')), 'run', '--out', str(out), '--phase', phase], start_new_session=True)
    peak = 0
    try:
        while child.poll() is None and reason is None:
            if time.monotonic()-started > 1800:
                reason = 'external_timeout'
                break
            p = subprocess.run(['ps', '-o', 'rss=', '-p', str(child.pid)], capture_output=True, text=True, timeout=5)
            if p.returncode == 0 and p.stdout.strip():
                peak = max(peak, int(p.stdout.strip())*1024)
                if peak > 2*2**30:
                    reason = 'external_RSS_bound'
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 2**30:
                reason = 'external_output_bound'
            time.sleep(.5)
    except BaseException as exc:
        reason = repr(exc)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        if time.monotonic()-started > 1800 and reason is None:
            reason = 'external_timeout_at_exit'
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 2**30 and reason is None:
            reason = 'external_output_bound_at_exit'
        if peak > 2*2**30 and reason is None:
            reason = 'external_RSS_bound_at_exit'
        result = dict(status='PASS_COLLECTOR_EXIT' if child.returncode == 0 and reason is None else 'STOP',
                      phase=phase, pid=child.pid, returncode=child.returncode, reason=reason,
                      seconds=time.monotonic()-started, peak_sampled_rss_bytes=peak, timeout_seconds=1800, retries=0)
        receipt.write_text(json.dumps(result, indent=2)+'\n')
        if result['status'] == 'STOP' and (out/phase).exists() and not (out/phase/'STOP.json').exists():
            (out/phase/'STOP.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'PASS_COLLECTOR_EXIT' else 1

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--phase', required=True, choices=['engineering', 'evaluation'])
    a = ap.parse_args()
    raise SystemExit(launch(a.out, a.phase))
