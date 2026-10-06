"""External watchdog for the single local collector; no retries."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time
HERE=Path(__file__).resolve().parent

def launch(out,receipt):
    out=Path(out);receipt=Path(receipt)
    if out.exists() or receipt.exists():raise FileExistsError('Refuse to repeat an existing attempt')
    if not (HERE/'audit.py').is_file():raise RuntimeError('Independent auditor must exist before launch')
    reason=None
    def on_signal(signum,frame):
        nonlocal reason
        reason='external_signal_'+str(signum)
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    t0=time.monotonic();child=subprocess.Popen([sys.executable,'-B',str(HERE/'run.py'),'--out',str(out)],start_new_session=True)
    peak=0
    try:
        while child.poll() is None:
            if reason is not None:break
            elapsed=time.monotonic()-t0
            if elapsed>900:reason='external_timeout';break
            r=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True,timeout=5)
            if r.returncode==0 and r.stdout.strip():
                rss=int(r.stdout.strip())*1024;peak=max(peak,rss)
                if rss>2*2**30:reason='external_RSS_bound';break
            time.sleep(.5)
    except BaseException as exc:
        reason=type(exc).__name__+': '+str(exc)
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        if time.monotonic()-t0>900 and reason is None:reason='external_timeout_at_exit'
        result=dict(status='PASS_COLLECTOR_EXIT' if child.returncode==0 and reason is None else 'STOP',pid=child.pid,returncode=child.returncode,reason=reason,elapsed_seconds=time.monotonic()-t0,peak_sampled_rss_bytes=peak,poll_seconds=.5,timeout_seconds=900,retries=0)
        receipt.parent.mkdir(parents=True,exist_ok=True)
        with receipt.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        if result['status']=='STOP' and out.exists() and not (out/'STOP.json').exists():
            with (out/'STOP.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result),flush=True)
    return 0 if result['status']=='PASS_COLLECTOR_EXIT' else 1
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--receipt',required=True);a=ap.parse_args();raise SystemExit(launch(a.out,a.receipt))
