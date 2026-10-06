"""Single bounded, no-network CPU pilot. Train all six fits before grading."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,resource,shutil,sys,time
import numpy as np
import torch
from torch import nn
from world import make_plan,probabilities,policy,episode

HERE=Path(__file__).resolve().parent

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def features(states):
    sf=np.zeros((len(states),12),np.float32)
    for i,(x,y,k) in enumerate(states):sf[i,x]=1;sf[i,5+y]=1;sf[i,10+k]=1
    sa=np.concatenate((np.repeat(sf,4,axis=0),np.tile(np.eye(4,dtype=np.float32),(len(states),1))),axis=1)
    return torch.from_numpy(sf),torch.from_numpy(sa)

class Direct(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,30))
    def forward(self,sa,candidates):return self.net(sa)
class Energy(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(28,71),nn.GELU(),nn.Linear(71,71),nn.GELU(),nn.Linear(71,1))
    def forward(self,sa,candidates):
        joint=torch.cat((sa[:,None,:].expand(-1,30,-1),candidates[None,:,:].expand(sa.shape[0],-1,-1)),dim=-1)
        return -self.net(joint).squeeze(-1)

def one_step(P,targets,indices):
    p=P.reshape(120,30)[indices];y=targets.reshape(-1)[indices]
    top=p.max(1);ties=(p==top[:,None]).sum(1);win=p.argmax(1)
    return dict(n=len(indices),correct=int(((win==y)&(ties==1)).sum()),ties=int((ties>1).sum()),
                accuracy=float(((win==y)&(ties==1)).mean()),capped_nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],np.finfo(float).tiny)).mean()),zero_true_probability=int((p[np.arange(len(y)),y]==0).sum()),
                true_probability=float(p[np.arange(len(y)),y].mean()))

def rollout(P,table,cases,prefixes):
    rows={str(h):[] for h in prefixes}
    for case in cases:
        d=np.zeros(30);s=case['start'];d[s]=1
        for h,a in enumerate(case['actions'],1):
            d=d@P[:,a,:];s=int(table[s,a])
            if h in prefixes:
                ties=np.flatnonzero(d==d.max());rows[str(h)].append((len(ties)==1 and int(ties[0])==s,len(ties)>1,float(d[s])))
    return {h:dict(n=len(v),correct=sum(int(t[0]) for t in v),ties=sum(int(t[1]) for t in v),accuracy=float(np.mean([t[0] for t in v])),true_probability=float(np.mean([t[2] for t in v])),capped_nll=float(np.mean([-np.log(max(t[2],np.finfo(float).tiny)) for t in v])),zero_true_probability=sum(int(t[2]==0) for t in v)) for h,v in rows.items()}

def planning_summary(rows):
    ok=[r for r in rows if r['success']]
    return dict(n=len(rows),successes=len(ok),success_rate=len(ok)/len(rows),cycles=sum(r['cycle'] for r in rows),
      mean_steps_success=float(np.mean([r['steps'] for r in ok])) if ok else None,
      mean_excess_steps_success=float(np.mean([r['steps']-r['optimal_steps'] for r in ok])) if ok else None,
      tied_decisions=sum(sum(n>1 for n in r['tie_counts']) for r in rows),decisions=sum(r['steps'] for r in rows),
      by_distance={label:dict(n=len(g),successes=sum(r['success'] for r in g)) for label,g in [('1-3',[r for r in rows if r['optimal_steps']<=3]),('4-6',[r for r in rows if 4<=r['optimal_steps']<=6]),('7+',[r for r in rows if r['optimal_steps']>=7])]})

def run(out):
    started=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    plan=make_plan()
    plan['created_at_utc']=datetime.now(timezone.utc).isoformat()
    plan['source_sha256']={p.name:digest(p) for p in sorted(HERE.glob('*.py'))}
    plan['runtime']={'python':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__}
    plan['nll_definition']='capped_nll = mean(-log(max(true_probability,float64 tiny))). Exact zero counts are reported separately; true NLL is infinite if any true probability is zero. No probability smoothing or renormalization is added.'
    plan['prediction_format']='float32 categorical logits, shape [30 states,4 actions,30 candidates]. energy_* keys store NEGATIVE energies, not raw energies. Probability reconstruction uses float64 softmax.'
    dump(out/'plan.json',plan)
    pid=os.getpid();dump(out/'OWNER.json',dict(pid=pid,status='RUNNING',started_at=plan['created_at_utc']))
    def guard():
        if time.monotonic()-started>900:raise RuntimeError('Time cap exceeded')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if rss>2*2**30:raise RuntimeError('RSS cap exceeded')
        if shutil.disk_usage(out).free<8*2**30:raise RuntimeError('Free-storage bound failed')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>128*2**20:raise RuntimeError('Output-storage bound failed')
        return int(rss)
    try:
        guard();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        states=plan['states'];table=np.array(plan['transitions']);sf,sa=features(states)
        train=np.array(plan['train']);targets=torch.tensor(table.reshape(-1),dtype=torch.long)
        fitted={};fit_records=[]
        for seed in plan['seeds']:
            rng=np.random.default_rng(seed)
            batches=rng.choice(train,size=(1500,32),replace=True)
            for kind,cls in [('direct',Direct),('energy',Energy)]:
                guard();torch.manual_seed(seed);model=cls();model.train()
                count=sum(p.numel() for p in model.parameters());assert count=={'direct':7198,'energy':7243}[kind]
                optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=0)
                trace=[];t0=time.monotonic()
                for update,b in enumerate(batches,1):
                    optimizer.zero_grad(set_to_none=True);logits=model(sa[b],sf)
                    loss=nn.functional.cross_entropy(logits,targets[b])
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();optimizer.step()
                    if update%250==0:
                        guard();trace.append({'update':update,'batch_loss':float(loss.detach())})
                model.eval();name=f'{kind}_{seed}';fitted[name]=model
                torch.save(model.state_dict(),out/f'{name}.pt')
                fit_records.append(dict(model=name,seed=seed,kind=kind,parameters=count,steps=1500,seconds=time.monotonic()-t0,training_loss_trace=trace,checkpoint_sha256=digest(out/f'{name}.pt')))
                dump(out/'FIT_PROGRESS.json',fit_records)
                print(json.dumps(dict(phase='training',completed=len(fit_records),of=6,model=name,seconds=round(fit_records[-1]['seconds'],2))),flush=True)
        # No held-out grading or planning precedes this immutable phase record.
        dump(out/'TRAINING_COMPLETE.json',dict(fits=fit_records,all_six_complete=True,elapsed_seconds=time.monotonic()-started,plan_sha256=digest(out/'plan.json')))
        raw={}
        with torch.no_grad():
            for name,model in fitted.items():
                guard();raw[name]=model(sa,sf).numpy().reshape(30,4,30)
                np.save(out/f'predictions-{name}.npy',raw[name],allow_pickle=False)
                if not np.isfinite(raw[name]).all():raise RuntimeError('Nonfinite predictions')
        np.savez_compressed(out/'predictions.npz',**raw)
        P={k:probabilities(v) for k,v in raw.items()};P['oracle']=np.eye(30)[table]
        report={'status':'COLLECTED_PENDING_INDEPENDENT_AUDIT','models':{},'fits':fit_records,'model_training_runs':6,'provider_calls':0,'new_spending_usd':0,'novelty_established':False}
        episodes=[]
        for name,p in P.items():
            guard();m={'one_step':{},'rollout':rollout(p,table,plan['rollout_cases'],plan['rollout_prefixes']),'planning':{}}
            for control,pc in [('native',p),('wrong_action',p[:,[1,2,3,0],:]),('action_average',np.repeat(p.mean(axis=1,keepdims=True),4,axis=1))]:
                m['one_step'][control]={split:one_step(pc,table,plan[split]) for split in ('train','test')}
            for horizon in plan['horizons']:
                bygoal={tuple(g):policy(p,states,g,horizon) for g in {tuple(c['goal']) for c in plan['planning_cases']}}
                rows=[]
                for case in plan['planning_cases']:
                    pi,ties=bygoal[tuple(case['goal'])];r=dict(model=name,horizon=horizon,**case,**episode(table,states,case['start'],case['goal'],pi,ties))
                    rows.append(r)
                episodes.extend(rows);m['planning'][str(horizon)]=planning_summary(rows)
            report['models'][name]=m
            print(json.dumps(dict(phase='evaluation',model=name,heldout_correct=m['one_step']['native']['test']['correct'],planning_successes={h:r['successes'] for h,r in m['planning'].items()})),flush=True)
        if report['models']['oracle']['planning']['16']['successes']!=600:raise RuntimeError('Oracle positive control failed')
        dump(out/'episodes.json',episodes);dump(out/'report.json',report)
        maxrss=guard();dump(out/'COLLECTED.json',dict(status='COLLECTED_PENDING_INDEPENDENT_AUDIT',seconds=time.monotonic()-started,max_rss_bytes=maxrss,episodes=len(episodes),model_fits=6,prediction_rows=720,source_sha256=plan['source_sha256'],files_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='OWNER.json'}))
        dump(out/'OWNER.json',dict(pid=pid,status='EXITED_COLLECTION_COMPLETE'))
    except BaseException as exc:
        dump(out/'STOP.json',dict(status='STOP',type=type(exc).__name__,reason=str(exc),elapsed_seconds=time.monotonic()-started))
        dump(out/'OWNER.json',dict(pid=pid,status='EXITED_STOP'));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();run(args.out)
