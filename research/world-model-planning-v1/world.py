"""Exact finite key/door world and a fixed finite-horizon planner."""
from collections import deque
import random
import numpy as np

ACTIONS = ((0,-1),(1,0),(0,1),(-1,0))
START = (0,0,0)
KEY = (0,4)
DOOR = (2,2)
WALLS = {(2,y) for y in range(5) if y != 2}
HORIZONS = (1,3,16)

def step(state, action):
    x,y,k=state; dx,dy=ACTIONS[action]; nx,ny=x+dx,y+dy
    if not (0<=nx<5 and 0<=ny<5) or (nx,ny) in WALLS or ((nx,ny)==DOOR and not k):
        return state
    return nx,ny,int(k or (nx,ny)==KEY)

def make_world():
    seen={START}; q=deque([START])
    while q:
        s=q.popleft()
        for a in range(4):
            t=step(s,a)
            if t not in seen:seen.add(t);q.append(t)
    states=sorted(seen); idx={s:i for i,s in enumerate(states)}
    table=np.array([[idx[step(s,a)] for a in range(4)] for s in states],dtype=np.int64)
    return states,table

def split(states):
    rng=random.Random(20261006); idx={s:i for i,s in enumerate(states)}
    forced_test={idx[(0,3,0)]*4+2,idx[(1,2,1)]*4+1}
    forced_train={idx[(1,4,0)]*4+3,idx[(3,2,1)]*4+3}
    held=[]
    for action in range(4):
        forced=sorted(i for i in forced_test if i%4==action)
        pool=[i*4+action for i in range(len(states)) if i*4+action not in forced_test|forced_train]
        rng.shuffle(pool);held+=forced+pool[:6-len(forced)]
    held=sorted(held);train=sorted(set(range(len(states)*4))-set(held))
    return train,held

def distance(table,states,start,goal):
    q=deque([(start,0)]);seen={start}
    while q:
        s,n=q.popleft()
        if tuple(states[s][:2])==tuple(goal):return n
        for t in table[s]:
            t=int(t)
            if t not in seen:seen.add(t);q.append((t,n+1))
    return None

def probabilities(logits):
    x=np.asarray(logits,dtype=np.float64)
    if x.ndim!=3 or not np.isfinite(x).all():raise ValueError('Invalid logits')
    x=x-x.max(axis=-1,keepdims=True);p=np.exp(x);p/=p.sum(axis=-1,keepdims=True)
    return p

def policy(P,states,goal,horizon):
    goalmask=np.array([tuple(s[:2])==tuple(goal) for s in states])
    v=np.array([abs(s[0]-goal[0])+abs(s[1]-goal[1]) for s in states],dtype=np.float64)
    for _ in range(horizon):
        q=1+np.einsum('san,n->sa',P,v,optimize=False)
        v=q.min(axis=1);v[goalmask]=0
    ties=q<=q.min(axis=1,keepdims=True)+1e-12
    return ties.argmax(axis=1),ties.sum(axis=1)

def episode(table,states,start,goal,pi,ties):
    s=start;trace=[s];actions=[];tc=[];visited={s};success=False;cycle=False
    for _ in range(40):
        if tuple(states[s][:2])==tuple(goal):success=True;break
        a=int(pi[s]);tc.append(int(ties[s]));actions.append(a);s=int(table[s,a]);trace.append(s)
        if tuple(states[s][:2])==tuple(goal):success=True;break
        if s in visited:cycle=True;break
        visited.add(s)
    return dict(states=trace,actions=actions,tie_counts=tc,success=success,cycle=cycle,steps=len(actions))

def make_plan():
    states,table=make_world();train,test=split(states)
    goals=sorted({s[:2] for s in states});cases=[]
    for start,s in enumerate(states):
        for g in goals:
            if tuple(s[:2])!=g:
                d=distance(table,states,start,g)
                if d is None:raise ValueError('Unexpected unreachable goal')
                cases.append(dict(start=start,goal=list(g),optimal_steps=d))
    rng=random.Random(20261007)
    rollouts=[dict(start=s,replicate=r,actions=[rng.randrange(4) for _ in range(10)]) for s in range(len(states)) for r in range(4)]
    assert len(states)==30 and len(cases)==600 and max(c['optimal_steps'] for c in cases)<=16
    return dict(schema='world-model-planning-pilot-v1',status='FROZEN_DEVELOPMENT_PILOT',
      world=dict(width=5,height=5,start=list(START),key=list(KEY),door=list(DOOR),walls=[list(s) for s in sorted(WALLS)],actions=['north','east','south','west'],pickup='automatic_permanent',door_rule='requires_key'),
      states=states,transitions=table.tolist(),train=train,test=test,split_seed=20261006,
      forced_holdout=[dict(state=[0,3,0],action=2),dict(state=[1,2,1],action=1)],
      forced_train=[dict(state=[1,4,0],action=3),dict(state=[3,2,1],action=3)],
      planning_cases=cases,rollout_cases=rollouts,rollout_seed=20261007,rollout_prefixes=[1,3,6,10],
      horizons=list(HORIZONS),episode_cap=40,tie_tolerance=1e-12,stop_on_first_true_state_cycle=True,
      objective='Expected unit step cost until goal; terminal Manhattan distance; absorbing goal; fixed first-action tie break N,E,S,W. Exact finite-horizon DP, replanned from fully observed true state.',
      seeds=[11,29,47],training=dict(steps=1500,batch_size=32,learning_rate=0.003,optimizer='Adam',weight_decay=0,activation='GELU',dtype='float32',device='cpu',threads=2,direct_width=64,energy_width=71,shared_batch_order_per_seed=True,all_six_fits_before_evaluation=True),
      fairness='Same observations, candidate vocabulary, examples, updates, batch order and categorical likelihood objective. Approximately matched parameter counts; energy evaluates all candidates and is NOT FLOP/time matched. No claim isolating energy objectives.',
      exposure='One known map. All reachable state labels are public to both models. 24 held-out transition targets excluded from loss; rollout sequences and goal pairs are development evaluations, not independent confirmation or unseen-world tests.',
      controls=['exact_dynamics_same_planner','full_BFS_shortest_paths','cyclic_wrong_action_one_step','action_average_one_step'],
      resources=dict(timeout_seconds=900,max_rss_bytes=2*2**30,max_output_bytes=128*2**20,min_free_bytes=8*2**30),
      budget=dict(provider_calls=0,new_estimated_usd=0,inclusive_cap_usd=20,prior_fees_and_reserves_usd=7.6774551949),
      claims='World-model pilot only. No language host, organ splice, latent JEPA, new architecture, novelty or energy-superiority claim. No outcome-based model, seed, horizon, case or threshold selection.')
