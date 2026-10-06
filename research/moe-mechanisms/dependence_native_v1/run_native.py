"""One-shot ungraded CPU BF16 native dependence-panel collection. Importing this file uses stdlib only.

This is not a registration tool. The owner refuses to run without separately
written, source-bound approval. Never call worker() in an existing model process.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREP = ROOT / 'work/dependence-native-panel-preparation'
OUT = ROOT / 'outputs/moe-mechanisms/DEPENDENCE_NATIVE_V1'
PLAN = PREP / 'DRAFT_PLAN.json'
REGISTRATION = PREP / 'REGISTRATION.json'
INPUTS = 'work/dependence-native-panel-inputs/tokenizer-v1/INPUTS.jsonl'
TOKENIZER_RECEIPT = 'work/dependence-native-panel-inputs/tokenizer-v1/COMPLETE.json'
# Prospectively fixed from the complete tokenizer-only input preflight.
EXPECTED_DEADLINE_SECONDS = 14400
EXPECTED_MAXIMUM_INPUT_TOKENS = 357
REPEAT_CELLS = ('c00','c05','c06','c07','c12','c16','c30','c52')
ENV = {
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
    'HF_ENABLE_PARALLEL_LOADING': 'false', 'TOKENIZERS_PARALLELISM': 'false',
    'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
}
FORBIDDEN_ENV = ('PYTORCH_ENABLE_MPS_FALLBACK', 'HF_TOKEN',
                 'HUGGING_FACE_HUB_TOKEN', 'ACCELERATE_USE_CPU')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path, guard=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            guard()
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def verify_file(root, record, guard=lambda: None):
    p = root / record['path']
    require(p.resolve().is_relative_to(root.resolve()), 'binding escapes root')
    require(p.is_file() and p.stat().st_size == record['bytes'], 'binding size: ' + str(p))
    require(digest(p, guard) == record['sha256'], 'binding hash: ' + str(p))


def plan_guard(plan):
    require(EXPECTED_DEADLINE_SECONDS is not None and EXPECTED_MAXIMUM_INPUT_TOKENS is not None,
            'prospective deadline and length cap not frozen')
    require(plan['schema'] == 'smollm3-dependence-native-collection-v1', 'wrong plan schema')
    require(plan['output'] == str(OUT.relative_to(ROOT)), 'wrong output')
    require(plan['input_jsonl'] == INPUTS and plan['tokenizer_receipt'] == TOKENIZER_RECEIPT,
            'fixed input archive changed')
    require(plan['forward_count'] == 472 and plan['maximum_input_tokens'] == EXPECTED_MAXIMUM_INPUT_TOKENS,
            'call or length scope changed')
    lengths=plan['input_lengths']
    require(type(lengths) is list and len(lengths)==472
            and all(type(x) is int and 0<x<=EXPECTED_MAXIMUM_INPUT_TOKENS for x in lengths),
            'fixed input lengths')
    codes=plan['candidate_token_ids']
    require(type(codes) is dict and set(codes)=={'X','Y'}
            and all(type(x) is int and 0<=x<128256 for x in codes.values())
            and codes=={'X':1630,'Y':816}, 'candidate token IDs')
    require(plan['limits'] == {
        'deadline_seconds': EXPECTED_DEADLINE_SECONDS, 'maximum_rss_bytes': 8 * 2**30,
        'maximum_new_evidence_bytes': 512 * 2**20,
        'minimum_free_disk_bytes': 6 * 2**30,
        'sample_interval_seconds': 0.2, 'emergency_reserve_bytes': 2**20,
    }, 'resource contract changed')


def expected_case(index):
    require(type(index) is int and 0<=index<472, 'case index')
    if index<464:
        cell=f'c{index//8:02d}'
        r,m,o=(index%8//4,index%4//2,index%2)
        variant=f'r{r}-m{m}-o{o}'
        return {'index':index,'id':cell+'-'+variant,'cell_id':cell,'variant_id':variant,
                'replicate':0,'repeat_of':None}
    cell=REPEAT_CELLS[index-464]; original=cell+'-r0-m0-o0'
    return {'index':index,'id':original+'-repeat','cell_id':cell,'variant_id':'r0-m0-o0',
            'replicate':1,'repeat_of':original}


def load_inputs(plan):
    """Validate frozen token records only; never read cases or evaluator gold."""
    receipt=read_json(ROOT/plan['tokenizer_receipt'])
    require(receipt['status']=='PASS_TOKENIZER_ONLY_NATIVE_PANEL_INPUTS'
            and receipt['rows']==472 and receipt['shared_evidence_prefix_groups']==14,
            'tokenizer preflight incomplete')
    require(receipt['candidate_token_ids']==plan['candidate_token_ids']
            and receipt['all_complete_roundtrips'] is True
            and receipt['all_one_token_extensions'] is True
            and receipt['all_evidence_prefixes_identical_across_late_contexts'] is True
            and receipt['all_repeats_input_exact'] is True and receipt['truncation'] is False,
            'tokenizer contract failed')
    path=ROOT/plan['input_jsonl']; rec=receipt['files']['INPUTS.jsonl']
    require(path.stat().st_size==rec['bytes'] and digest(path)==rec['sha256'], 'input receipt binding')
    records=[json.loads(line) for line in path.read_text().splitlines()]
    require(len(records)==472, 'input census')
    keys={'index','id','cell_id','variant_id','replicate','repeat_of','user_text_sha256',
          'evidence_prefix_sha256','input_ids','attention_mask','token_count','candidate_ids',
          'rendered_sha256','evidence_prefix_token_ids','evidence_prefix_last_token_index',
          'evidence_prefix_rendered_utf8_bytes'}
    seen={}; prefix_groups=set()
    for i,row in enumerate(records):
        require(type(row) is dict and set(row)==keys, 'input record schema')
        for key,value in expected_case(i).items():
            require(type(row[key]) is type(value) and row[key]==value, 'input schedule '+key)
        ids=row['input_ids']; mask=row['attention_mask']; count=row['token_count']
        require(type(count) is int and count==plan['input_lengths'][i], 'token count')
        require(type(ids) is list and len(ids)==count
                and all(type(x) is int and 0<=x<128256 for x in ids), 'input IDs')
        require(type(mask) is list and len(mask)==count
                and all(type(x) is int and x==1 for x in mask), 'attention mask')
        codes=row['candidate_ids']
        require(type(codes) is dict and set(codes)=={'X','Y'}
                and all(type(x) is int for x in codes.values())
                and codes==plan['candidate_token_ids'], 'candidate metadata')
        last=row['evidence_prefix_last_token_index']; prefix=row['evidence_prefix_token_ids']
        require(type(last) is int and 0<=last<count-1 and type(prefix) is list
                and all(type(x) is int for x in prefix) and prefix==ids[:last+1], 'evidence prefix IDs')
        require(type(row['evidence_prefix_rendered_utf8_bytes']) is int
                and row['evidence_prefix_rendered_utf8_bytes']>0, 'prefix byte boundary')
        for key in ('user_text_sha256','evidence_prefix_sha256','rendered_sha256'):
            value=row[key]
            require(type(value) is str and len(value)==64
                    and all(x in '0123456789abcdef' for x in value), 'text hash')
        prefix_groups.add((row['evidence_prefix_sha256'],tuple(prefix)))
        if row['repeat_of'] is not None:
            original=seen[row['repeat_of']]
            require(all(row[k]==original[k] for k in keys-set(expected_case(i))), 'repeat input changed')
        seen[row['id']]=row
    require(len(prefix_groups)==14, 'shared prefix census')
    return records


def resource_violations(sample, limits):
    issues = []
    for key, ceiling in [('rss_bytes', 'maximum_rss_bytes'),
                         ('evidence_bytes', 'maximum_new_evidence_bytes')]:
        if key in sample and sample[key] > limits[ceiling]:
            issues.append(key)
    if sample.get('free_disk_bytes', limits['minimum_free_disk_bytes']) < limits['minimum_free_disk_bytes']:
        issues.append('free_disk_bytes')
    return issues


def ps_rows(text):
    rows = []
    for line in text.splitlines():
        fields = line.split()
        require(len(fields) == 4, 'malformed ps row')
        pid, ppid, pgid, rss_kib = map(int, fields)
        rows.append({'pid': pid, 'ppid': ppid, 'pgid': pgid, 'rss_bytes': rss_kib * 1024})
    return rows


def process_table():
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,pgid=,rss='],
                            check=True, capture_output=True, text=True, timeout=2)
    return ps_rows(result.stdout)


def process_tree(rows, root_pid):
    selected = {root_pid}
    while True:
        enlarged = selected | {r['pid'] for r in rows if r['ppid'] in selected}
        if enlarged == selected:
            return [r for r in rows if r['pid'] in selected]
        selected = enlarged


def group_members(pgid):
    return [r['pid'] for r in process_table() if r['pgid'] == pgid]


def cleanup_group(proc, deadline):
    """Always clean the owned group, including after its leader already exited."""
    actions = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        proc.poll()
        if not group_members(proc.pid):
            proc.wait(timeout=max(.01, deadline - time.monotonic()))
            return {'actions': actions, 'group_absent': True, 'exit_code': proc.returncode}
        try:
            os.killpg(proc.pid, sig); actions.append(signal.Signals(sig).name)
        except ProcessLookupError:
            pass
        until = min(deadline, time.monotonic() + 2.0)
        while time.monotonic() < until:
            proc.poll()  # reap the direct child before testing group absence
            if not group_members(proc.pid):
                proc.wait(timeout=max(.01, deadline - time.monotonic()))
                return {'actions': actions, 'group_absent': True, 'exit_code': proc.returncode}
            time.sleep(.05)
    proc.poll()
    members = group_members(proc.pid)
    return {'actions': actions, 'group_absent': not members,
            'remaining_pids': members, 'exit_code': proc.returncode}


def evidence_size(directory):
    return sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())


def validate_loading_info(info):
    require(type(info) is dict, 'loading_info missing')
    require(set(info) == {'missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'},
            'unknown loading_info fields')
    require(all(info[k] == [] for k in info), 'nonempty native loading diagnostics')


def watch_lifetime(owner_pid, deadline, stop_event, on_failure, interval=0.2):
    """Stdlib-only orphan/deadline fallback; it does not measure resources."""
    while not stop_event.is_set():
        reason = ('owner_lost' if os.getppid() != owner_pid else
                  'deadline' if time.monotonic() >= deadline else None)
        if reason is not None:
            try:
                on_failure(reason)
            finally:
                os.kill(os.getpid(), signal.SIGTERM)
            return
        stop_event.wait(interval)


def publish_terminal(directory, terminal, deadline, limits):
    """Durable pending bytes precede the success name; known overruns remain STOP."""
    if terminal['status']!='NATIVE_COLLECTION_COMPLETE_UNGRADED':
        write_json(directory/'STOPPED.json',terminal)
        return terminal
    pending=directory/'TERMINAL_PENDING.json'
    published=False
    try:
        # The estimate includes its own fields; iterate until serialized length is stable.
        terminal['publication']='Pending receipt becomes COMPLETE only after guarded durable write.'
        terminal['pending_payload_bytes']=0
        for _ in range(12):
            size=len(json.dumps(terminal,indent=2,sort_keys=True,allow_nan=False).encode())+1
            if size==terminal['pending_payload_bytes']: break
            terminal['pending_payload_bytes']=size
        else: raise RuntimeError('terminal size did not stabilize')
        space={'evidence_bytes':evidence_size(directory)+size,
               'free_disk_bytes':shutil.disk_usage(directory).free-size}
        require(time.monotonic()<deadline and not resource_violations(space,limits),
                'terminal prewrite limit')
        write_json(pending,terminal)
        require(pending.stat().st_size==size,'terminal serialization size differs')
        space={'evidence_bytes':evidence_size(directory),
               'free_disk_bytes':shutil.disk_usage(directory).free}
        require(time.monotonic()<deadline and not resource_violations(space,limits),
                'terminal durable-write limit')
        os.link(pending,directory/'COMPLETE.json')  # exclusive publication, never overwrite
        published=True
        if time.monotonic()>=deadline:
            (directory/'COMPLETE.json').unlink()
            raise RuntimeError('terminal publication deadline exceeded')
        pending.unlink()
        if time.monotonic()>=deadline:
            os.link(directory/'COMPLETE.json',pending)
            (directory/'COMPLETE.json').unlink()
            raise RuntimeError('terminal finalization deadline exceeded')
        return terminal
    except BaseException as e:
        if published and (directory/'COMPLETE.json').exists():
            if not pending.exists(): os.link(directory/'COMPLETE.json',pending)
            (directory/'COMPLETE.json').unlink()
        failure=dict(terminal)
        failure.update(status='NATIVE_COLLECTION_STOPPED_UNGRADED',
                       publication_failure=repr(e),publication_failure_utc=now())
        if pending.exists():
            failure['preserved_pending']={'path':pending.name,'bytes':pending.stat().st_size,
                                          'sha256':digest(pending)}
        # This emergency terminal is retained even when a known limit has failed.
        write_json(directory/'STOPPED.json',failure)
        return failure


def verify_bindings(plan, guard=lambda: None):
    for record in plan['bindings']:
        guard(); verify_file(ROOT, record, guard)
    model_dir=ROOT/plan['model_path']
    expected={r['path'] for r in plan['bindings'] if Path(r['path']).is_relative_to(plan['model_path'])}
    actual={str(p.relative_to(ROOT)) for p in model_dir.rglob('*') if p.is_file()}
    require(actual==expected and len(expected)==28,'model asset and download-record inventory changed')
    for previous in plan.get('prior_stopped_runs', []):
        directory=ROOT/previous['output']
        actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        require(actual==set(previous['inventory']), 'prior stopped inventory changed')
        require(previous['status']=='NATIVE_NOOP_ADMISSION_STOPPED'
                and read_json(directory/'STOPPED.json')['status']==previous['status'],
                'prior STOP not preserved')
    guard()


class Store:
    def __init__(self, directory, limits):
        self.directory, self.limits = directory, limits
        self.lock = threading.RLock()

    def check_space(self, extra=0):
        require(shutil.disk_usage(self.directory).free >= self.limits['minimum_free_disk_bytes'],
                'free disk floor')
        require(evidence_size(self.directory) + extra <=
                self.limits['maximum_new_evidence_bytes'] - self.limits['emergency_reserve_bytes'],
                'ordinary evidence capacity')

    def json(self, name, value, emergency=False):
        with self.lock:
            if not emergency:
                self.check_space(len(canonical(value)) * 2 + 2048)
            write_json(self.directory / name, value)

    def event(self, kind, **fields):
        row = {'kind': kind, 'utc': now(), 'monotonic': time.monotonic(), **fields}
        with self.lock:
            self.check_space(len(canonical(row)) + 1)
            with (self.directory / 'worker-events.jsonl').open('a') as f:
                f.write(canonical(row).decode() + '\n'); f.flush(); os.fsync(f.fileno())

    def tensor(self, name, tensor):
        """One tensor at a time; raw bits are durable before any finite check."""
        import torch
        cpu = tensor.detach().to('cpu').contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        path = self.directory / (name + '.bin'); path.parent.mkdir(exist_ok=True)
        with self.lock:
            self.check_space(len(raw) + 4096)
            with path.open('xb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
        rec = {'path':str(path.relative_to(self.directory)), 'shape':list(cpu.shape),
               'dtype':str(cpu.dtype), 'original_device':str(tensor.device),
               'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
        self.json(name + '.json', rec)
        return rec


def tensor_digest(tensor):
    import torch
    cpu = tensor.detach().to('cpu').contiguous()
    a = cpu.reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(a)).hexdigest()


def identities(model):
    """Runtime identities remain meaningful within this worker, not across PIDs."""
    import torch
    modules = {}
    for name, m in model.named_modules():
        modules[name] = {
            'class': type(m).__module__ + '.' + type(m).__qualname__,
            'training':m.training,
            'methods':{k:id(getattr(getattr(m, k), '__func__', getattr(m, k)))
                       for k in ('forward', 'to', 'train', 'requires_grad_', '_call_impl', '_wrapped_call_impl')},
            'instance_forward': 'forward' in vars(m),
            'instance_device_methods': [k for k in ('to','cuda') if k in vars(m)],
            'compiled_call': getattr(m,'_compiled_call_impl',None) is not None,
            'dispatch_attributes': [k for k in ('_hf_hook', '_old_forward') if hasattr(m, k)],
            'hooks':{k:{str(i):id(v) for i,v in getattr(m,k).items()}
                     for k in ('_forward_hooks','_forward_pre_hooks','_backward_hooks')},
        }
    params = {n:{'id':id(p), 'ptr':p.data_ptr(), 'shape':list(p.shape), 'dtype':str(p.dtype),
                 'device':str(p.device), 'requires_grad':p.requires_grad, 'grad_none':p.grad is None}
              for n,p in model.named_parameters(remove_duplicate=False)}
    buffers = {n:{'id':id(b), 'shape':list(b.shape),'dtype':str(b.dtype),'device':str(b.device),
                  'sha256':tensor_digest(b)} for n,b in model.named_buffers(remove_duplicate=False)}
    module_api = torch.nn.modules.module
    global_hooks={k:{str(i):id(v) for i,v in getattr(module_api,k).items()}
                  for k in ('_global_forward_hooks','_global_forward_pre_hooks','_global_backward_hooks')}
    return {'modules':modules,'parameters':params,'buffers':buffers,'global_hooks':global_hooks,
            'model_config':model.config.to_dict(),
            'generation_config':model.generation_config.to_dict(),
            'attention_implementation':model.config._attn_implementation}


def hash_parameters(model, store, phase, guard):
    rows = {}; aliases = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        guard()
        if id(p) not in aliases:
            aliases[id(p)] = (name, tensor_digest(p))
        original, hashed = aliases[id(p)]
        row={'shape':list(p.shape),'dtype':str(p.dtype),'device':str(p.device),
             'sha256':hashed,'same_parameter_as':None if original == name else original}
        rows[name]=row
        store.event('parameter_hashed', phase=phase, name=name, **row)
    store.json('PARAMETERS_' + phase + '.json', rows)
    return rows


def check_native(model, info, plan, inv):
    from transformers.models.smollm3.modeling_smollm3 import SmolLM3ForCausalLM
    import torch
    validate_loading_info(info)
    require(type(model) is SmolLM3ForCausalLM, 'not exact native class')
    require(model.hf_device_map == {'':'cpu'}, 'device dispatch map')
    require(not getattr(model,'is_quantized',False), 'quantization')
    expected=set(read_json(ROOT/plan['model_path']/'model.safetensors.index.json')['weight_map'])
    require(len(expected)==326 and 'lm_head.weight' not in expected, 'stored tensor set')
    params=dict(model.named_parameters(remove_duplicate=False))
    require(set(params)==expected|{'lm_head.weight'}, 'runtime tensor names')
    require(len({id(p) for p in params.values()})==326, 'unique parameter aliases')
    require(sum(p.numel() for p in model.parameters())==3075098624, 'parameter count')
    a,b=model.model.embed_tokens.weight,model.lm_head.weight
    require(a is b and a.data_ptr()==b.data_ptr(), 'head/embedding alias')
    require(all(p.device.type=='cpu' and p.dtype==torch.bfloat16 for p in params.values()),
            'parameter placement/dtype')
    require(all(b.device.type=='cpu' for b in model.buffers()), 'buffer placement')
    require(len(model.model.layers)==36 and model.config.hidden_size==2048, 'architecture')
    require(model.config._attn_implementation=='eager' and model.config.use_cache is False,
            'attention/cache config')
    for module in inv['modules'].values():
        require(not module['instance_forward'] and not module['dispatch_attributes']
                and not module['instance_device_methods'] and not module['compiled_call'], 'native method wrapped')
        require(not any(module['hooks'].values()), 'undeclared module hook')
    require(not any(inv['global_hooks'].values()), 'undeclared global hook')
    require(all(x['grad_none'] for x in inv['parameters'].values()), 'preexisting gradient')
    return inv


def worker(plan_path, expected_plan_sha, owner_pid):
    require(os.getppid()==owner_pid, 'worker is not direct original child')
    require(digest(plan_path)==expected_plan_sha, 'worker plan changed')
    plan=read_json(plan_path); plan_guard(plan)
    started=read_json(OUT/'STARTED.json')
    require(started['owner_pid']==owner_pid and started['plan_sha256']==expected_plan_sha, 'owner receipt')
    deadline=started['deadline_monotonic']; limits=plan['limits']; store=Store(OUT,limits)
    counts={'load_attempted':0,'load_finished':0,'forward_attempted':0,'forward_finished':0,
            'generation':0,'backward':0,'fits':0,'updates':0,'provider_calls':0,
            'tokenizer_calls':0,'scoring_calls':0}
    model=None; lifetime_thread=None; stop_lifetime=threading.Event()
    original=None; restored=None

    def guard():
        require(os.getppid()==owner_pid, 'owner lost')
        require(time.monotonic()<deadline, 'inclusive deadline')
        store.check_space()

    failure=None; checks={}; before=None; after=None
    try:
        store.event('worker_started',pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),
                    sid=os.getsid(0),plan_sha256=expected_plan_sha)
        require(os.getpgrp()==os.getpid() and os.getsid(0)==os.getpid(), 'worker session ownership')
        def lifetime_failure(reason):
            store.json('LIFETIME_STOP.json',{'utc':now(),'reason':reason,'counts':dict(counts)},emergency=True)
        lifetime_thread=threading.Thread(target=watch_lifetime,
            args=(owner_pid,deadline,stop_lifetime,lifetime_failure),daemon=True)
        lifetime_thread.start()
        for k,v in ENV.items(): require(os.environ.get(k)==v,'environment '+k)
        require(all(k not in os.environ for k in FORBIDDEN_ENV),'forbidden environment')
        require(os.environ.get('PYTHONPATH')==str(ROOT/plan['addon_path']), 'addon search path')
        verify_bindings(plan,guard)
        inputs=load_inputs(plan)
        import torch
        import transformers
        import accelerate
        import psutil
        import safetensors
        import numpy
        imported={'torch':torch,'transformers':transformers,'accelerate':accelerate,
                  'psutil':psutil,'safetensors':safetensors,'numpy':numpy}
        runtime={}
        for name,module in imported.items():
            require(importlib.metadata.version(name)==plan['versions'][name],'runtime version '+name)
            package_path=Path(module.__file__).resolve()
            prefix=ROOT/(plan['addon_path'] if name in ('accelerate','psutil') else plan['base_site_packages'])
            require(package_path.is_relative_to(prefix.resolve()),'runtime provenance '+name)
            runtime[name]={'version':importlib.metadata.version(name),'file':str(package_path),
                           'sha256':digest(package_path)}
        store.json('RUNTIME.json',{'packages':runtime,'sys_path':sys.path,'executable':sys.executable,
                                  'environment':{k:os.environ.get(k) for k in (*ENV,'PYTHONPATH',*FORBIDDEN_ENV)}})
        require(Path(sys.executable).resolve()==(ROOT/plan['python']).resolve(),'wrong interpreter')
        torch.set_num_threads(4); torch.random.default_generator.manual_seed(42)
        require(torch.initial_seed()==42, 'CPU generator seed')
        store.json('CONFIGURED_RUNTIME.json',{'cpu_threads':torch.get_num_threads(),
            'seed':torch.initial_seed(),'default_dtype':str(torch.get_default_dtype()),
            'default_device':str(torch.get_default_device()),'python_version':sys.version,
            'target_device':'cpu'})
        guard(); counts['load_attempted']+=1; store.event('load_started',counts=dict(counts))
        model,info=transformers.AutoModelForCausalLM.from_pretrained(
            ROOT/plan['model_path'],dtype=torch.bfloat16,device_map={'':'cpu'},
            attn_implementation='eager',local_files_only=True,token=False,trust_remote_code=False,
            use_cache=False,use_kernels=False,use_safetensors=True,weights_only=True,
            output_loading_info=True)
        counts['load_finished']+=1
        store.json('LOADING_INFO.json',info); store.event('load_finished',counts=dict(counts))
        original=identities(model); store.json('IDENTITY_AS_LOADED.json',original)
        guard(); check_native(model,info,plan,original)
        before=hash_parameters(model,store,'BEFORE',guard)
        model.eval(); model.requires_grad_(False)
        configured=identities(model); store.json('IDENTITY_CONFIGURED.json',configured)
        with torch.inference_mode():
            for entry in inputs:
                guard(); index=entry['index']; stem=f'calls/{index:03d}'
                rows={}; raw_inputs={}; error=None; returned=False
                record_sha=hashlib.sha256(canonical(entry)).hexdigest()
                try:
                    ids=torch.tensor([entry['input_ids']],dtype=torch.long,device='cpu')
                    mask=torch.tensor([entry['attention_mask']],dtype=torch.long,device='cpu')
                    raw_inputs['input_ids']=store.tensor(stem+'-input_ids',ids)
                    raw_inputs['attention_mask']=store.tensor(stem+'-attention_mask',mask)
                    counts['forward_attempted']+=1
                    store.event('forward_started',index=index,id=entry['id'],
                                input_record_sha256=record_sha,counts=dict(counts))
                    output=model(input_ids=ids,attention_mask=mask,use_cache=False,past_key_values=None,
                                 logits_to_keep=1,output_hidden_states=False,output_attentions=False,
                                 return_dict=True)
                    counts['forward_finished']+=1; returned=True
                    # Retain the complete native final vocabulary before finite validation.
                    rows['logits']=store.tensor(stem+'-logits',output.logits)
                    store.event('forward_returned',index=index,id=entry['id'],
                                counts=dict(counts),logits=rows['logits'])
                    require(output.logits.shape==(1,1,128256),'logit shape')
                    require(output.logits.device.type=='cpu' and ids.device.type=='cpu'
                            and mask.device.type=='cpu','call placement')
                    require(torch.is_inference_mode_enabled() and not torch.is_grad_enabled(),'inference context')
                    require(output.logits.dtype in (torch.bfloat16,torch.float32),'unexpected logit dtype')
                    require(bool(torch.isfinite(output.logits).all().item()),'nonfinite logits')
                    require(output.past_key_values is None,'unexpected KV cache')
                    require(torch.equal(ids,torch.tensor([entry['input_ids']],device='cpu')),'input mutated')
                    require(torch.equal(mask,torch.tensor([entry['attention_mask']],device='cpu')),'attention mask mutated')
                    del output
                    guard()
                    require(identities(model)==configured,'post-call ownership/config/buffer change')
                except BaseException as e:
                    error=repr(e); raise
                finally:
                    store.json(stem+'-CALL.json',{
                        **expected_case(index),'input_record_sha256':record_sha,
                        'candidate_ids':entry['candidate_ids'],'rendered_sha256':entry['rendered_sha256'],
                        'evidence_prefix_sha256':entry['evidence_prefix_sha256'],
                        'evidence_prefix_last_token_index':entry['evidence_prefix_last_token_index'],
                        'forward_returned':returned,'inputs':raw_inputs,'outputs':rows,'error':error},
                        emergency=error is not None)
        after=hash_parameters(model,store,'AFTER',guard)
        checks['all_parameters_unchanged']=before==after
        checks['all_472_calls']=counts['forward_attempted']==counts['forward_finished']==472
        require(all(checks.values()),'declared collection checks failed')
    except BaseException as e:
        failure={'error':repr(e),'traceback':traceback.format_exc()}
    finally:
        cleanup_errors=[]
        if model is not None and original is not None:
            try:
                store.json('IDENTITY_BEFORE_RESTORE.json',identities(model),emergency=failure is not None)
                for name,module in model.named_modules(): module.training=original['modules'][name]['training']
                for name,p in model.named_parameters(remove_duplicate=False):
                    p.requires_grad_(original['parameters'][name]['requires_grad'])
                restored=identities(model)
                store.json('IDENTITY_RESTORED.json',restored,emergency=failure is not None)
                require(restored==original,'as-loaded ownership/buffers not restored')
                checks['ownership_restored']=True
            except BaseException as e: cleanup_errors.append(repr(e))
        if lifetime_thread is not None:
            stop_lifetime.set(); lifetime_thread.join(timeout=2)
            if lifetime_thread.is_alive(): cleanup_errors.append('lifetime watchdog did not stop')
        if cleanup_errors or time.monotonic()>=deadline:
            failure={'original_failure':failure,'cleanup_errors':cleanup_errors,
                     'deadline_exceeded':time.monotonic()>=deadline}
        terminal={'status':'NATIVE_WORKER_COLLECTED_UNGRADED' if failure is None else 'NATIVE_WORKER_STOPPED_UNGRADED', 'utc':now(),
                  'pid':os.getpid(),'owner_pid':owner_pid,'plan_sha256':expected_plan_sha,
                  'counts':counts,'checks':checks,'failure':failure,
                  'final_parameter_hashes_available':after is not None,
                  'restoration_inventory_available':restored is not None}
        store.json('WORKER_RESULT.json',terminal,emergency=failure is not None)
    return 0 if failure is None else 1


def owner():
    begin=time.monotonic()
    plan=read_json(PLAN); plan_guard(plan); deadline=begin+plan['limits']['deadline_seconds']; plan_sha=digest(PLAN)
    reg=read_json(REGISTRATION)
    require(reg['status']=='REGISTERED_APPROVED' and reg['plan_sha256']==plan_sha,'no source-bound registration')
    require(reg['runner_sha256']==digest(Path(__file__)),'runner not registered')
    review_path=ROOT/reg['review_path']; review=read_json(review_path)
    require(digest(review_path)==reg['review_sha256'] and review['status']=='PASS', 'independent source review required')
    require(review['runner_sha256']==reg['runner_sha256'] and review['plan_sha256']==plan_sha,'stale source review')
    OUT.mkdir(exist_ok=False)
    write_json(OUT/'STARTED.json',{'utc':now(),'owner_pid':os.getpid(),'plan_sha256':plan_sha,
               'registration_sha256':digest(REGISTRATION),'deadline_monotonic':deadline,
               'deadline_utc':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=max(0,deadline-time.monotonic()))).isoformat()})
    proc=None; failure=None; cleanup=None; limits=plan['limits']; last_sample=None
    prior_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    interrupted=[]
    def on_signal(number, frame):
        if not interrupted:
            interrupted.append(number)
            raise RuntimeError('owner received '+signal.Signals(number).name)
    for sig in prior_handlers: signal.signal(sig,on_signal)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    try:
        def owner_guard():
            require(time.monotonic()<deadline-5,'inclusive deadline (cleanup reserve)')
            violations=resource_violations({'free_disk_bytes':shutil.disk_usage(OUT).free,
                                           'evidence_bytes':evidence_size(OUT)},limits)
            require(not violations,'owner resource '+str(violations))
        owner_guard(); verify_bindings(plan,owner_guard)
        env=dict(os.environ)
        for key in FORBIDDEN_ENV: env.pop(key,None)
        env.update(ENV); env['PYTHONPATH']=str(ROOT/plan['addon_path'])
        command=[str(ROOT/plan['python']),str(Path(__file__).resolve()),'--worker',str(PLAN),
                 '--plan-sha256',plan_sha,'--owner-pid',str(os.getpid())]
        with (OUT/'worker.log').open('xb') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(OUT/'WORKER_PID.json',{'pid':proc.pid,'pgid':proc.pid,'sid':proc.pid,'command':command})
            with (OUT/'owner-resources.jsonl').open('x') as resources:
                while True:
                    owner_guard(); timestamp=time.monotonic(); table=process_table()
                    exited=proc.poll() is not None
                    tree=process_tree(table,os.getpid())
                    rss=sum(r['rss_bytes'] for r in tree)
                    sample={'utc':now(),'monotonic':timestamp,'rss_bytes':rss,'processes':tree,
                            'evidence_bytes':evidence_size(OUT),'free_disk_bytes':shutil.disk_usage(OUT).free,
                            'sample_gap_seconds':None if last_sample is None else timestamp-last_sample}
                    last_sample=timestamp
                    resources.write(canonical(sample).decode()+'\n'); resources.flush()
                    require(not resource_violations(sample,limits),'external resource ceiling')
                    if exited: break
                    time.sleep(limits['sample_interval_seconds'])
        require(proc.returncode==0,'worker nonzero exit')
        result=read_json(OUT/'WORKER_RESULT.json')
        require(result['status']=='NATIVE_WORKER_COLLECTED_UNGRADED' and result['plan_sha256']==plan_sha,'worker did not pass')
        require(result['counts']['load_attempted']==result['counts']['load_finished']==1,'load count')
        require(result['counts']['forward_attempted']==result['counts']['forward_finished']==472,'forward count')
        require(all(result['counts'][k]==0 for k in ('generation','backward','fits','updates',
                                                   'provider_calls','tokenizer_calls','scoring_calls')),
                'unexpected operation count')
        owner_guard(); verify_bindings(plan,owner_guard)
    except BaseException as e:
        failure={'error':repr(e),'traceback':traceback.format_exc()}
    finally:
        # Cleanup has a reserved interval; do not let a second signal skip it.
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig in prior_handlers: signal.signal(sig,signal.SIG_IGN)
        if proc is not None:
            try:
                cleanup=cleanup_group(proc,deadline)
                if not cleanup['group_absent']: raise RuntimeError('owned process group survives')
            except BaseException as e:
                failure={'original_failure':failure,'cleanup_failure':repr(e)}
        if time.monotonic()>=deadline:
            failure={'original_failure':failure,'deadline_exceeded':True}
        final_resources={'evidence_bytes':evidence_size(OUT),'free_disk_bytes':shutil.disk_usage(OUT).free}
        if resource_violations(final_resources,limits):
            failure={'original_failure':failure,'final_resource_failure':final_resources}
        inventory={}
        try:
            def final_guard():
                require(time.monotonic()<deadline,'deadline during terminal inventory')
            for p in sorted(OUT.rglob('*')):
                if p.is_file(): inventory[str(p.relative_to(OUT))]={'bytes':p.stat().st_size,'sha256':digest(p,final_guard)}
        except BaseException as e:
            failure={'original_failure':failure,'inventory_incomplete':repr(e)}
        terminal={'status':'NATIVE_COLLECTION_COMPLETE_UNGRADED' if failure is None else 'NATIVE_COLLECTION_STOPPED_UNGRADED',
                  'utc':now(),'plan_sha256':plan_sha,'owner_pid':os.getpid(),
                  'elapsed_seconds':time.monotonic()-begin,'cleanup':cleanup,'failure':failure,
                  'final_resources':final_resources,'inventory':inventory,
                  'scope':'Complete raw native CPU BF16 collection only; all 472 rows retained, no scientific scoring or decoded answers.'}
        if time.monotonic()>=deadline or resource_violations(final_resources,limits):
            failure={'original_failure':failure,'terminal_commit_limit':final_resources,
                     'deadline_exceeded':time.monotonic()>=deadline}
            terminal.update(status='NATIVE_COLLECTION_STOPPED_UNGRADED',failure=failure)
        terminal.update(elapsed_seconds_before_publication=time.monotonic()-begin,owner_signals=interrupted)
        # Inventory excludes this immutable terminal receipt; worker log is closed.
        try:
            terminal=publish_terminal(OUT,terminal,deadline,limits)
            if terminal['status']!='NATIVE_COLLECTION_COMPLETE_UNGRADED': failure=terminal.get('failure') or terminal.get('publication_failure')
        finally:
            for sig,handler in prior_handlers.items(): signal.signal(sig,handler)
    print(json.dumps({'status':terminal['status'],'elapsed_seconds':terminal['elapsed_seconds']}))
    return 0 if failure is None else 1


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--owner',action='store_true'); mode.add_argument('--worker',type=Path)
    parser.add_argument('--plan-sha256'); parser.add_argument('--owner-pid',type=int)
    args=parser.parse_args()
    if args.owner: return owner()
    require(args.plan_sha256 and args.owner_pid,'worker owner arguments required')
    return worker(args.worker,args.plan_sha256,args.owner_pid)


if __name__=='__main__':
    raise SystemExit(main())
