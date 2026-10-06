"""Create-only prospective case archive and tokenizer-only native input preflight.

Neither command loads a language model, scores responses or contacts a provider.
The evaluator reference archive is never included in the rendered messages.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name('cases.py')
OUT = ROOT / 'work/dependence-native-panel-inputs'
TOK = OUT / 'tokenizer-v1'
MODEL = ROOT / 'work/dependence-controller-acquisition/model'
CASE_SOURCE_SHA = '34c04f74bbd2b9a7462ad9f4c5bd71473a141f0be6f302384d74d8725fa7a747'
SCORE_CONTRACT = ROOT / 'work/dependence-native-panel-preparation/SCORE_CONTRACT.json'
SCORE_SHA = 'd730df43db53254c53f76e26d30b7cc22b87dc40d9fc58249cf5e824b300b997'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build():
    require(sha(SOURCE) == CASE_SOURCE_SHA, 'Case source changed')
    require(sha(SCORE_CONTRACT) == SCORE_SHA, 'Scoring contract changed')
    from cases import build_panel
    panel = build_panel()
    require(panel['counts'] == dict(semantic_cells=58, variants_per_cell=8,
                                    primary=464, repeats=8, total=472), 'Panel census')
    OUT.mkdir(exist_ok=False)
    json_write(OUT/'CASES.json', {k: panel[k] for k in
        ('schema', 'counts', 'cells', 'cases', 'cases_sha256')})
    json_write(OUT/'REFERENCES.json', {'scope': 'EVALUATOR_ONLY_NEVER_RENDER',
        'references': panel['references'], 'references_sha256': panel['references_sha256']})
    manifest = {'status': 'CASE_ARCHIVE_ONLY_NO_TOKENIZER_OR_MODEL', 'at_utc': stamp(),
        'source_sha256': sha(__file__), 'case_source_sha256': sha(SOURCE),
        'score_contract_sha256': sha(SCORE_CONTRACT), 'counts': panel['counts'],
        'files': {name: {'bytes': (OUT/name).stat().st_size, 'sha256': sha(OUT/name)}
                  for name in ('CASES.json', 'REFERENCES.json')},
        'model_loads': 0, 'forward_calls': 0, 'tokenizer_calls': 0, 'fits': 0,
        'provider_calls': 0, 'scientific_registration': False}
    json_write(OUT/'BUILD_MANIFEST.json', manifest)
    print(json.dumps({'status': manifest['status'], 'sha256': sha(OUT/'BUILD_MANIFEST.json')}))


def fixed_date(fmt):
    return dt.datetime(2026, 10, 6).strftime(fmt)


def tokenize():
    require(sha(SOURCE) == CASE_SOURCE_SHA and sha(SCORE_CONTRACT) == SCORE_SHA,
            'Prospective source/score contract changed')
    manifest = json.loads((OUT/'BUILD_MANIFEST.json').read_text())
    require(manifest['source_sha256'] == sha(__file__), 'Input preparer changed after build')
    for name, rec in manifest['files'].items():
        require((OUT/name).stat().st_size == rec['bytes'] and sha(OUT/name) == rec['sha256'],
                'Case archive changed')
    cases = json.loads((OUT/'CASES.json').read_text())['cases']
    require(len(cases) == 472 and [r['index'] for r in cases] == list(range(472)), 'Input order')
    TOK.mkdir(exist_ok=False)
    json_write(TOK/'STARTED.json', {'at_utc': stamp(), 'pid': os.getpid(),
        'source_sha256': sha(__file__), 'build_manifest_sha256': sha(OUT/'BUILD_MANIFEST.json'),
        'scope': 'Tokenizer only; no model loads, task forwards, fits or provider calls.'})
    rows_written = 0
    try:
        for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                          'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
                          'HF_HUB_DISABLE_TELEMETRY': '1',
                          'TOKENIZERS_PARALLELISM': 'false'}.items():
            os.environ[key] = value
        for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
            os.environ.pop(key, None)
        import transformers
        from transformers import AutoTokenizer
        require(transformers.__version__ == '4.56.2', 'Tokenizer library version')
        acquisition = json.loads((ROOT/'work/dependence-controller-acquisition/COMPLETE.json').read_text())
        token_assets = [r for r in acquisition['assets'] if not r['path'].endswith('.safetensors')]
        require(len(token_assets) == 7, 'Tokenizer/config asset census')
        for rec in token_assets:
            path = ROOT/rec['path']
            require(path.stat().st_size == rec['bytes'] and sha(path) == rec['sha256'], 'Tokenizer asset changed')
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True,
                                                   token=False, trust_remote_code=False)
        require(tokenizer.is_fast, 'Offset-capable native tokenizer required')
        require(tokenizer.chat_template == (MODEL/'chat_template.jinja').read_text(), 'Native template changed')
        groups, code_pairs, lengths, by_id = {}, set(), [], {}
        kwargs = dict(add_generation_prompt=True, enable_thinking=False, strftime_now=fixed_date)
        with (TOK/'INPUTS.jsonl').open('x') as inputs, (TOK/'RENDERED.jsonl').open('x') as rendered:
            for case in cases:
                text = case['user_text']
                require(text == case['evidence_prefix']+case['task_suffix'], 'Prompt decomposition')
                require(case['assistant_prefill'] == 'Answer:' and
                        case['candidate_text'] == {'X':' X','Y':' Y'}, 'Fixed response context')
                messages = [{'role':'user', 'content':text}]
                native = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
                native_ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
                require(native_ids == tokenizer.encode(native, add_special_tokens=False), 'Native chat token parity')
                require(native.count(text) == 1, 'User content placement')
                require('Today Date: 06 October 2026' in native and 'Reasoning Mode: /no_think' in native,
                        'Fixed native framing')
                require(native.endswith('<|im_start|>assistant\n<think>\n\n</think>\n'), 'Native assistant boundary')
                prompt = native + case['assistant_prefill']
                encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
                ids = encoded['input_ids']; offsets = encoded['offset_mapping']
                require(tokenizer.decode(ids, skip_special_tokens=False) == prompt, 'Complete roundtrip')
                require(len(ids) > 1 and min(ids) >= 0 and max(ids) < 128256, 'Token vocabulary')
                prefix_end = prompt.index(text) + len(case['evidence_prefix'])
                require(not any(start < prefix_end < end for start, end in offsets), 'Token crosses evidence boundary')
                boundary = next(i for i,(start,end) in enumerate(offsets) if start >= prefix_end)
                require(boundary > 0 and offsets[boundary-1][1] == prefix_end,
                        'Exact evidence-prefix token boundary')
                prefix_ids = ids[:boundary]
                require(tokenizer.decode(prefix_ids, skip_special_tokens=False) == prompt[:prefix_end],
                        'Prefix roundtrip')
                group = case['law'] + '-r' + str(case['row_order'])
                identity = {'text': prompt[:prefix_end], 'ids': prefix_ids}
                if group in groups:
                    require(groups[group] == identity, 'Late context changed captured prefix')
                else:
                    groups[group] = identity
                candidate_ids = {}
                for code in ('X','Y'):
                    extended = tokenizer.encode(prompt+case['candidate_text'][code], add_special_tokens=False)
                    require(len(extended) == len(ids)+1 and extended[:-1] == ids,
                            'Candidate is not exactly one unchanged-prefix next token')
                    require(tokenizer.decode(extended, skip_special_tokens=False) == prompt+case['candidate_text'][code],
                            'Candidate extension roundtrip')
                    candidate_ids[code] = extended[-1]
                require(candidate_ids['X'] != candidate_ids['Y'], 'Candidate collision')
                code_pairs.add(tuple(candidate_ids[k] for k in ('X','Y')))
                record = {k: case[k] for k in ('index','id','cell_id','variant_id','replicate','repeat_of',
                                              'user_text_sha256','evidence_prefix_sha256')}
                record.update(input_ids=ids, attention_mask=[1]*len(ids), token_count=len(ids),
                    candidate_ids=candidate_ids, rendered_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    evidence_prefix_token_ids=prefix_ids, evidence_prefix_last_token_index=boundary-1,
                    evidence_prefix_rendered_utf8_bytes=len(prompt[:prefix_end].encode()))
                if case['repeat_of'] is not None:
                    original = by_id[case['repeat_of']]
                    for key in ('input_ids','attention_mask','candidate_ids','rendered_sha256'):
                        require(record[key] == original[key], 'Repeat input changed')
                by_id[case['id']] = record
                inputs.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
                rendered.write(json.dumps({'id':case['id'], 'prompt':prompt}, sort_keys=True)+'\n')
                lengths.append(len(ids)); rows_written += 1
            inputs.flush(); os.fsync(inputs.fileno()); rendered.flush(); os.fsync(rendered.fileno())
        require(rows_written == 472 and len(groups) == 14 and len(code_pairs) == 1, 'Final tokenizer census')
        for rec in token_assets:
            require(sha(ROOT/rec['path']) == rec['sha256'], 'Tokenizer asset changed during preflight')
        result = {'status':'PASS_TOKENIZER_ONLY_NATIVE_PANEL_INPUTS', 'at_utc':stamp(),
            'source_sha256':sha(__file__), 'case_source_sha256':sha(SOURCE),
            'build_manifest_sha256':sha(OUT/'BUILD_MANIFEST.json'), 'score_contract_sha256':sha(SCORE_CONTRACT),
            'tokenizer_class':type(tokenizer).__name__, 'transformers_version':transformers.__version__,
            'native_template_sha256':sha(MODEL/'chat_template.jinja'), 'native_template_unchanged':True,
            'fixed_date':'2026-10-06', 'enable_thinking':False, 'assistant_prefill':'Answer:',
            'candidate_text':{'X':' X','Y':' Y'}, 'candidate_token_ids':dict(zip(('X','Y'),next(iter(code_pairs)))),
            'rows':rows_written, 'shared_evidence_prefix_groups':len(groups),
            'token_lengths':{'minimum':min(lengths),'maximum':max(lengths),'sum':sum(lengths),
                             'mean':sum(lengths)/len(lengths)},
            'all_complete_roundtrips':True, 'all_one_token_extensions':True,
            'all_evidence_prefixes_identical_across_late_contexts':True, 'all_repeats_input_exact':True,
            'truncation':False, 'files':{name:{'bytes':(TOK/name).stat().st_size,'sha256':sha(TOK/name)}
                for name in ('STARTED.json','INPUTS.jsonl','RENDERED.jsonl')},
            'model_loads':0,'forward_calls':0,'generated_tokens':0,'fits':0,'provider_calls':0,
            'limit':'Tokenization and input identity only; no scientific behavior, CPU capacity or competence established.'}
        json_write(TOK/'COMPLETE.json', result)
        print(json.dumps({'status':result['status'], 'sha256':sha(TOK/'COMPLETE.json'),
                          'token_lengths':result['token_lengths'], 'candidate_token_ids':result['candidate_token_ids']}))
    except BaseException as error:
        json_write(TOK/'STOPPED.json', {'status':'TOKENIZER_PREFLIGHT_FAILED_NO_MODEL_CALLS',
            'at_utc':stamp(),'source_sha256':sha(__file__),'completed_rows':rows_written,
            'error':repr(error),'traceback':traceback.format_exc(),'model_loads':0,'forward_calls':0,'provider_calls':0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('build','tokenize'))
    args = parser.parse_args()
    (build if args.command == 'build' else tokenize)()
