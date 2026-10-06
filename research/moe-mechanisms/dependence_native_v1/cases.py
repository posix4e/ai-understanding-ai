"""Fixed native dependence panel: pure exact math and text, no tokenizer/runtime.

Only ``cases`` are inference inputs. ``references`` are evaluator-only and must
never be appended to a prompt. The native chat wrapper and single-token X/Y
extension proof are deliberately deferred to a separate tokenizer preflight.
"""
from fractions import Fraction as F
from itertools import product
import hashlib
import json

SCHEMA = 'dependence-native-cases-v1'
CODES = ('X', 'Y')
ASSISTANT_PREFILL = 'Answer:'
CANDIDATE_TEXT = {'X': ' X', 'Y': ' Y'}
BITS = ((0, 0), (0, 1), (1, 0), (1, 1))
MAIN_LAWS = ('positive', 'negative')
PRODUCT_LAWS = ('baseline_product', 'A_lower_only', 'A_higher_only',
                'B_lower_only', 'B_higher_only')
SENSORS = {'perfect': (F(0), F(1)),
           'positive_biased': (F(1, 2), F(1)),
           'negative_biased': (F(0), F(1, 2))}
COSTS = ('1/40', '1/20')
THRESHOLDS = ('19/40', '21/40')
BLOCKS = ('donor_first_action', 'donor_forced_post_query',
          'receiver_first_action', 'receiver_forced_post_query',
          'main_law_marginal_controls', 'product_marginal_positive_controls')
REPEAT_CELLS = ('c00', 'c05', 'c06', 'c07', 'c12', 'c16', 'c30', 'c52')
COMMON_INSTRUCTION = 'Read the evidence, then answer the task using one of its response codes.'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def joint_law(name):
    """Canonical (00,01,10,11) probabilities; no reference decisions."""
    if name == 'positive':
        values = (F(2, 5), F(1, 10), F(1, 10), F(2, 5))
    elif name == 'negative':
        values = (F(1, 10), F(2, 5), F(2, 5), F(1, 10))
    else:
        marginals = {'baseline_product': (F(1, 2), F(1, 2)),
                     'A_lower_only': (F(9, 20), F(1, 2)),
                     'A_higher_only': (F(11, 20), F(1, 2)),
                     'B_lower_only': (F(1, 2), F(9, 20)),
                     'B_higher_only': (F(1, 2), F(11, 20))}
        require(name in marginals, 'unknown law')
        pa, pb = marginals[name]
        values = tuple((pa if a else 1-pa)*(pb if b else 1-pb) for a, b in BITS)
    return dict(zip(BITS, values))


def diagnostic_math(joint, likelihood, cost, backup=F(3, 8)):
    """Exact evaluator math. Enumerate all four post-signal policies."""
    require(set(joint) == set(BITS), 'joint support keys')
    require(all(type(x) is F and x >= 0 for x in joint.values())
            and sum(joint.values()) == 1, 'exact normalized joint required')
    require(len(likelihood) == 2 and all(type(x) is F and 0 <= x <= 1
                                        for x in likelihood), 'exact likelihood')
    require(type(cost) is F and cost >= 0 and type(backup) is F and backup >= 0,
            'exact nonnegative costs')
    masses, conditional = {}, {}
    for z in (0, 1):
        weighted = {ab: p*(likelihood[ab[0]] if z else 1-likelihood[ab[0]])
                    for ab, p in joint.items()}
        masses[z] = sum(weighted.values())
        require(masses[z] > 0, 'impossible signal branch')
        conditional[z] = sum(p for (a, b), p in weighted.items() if b == 1)/masses[z]
    policies = {}
    for policy in product(('B', 'backup'), repeat=2):
        policies[','.join(policy)] = cost + sum(
            masses[z]*(conditional[z] if policy[z] == 'B' else backup) for z in (0, 1))
    best = min(policies.values())
    return {'signal_mass': masses, 'conditional_B_failure': conditional,
            'contingent_policy_losses': policies, 'query_loss': best,
            'backup_loss': backup, 'gross_query_value': backup-(best-cost)}


def semantic_cells():
    """Fixed block/factor ordering; all nonapplicable fields are null."""
    cells = []
    def add(block, law, stage, sensor=None, cost=None, signal=None,
            variable=None, threshold=None):
        cells.append({'cell_id': f'c{len(cells):02d}', 'block': block, 'law': law,
                      'stage': stage, 'sensor': sensor, 'cost': cost, 'signal': signal,
                      'variable': variable, 'threshold': threshold})
    for law in MAIN_LAWS:
        add(BLOCKS[0], law, 'first', 'perfect', COSTS[0])
    for law, signal in product(MAIN_LAWS, (0, 1)):
        add(BLOCKS[1], law, 'post', 'perfect', COSTS[0], signal)
    for law, sensor, cost in product(MAIN_LAWS, ('positive_biased', 'negative_biased'), COSTS):
        add(BLOCKS[2], law, 'first', sensor, cost)
    for law, sensor, cost, signal in product(MAIN_LAWS, ('positive_biased', 'negative_biased'), COSTS, (0, 1)):
        add(BLOCKS[3], law, 'post', sensor, cost, signal)
    for law, variable, threshold in product(MAIN_LAWS, ('A', 'B'), THRESHOLDS):
        add(BLOCKS[4], law, 'marginal', variable=variable, threshold=threshold)
    for law, variable, threshold in product(PRODUCT_LAWS, ('A', 'B'), THRESHOLDS):
        add(BLOCKS[5], law, 'marginal', variable=variable, threshold=threshold)
    require(len(cells) == 58, 'cell census')
    return cells


def validate_cell(cell):
    require(type(cell) is dict, 'cell object')
    known = {c['cell_id']: c for c in semantic_cells()}
    require(cell.get('cell_id') in known and canonical_sha(cell) == canonical_sha(known[cell['cell_id']]),
            'unknown or changed cell')


def semantic_options(cell):
    if cell['stage'] == 'first':
        return ('query', 'backup')
    if cell['stage'] == 'post':
        return ('B', 'backup')
    return ('use_variable', 'backup')


def reference(cell):
    """Evaluator-only exact fractions serialized as reduced rational strings."""
    validate_cell(cell)
    joint = joint_law(cell['law'])
    if cell['stage'] == 'marginal':
        pos = 0 if cell['variable'] == 'A' else 1
        marginal = sum(p for ab, p in joint.items() if ab[pos] == 1)
        losses = {'use_variable': marginal, 'backup': F(cell['threshold'])}
        detail = {'marginal_failure': str(marginal)}
    else:
        values = diagnostic_math(joint, SENSORS[cell['sensor']], F(cell['cost']))
        if cell['stage'] == 'first':
            losses = {'query': values['query_loss'], 'backup': values['backup_loss']}
        else:
            losses = {'B': values['conditional_B_failure'][cell['signal']],
                      'backup': values['backup_loss']}
        detail = {'signal_mass': {str(k): str(v) for k, v in values['signal_mass'].items()},
                  'conditional_B_failure': {str(k): str(v) for k, v in values['conditional_B_failure'].items()},
                  'contingent_policy_losses': {k: str(v) for k, v in values['contingent_policy_losses'].items()},
                  'gross_query_value': str(values['gross_query_value'])}
    best = min(losses.values())
    winners = [name for name in semantic_options(cell) if losses[name] == best]
    require(len(winners) == 1, 'fixed panel requires unique exact reference')
    return {**cell, 'option_losses': {k: str(v) for k, v in losses.items()},
            'winners': winners, 'unique': True,
            'absolute_loss_gap': str(abs(list(losses.values())[0]-list(losses.values())[1])),
            'details': detail}


def evidence_prefix(law, row_order):
    require(type(row_order) is int and row_order in (0, 1), 'row order')
    rows = BITS if row_order == 0 else tuple(reversed(BITS))
    joint = joint_law(law)
    table = '\n'.join(f'{a} | {b} | {joint[a,b]}' for a, b in rows)
    return (COMMON_INSTRUCTION + '\n\nEvidence\n'
            'A and B are failure indicators: 0 means works; 1 means fails.\n'
            'This table gives their joint probabilities.\n'
            'A | B | Probability\n' + table + '\n\n')


def task_text(cell):
    if cell['stage'] == 'marginal':
        variable = cell['variable']
        return (f'Task\nNo diagnostic is available. Using {variable} incurs loss 1 if {variable}=1 '
                f'and loss 0 if {variable}=0. The backup has fixed loss {cell["threshold"]}.\n'
                'Choose the option with the lower expected loss.\n')
    l0, l1 = SENSORS[cell['sensor']]
    common = ('Task\nUsing B incurs loss 1 if B=1 and loss 0 if B=0. '
              'The backup has fixed loss 3/8.\n'
              'A diagnostic produces signal Z in {0,1}. Its signal depends only on A '
              '(it is independent of B given A).\n'
              f'P(Z=1 | A=0)={l0}; P(Z=1 | A=1)={l1}. '
              'For each A, P(Z=0 | A)=1-P(Z=1 | A).\n')
    if cell['stage'] == 'first':
        return (common + f'Requesting the diagnostic costs {cell["cost"]}. No signal has been observed.\n'
                'If you request it, pay that cost, observe Z, then choose B or the backup '
                'to minimize conditional expected loss. If you skip it, take the backup without paying the diagnostic cost.\n'
                'Choose the first action with the lower expected total loss.\n')
    return (common + f'The diagnostic was already requested at cost {cell["cost"]} '
            f'and produced Z={cell["signal"]}. That cost is already paid and is sunk.\n'
            'Choose B or the backup to minimize the remaining conditional expected loss. '
            'No further diagnostic is available.\n')


def render(cell, row_order, mapping, option_order):
    validate_cell(cell)
    require(all(type(x) is int and x in (0, 1) for x in (row_order, mapping, option_order)),
            'nuisance factors')
    semantics = semantic_options(cell)
    meanings = dict(zip(CODES, semantics if mapping == 0 else tuple(reversed(semantics))))
    displayed = semantics if option_order == 0 else tuple(reversed(semantics))
    descriptions = {'query': 'Request the diagnostic, then choose B or the backup after observing Z.',
                    'backup': ('Skip the diagnostic and take the backup.' if cell['stage'] == 'first'
                               else 'Take the backup.'),
                    'B': 'Use B.', 'use_variable': f'Use {cell["variable"]}.'}
    choices = '\n'.join(f'{next(code for code in CODES if meanings[code] == semantic)}: '
                        f'{descriptions[semantic]}' for semantic in displayed)
    prefix = evidence_prefix(cell['law'], row_order)
    suffix = task_text(cell) + 'Options\n' + choices + '\nReturn only X or Y.'
    text = prefix + suffix
    return {'row_order': row_order, 'mapping': mapping, 'option_order': option_order,
            'variant_id': f'r{row_order}-m{mapping}-o{option_order}',
            'answer_meanings': meanings, 'option_display_order': list(displayed),
            'evidence_prefix': prefix, 'task_suffix': suffix, 'user_text': text,
            'evidence_prefix_sha256': sha_text(prefix), 'task_suffix_sha256': sha_text(suffix),
            'user_text_sha256': sha_text(text), 'evidence_prefix_utf8_bytes': len(prefix.encode()),
            'assistant_prefill': ASSISTANT_PREFILL, 'candidate_text': dict(CANDIDATE_TEXT)}


def build_panel():
    """472 deterministic cases and separate exact references; no I/O or runtime."""
    cells = semantic_cells()
    cases, references = [], {}
    def append(cell, row_order, mapping, option_order, replicate=0, repeat_of=None):
        rendering = render(cell, row_order, mapping, option_order)
        original_id = f'{cell["cell_id"]}-{rendering["variant_id"]}'
        case_id = original_id if replicate == 0 else original_id + '-repeat'
        row = {'index': len(cases), 'id': case_id, **cell, **rendering,
               'replicate': replicate, 'repeat_of': repeat_of}
        ref = reference(cell)
        refs = {**ref, 'case_id': case_id,
                'winning_codes': [code for code in CODES if row['answer_meanings'][code] in ref['winners']]}
        cases.append(row); references[case_id] = refs
    for cell in cells:
        for factors in product((0, 1), repeat=3):
            append(cell, *factors)
    by_id = {c['cell_id']: c for c in cells}
    for cell_id in REPEAT_CELLS:
        append(by_id[cell_id], 0, 0, 0, 1, f'{cell_id}-r0-m0-o0')
    require(len(cases) == 472 and len(references) == 472, 'complete panel census')
    return {'schema': SCHEMA, 'cells': cells, 'cases': cases,
            'references': references,
            'counts': {'semantic_cells': 58, 'variants_per_cell': 8,
                       'primary': 464, 'repeats': 8, 'total': 472},
            'cases_sha256': canonical_sha(cases), 'references_sha256': canonical_sha(references)}
