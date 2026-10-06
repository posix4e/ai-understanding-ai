"""Build the public acquisition report from independently checked, saved results.

This script never trains, selects observations, or alters scientific evidence.
The caller must explicitly request publication after the independent checks pass.
It does not edit the site's shared evidence manifest or other HTML pages.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from html.parser import HTMLParser
from pathlib import Path
import zipfile
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'research/world-model-acquisition-v1'
OUT = ROOT / 'outputs/world-model-planning/ACQUISITION_V1'
DIAGNOSTIC = ROOT / 'outputs/world-model-planning/ACQUISITION_DIAGNOSTIC_V1'
ARMS = ('random', 'uncertainty', 'decision', 'replay')
LABELS = {
    'random': 'Random choice',
    'uncertainty': 'Most uncertain move',
    'decision': 'Decision sensitivity',
    'replay': 'No new observations',
}
BUDGETS = (0, 1, 2, 4)


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def deterministic_zip(destination: Path, paths: list[Path]) -> dict:
    """Preserve exact source bytes; stable archive metadata makes re-runs identical."""
    files = sorted(set(p.resolve() for p in paths if p.is_file()))
    if not files:
        raise ValueError(f'No files for {destination.name}')
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            record = zipfile.ZipInfo(name, date_time=(2026, 10, 6, 0, 0, 0))
            record.compress_type = zipfile.ZIP_DEFLATED
            record.external_attr = 0o100644 << 16
            archive.writestr(record, path.read_bytes())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        for path in files:
            assert archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes()
    return {
        'file': destination.name, 'sha256': sha(destination),
        'bytes': destination.stat().st_size, 'entries': len(files),
        'source_files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
    }


def pct(number: float) -> str:
    return f'{100 * number:.2f}%'


def difference(number: float) -> str:
    return f'{100 * number:+.2f}'


def table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    body = []
    for row in rows:
        body.append('<tr><th scope="row">' + row[0] + '</th>' + ''.join('<td>' + v + '</td>' for v in row[1:]) + '</tr>')
    return '<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable results table"><table><caption>' + caption + '</caption><thead><tr>' + ''.join('<th scope="col">' + html.escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'


CSS = '''
.acquisition-hero{padding:65px 0 35px;max-width:950px}.acquisition-hero h1{max-width:850px}.acquisition-hero .hero-deck{max-width:820px}.acquisition-body{max-width:1000px;padding-bottom:65px}.acquisition-body h2{font-size:36px;line-height:1.15;margin:42px 0 22px}.acquisition-body h3{font-size:22px}.acquisition-body p{max-width:850px}.finding{background:var(--soft);border-left:4px solid var(--teal);padding:24px;max-width:900px}.finding p:last-child{margin-bottom:0}.table-scroll{overflow-x:auto;margin:25px 0}table{border-collapse:collapse;width:100%;font-size:14px}caption{text-align:left;color:var(--muted);margin-bottom:12px}th,td{border-bottom:1px solid var(--line);padding:12px 10px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:var(--soft)}.meta{color:var(--muted);font-size:13px}.acquisition-body li{margin-bottom:9px}.process{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.process article{background:#fcfaf5;border:1px solid var(--line);padding:20px}.process h3{font-size:18px}.process p{font-size:15px;margin-bottom:0}.evidence-list{columns:2;column-gap:40px}.evidence-list li{break-inside:avoid}details{border:1px solid var(--line);padding:15px 20px;margin:25px 0}summary{cursor:pointer;font-weight:600}.table-scroll:focus-visible{outline:3px solid #aa6833;outline-offset:4px}.source-hash{font-family:ui-monospace,monospace;font-size:11px;overflow-wrap:anywhere}@media(max-width:760px){.acquisition-hero{padding-top:38px}.acquisition-body h2{font-size:29px}.process{grid-template-columns:1fr}.evidence-list{columns:1}.finding{padding:18px}.site-header nav{gap:12px}}@media print{.acquisition-hero{padding-top:20px}.table-scroll{overflow:visible}table{font-size:10pt}th,td{padding:8px 5px}.process{display:block}.process article{margin-bottom:10px}}
'''


def render_page(summary: dict, artifacts: list[dict]) -> str:
    rates = summary['evaluation']['mean_navigation_rates']
    rows = summary['evaluation']['maps']
    delta = summary['evaluation']['primary_mean_difference']
    decision = rates['decision']['4']
    random = rates['random']['4']
    direction = 'higher' if delta > 0 else 'lower' if delta < 0 else 'unchanged'
    delta_sentence = ('The average difference was zero.' if delta == 0 else
        f'That is {abs(100 * delta):.2f} percentage points {direction} on these eight maps.')
    wins = sum(row['decision_minus_random'] > 1e-12 for row in rows)
    losses = sum(row['decision_minus_random'] < -1e-12 for row in rows)
    ties = len(rows) - wins - losses
    result_table = table(
        ['Learning round', *[LABELS[a] for a in ARMS]],
        [[str(b), *[pct(rates[a][str(b)]) for a in ARMS]] for b in BUDGETS],
        'Navigation success on the eight reserved maps. Each value gives equal weight to maps, model types and seeds. The three query methods receive one new observation per round. The no-observation control receives only extra training.')
    map_table = table(
        ['Evaluation map', 'Random', 'Uncertainty', 'Decision sensitivity', 'No observations', 'Decision − random (points)'],
        [[html.escape(row['map_id']), *[pct(row['rates'][a]) for a in ARMS], difference(row['decision_minus_random'])] for row in rows],
        'Every reserved evaluation map, after four learning rounds. Each row averages two model types and three seeds. These eight maps are the units of comparison; the thousands of navigation tasks are not independent experiments.')
    engineering = summary['engineering']['maps']
    engineering_table = table(
        ['Engineering map', 'Random', 'Uncertainty', 'Decision sensitivity', 'No observations', 'Decision − random (points)'],
        [[html.escape(row['map_id']), *[pct(row['rates'][a]) for a in ARMS], difference(row['decision_minus_random'])] for row in engineering],
        'The four engineering maps are shown separately. They are excluded from the main comparison.')
    type_rows = []
    for kind, label in [('direct', 'Ordinary predictor'), ('energy', 'Energy scorer')]:
        by_type = summary['evaluation']['by_model_type'][kind]
        type_rows.append([label, *[pct(by_type['rates'][a]) for a in ARMS], difference(by_type['decision_minus_random'])])
    model_type_table = table(
        ['Model type', 'Random', 'Uncertainty', 'Decision sensitivity', 'No observations', 'Decision − random (points)'],
        type_rows,
        'Navigation after four learning rounds, separated by model type. Each row averages the same eight maps and three seeds; computation is not matched.')
    secondary = summary['evaluation']['prediction_audit_round4']
    timing = summary['evaluation']['timing_totals']
    secondary_table = table(
        ['Method', 'Untouched-move accuracy', 'Mean log loss', 'Choice time (seconds)', 'Extra training (seconds)'],
        [[LABELS[a], pct(secondary[a]['accuracy']), f"{secondary[a]['mean_capped_nll']:.3f}", f"{timing[a]['query_choice_seconds']:.2f}", f"{timing[a]['training_seconds']:.2f}"] for a in ARMS],
        'Secondary results after round four on the reserved maps. Lower log loss is better. Times sum all 192 learning rounds per method across 48 fits on the same local machine; choice time includes input loading and logging preparation. The no-observation control has only bookkeeping in that column.')
    d = summary['diagnostic']
    two_by_two = table(
        ['Predictions used', 'True state after each move', 'Model state after each move'],
        [['Full probability distribution', f"{d['execution_2x2']['soft_feedback']['successes']} / 600", f"{d['execution_2x2']['soft_blind']['successes']} / 600"],
         ['Only the most likely next state', f"{d['execution_2x2']['argmax_feedback']['successes']} / 600", f"{d['execution_2x2']['argmax_blind']['successes']} / 600"]],
        'Exploratory analysis of the saved ordinary predictor, seed 11, on the original map. No model was trained again. Each cell uses the same 600 tasks and 16-step look-ahead.')
    evidence = ''.join(f'<li><a href="./research-evidence/{html.escape(a["file"])}">{html.escape(a["label"])}</a> <span class="meta">({a["bytes"] / 1_048_576:.2f} MiB)</span></li>' for a in artifacts)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A checked study of which four observations a small world model should learn. All eight evaluation maps and the earlier planning surprise are reported.">
<title>Which four moves should a model learn? | AI Understanding AI</title>
<link rel="canonical" href="https://posix4e.github.io/ai-understanding-ai/world-model-acquisition.html"><link rel="stylesheet" href="./styles.css?v=3"><style>{CSS}</style></head>
<body><a class="skip-link" href="#main">Skip to content</a><header class="site-header wrap"><a class="wordmark" href="./index.html">AI UNDERSTANDING AI<span class="wordmark-dot">.</span></a><nav aria-label="Main navigation"><a href="./research.html">Research</a><a href="./world-model-pilot.html">First world-model test</a><a href="./research-review-2026-10-06.html">Paper review</a></nav></header>
<main class="wrap" id="main"><section class="acquisition-hero"><p class="eyebrow">World models / 6 October 2026</p><h1>Which four moves should a model learn?</h1><p class="hero-deck">A model cannot check everything. We tested whether it can choose a few useful observations, learn from them, and then reach more goals.</p><p class="meta">12 new maps · 4 engineering maps · 8 reserved evaluation maps · Independent checks passed · $0 provider spending</p></section><div class="acquisition-body">
<div class="finding"><p><strong>The main result:</strong> after four observations, decision sensitivity solved {pct(decision)} of navigation tasks. Random choice solved {pct(random)}. {delta_sentence}</p><p>Decision sensitivity did better on {wins} maps, worse on {losses}, and tied on {ties}. This is a description of these maps, not a test of a general advantage.</p></div>
<h2>One small world, three ways to choose</h2><p>Each world is a 5 × 5 grid with a key, a door and a wall. We changed their layout, then trained new models for each map. This tests learning within a map. It does not test transfer of one trained model to a new map.</p>
<div class="process"><article><h3>1. Learn some moves</h3><p>Each model starts with 72 known transitions. A transition is one action and the state it produces.</p></article><article><h3>2. Ask four times</h3><p>Choose from 24 other transitions. Reset to the chosen state for free, take the action, and learn the result.</p></article><article><h3>3. Test navigation</h3><p>After learning is complete, test all 600 start–goal pairs. Keep another 24 transitions out of training and queries.</p></article></div>
<p><strong>Random choice</strong> follows a fixed random order. <strong>Uncertainty</strong> chooses the move with the most spread-out prediction. <strong>Decision sensitivity</strong> chooses a move whose possible outcomes could change an action choice, weighted by where the current model expects the agent to go.</p><p>The last method uses its own predictions. A model that is confidently wrong may fail to ask about an important move. Its score estimates the effect of receiving an exact answer; it does not predict how much neural training will improve.</p><p>A fourth control gets the same number of extra training updates but no new observations. All methods learn through weight updates. We do not replace their predictions with the simulator’s correct answers.</p>
<h2>What changed as models learned?</h2>{result_table}<p>Each observation is followed by 250 training updates. The query methods mix 16 original examples with 16 examples from observations collected so far. The control uses 32 original examples. Checkpoints were saved before final navigation and untouched-transition grading.</p>
<h3>Does the model type change the result?</h3>{model_type_table}
<details><summary>Untouched-move prediction and computation time</summary>{secondary_table}<p class="meta">The probability floor for log loss is the smallest positive normal float64 value; the exact-zero counts are preserved in the evidence.</p></details>
<h2>Show every evaluation map</h2>{map_table}<details><summary>Show the four engineering maps</summary>{engineering_table}</details>
<p>The maps and splits were fixed before fitting. The first four maps checked the implementation. The remaining eight were evaluated after the implementation freeze. We report all maps and all three seeds for both model types in the downloadable records.</p>
<h2>Why did the earlier model navigate well?</h2><p>In the <a href="./world-model-pilot.html">first test</a>, the ordinary predictor with seed 11 solved all 600 tasks, although it got only 11 of 24 held-out moves right. We examined its saved predictions to separate prediction errors from action choices.</p>
{two_by_two}
<p>{html.escape(d['execution_text'])}</p>
<p>{html.escape(d['census_text'])}</p><p>{html.escape(d['repair_text'])}</p>
<p>With true-state feedback, the planner sees where the agent actually arrived after each action. Without feedback, it advances an internal state using its most likely predicted successor. We stop blind execution on reaching the true goal, repeating the joint true and internal state, or taking 40 actions. These are controlled changes to the saved model and planner, not proof of a unique hidden mechanism.</p>
<h2>What this does and does not show</h2><ul>
<li>These are small, fully observed symbolic worlds. They are not visual JEPA models or language-model components.</li>
<li>The ordinary predictor and energy scorer use different model forms but the same categorical likelihood objective. Their initial training examples, update counts and seeds are paired. Selected observations can differ. Their computation is not equal.</li>
<li>Both models receive the valid-state vocabulary. That list reveals some world structure. Choosing a query has no travel cost, so this is not an embodied exploration test.</li>
<li>The decision-sensitivity score uses predicted visits over 40 steps and omits the evaluator’s cycle-stopping rule. It is an information-choice heuristic, not an optimal value-of-information calculation.</li>
<li>Eight reserved maps support a descriptive comparison. We make no statistical-significance, novelty or general energy-model advantage claim.</li>
<li>The earlier-predictor analysis was chosen after seeing the pilot. Its exact-rule repairs are diagnostic interventions, not learned improvements.</li>
<li>Independent code checked the new study’s selections and metrics, recomputed all 568,800 navigation episodes, and reproduced predictions from all 1,224 saved checkpoints. It did not replay training gradients. A separate author also recomputed all 17,400 earlier-model diagnostic episodes from the saved predictions. This checks the calculations; it does not repeat training or provide new experimental data.</li>
<li>Earlier studies and their STOP records remain closed; GPT-2 work remains paused.</li></ul>
<h2>Read and check the evidence</h2><p>Each map bundle contains its saved experiment records. The shared bundle contains frozen sources, plans and audit records. The separate diagnostic bundle preserves the earlier-model analysis.</p><ul class="evidence-list">{evidence}</ul><p><a href="./research-evidence/world-model-acquisition-hashes.json">SHA-256 hashes and original source paths</a> · <a href="./research-evidence/manifest.json">Site evidence index</a></p>
<p class="meta">Provider spending for this study: $0. The useful outcome is a checked comparison, including failures. A larger study would be needed before making a broad research claim.</p></div></main></body></html>'''


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_inventory(directory: Path, manifest: dict) -> None:
    for relative, expected in manifest.items():
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory.resolve()), 'Unsafe evidence path')
        require(path.is_file() and sha(path) == expected, f'Changed evidence: {relative}')


def normalize_phase(phase: str, report: dict) -> dict:
    expected_maps = 4 if phase == 'engineering' else 8
    expected_models = {f'{kind}_{seed}' for kind in ('direct', 'energy') for seed in (11, 29, 47)}
    require(report['phase'] == phase and len(report['maps']) == expected_maps, 'Unexpected phase size')
    result = {'maps': [], 'mean_navigation_rates': {a: {} for a in ARMS}, 'prediction_audit_round4': {}, 'timing_totals': {}, 'by_model_type': {}}
    all_rates = {a: {str(b): [] for b in BUDGETS} for a in ARMS}
    expected_ids = [f'map_{i:02d}' for i in (range(4) if phase == 'engineering' else range(4, 12))]
    require([m['map'] for m in report['maps']] == expected_ids, 'Missing or reordered maps')
    for m in report['maps']:
        require(set(m['models']) == expected_models, 'Incomplete model panel')
        row = {'map_id': m['map'], 'rates': {}}
        for a in ARMS:
            for b in BUDGETS:
                scores = []
                for fit in m['models'].values():
                    item = fit['base'] if b == 0 else fit['arms'][a][str(b)]
                    metric = item['planning']
                    require(metric['n'] == 600 and 0 <= metric['successes'] <= 600, 'Invalid navigation count')
                    rate = metric['successes'] / 600
                    require(math.isclose(rate, metric['success_rate'], rel_tol=0, abs_tol=1e-12), 'Invalid reported rate')
                    scores.append(rate)
                mean = sum(scores) / len(scores)
                all_rates[a][str(b)].append(mean)
                if b == 4:
                    row['rates'][a] = mean
        row['decision_minus_random'] = row['rates']['decision'] - row['rates']['random']
        require(math.isclose(row['decision_minus_random'], m['primary_difference'], rel_tol=0, abs_tol=1e-12), 'Invalid map difference')
        result['maps'].append(row)
    for a in ARMS:
        for b in BUDGETS:
            values = all_rates[a][str(b)]
            result['mean_navigation_rates'][a][str(b)] = sum(values) / len(values)
    result['primary_mean_difference'] = sum(r['decision_minus_random'] for r in result['maps']) / expected_maps
    require(math.isclose(result['primary_mean_difference'], report['primary_mean_difference'], rel_tol=0, abs_tol=1e-12), 'Invalid primary mean')
    for kind in ('direct', 'energy'):
        means = {a: sum(fit['arms'][a]['4']['planning']['success_rate'] for m in report['maps'] for name, fit in m['models'].items() if name.startswith(kind+'_')) / (expected_maps*3) for a in ARMS}
        result['by_model_type'][kind] = {'rates': means, 'decision_minus_random': means['decision']-means['random']}
    for arm in ARMS:
        values = [fit['arms'][arm]['4']['prediction_audit'] for m in report['maps'] for fit in m['models'].values()]
        require(all(v['n'] == 24 for v in values), 'Incomplete untouched-transition audit')
        result['prediction_audit_round4'][arm] = {'accuracy': sum(v['correct']/24 for v in values)/len(values), 'mean_capped_nll': sum(v['capped_nll'] for v in values)/len(values)}
        query_paths = sorted((OUT/phase).glob(f'map_*/*/{arm}/round_*_query.json'))
        training_paths = sorted((OUT/phase).glob(f'map_*/*/{arm}/round_*_training.json'))
        require(len(query_paths) == len(training_paths) == expected_maps*6*4, 'Incomplete timing records')
        result['timing_totals'][arm] = {'query_choice_seconds': sum(load(p)['seconds'] for p in query_paths), 'training_seconds': sum(load(p)['seconds'] for p in training_paths), 'learning_rounds': len(query_paths)}
    result['seconds'] = report['seconds']
    result['peak_rss_bytes'] = report['peak_rss_bytes']
    return result


def describe_diagnostic(diagnostic: dict) -> dict:
    out = dict(diagnostic)
    census = diagnostic['native_census']
    agreement = census['bfs_optimal_action']
    used = census['groups']['on_policy_unique']
    unused = census['groups']['off_policy_unique']
    error_label = 'a wrong prediction' if used['ties'] == unused['ties'] == 0 else 'a wrong or tied prediction'
    next_state_phrase = 'its most likely next state was wrong' if census['groups']['execution_weighted']['ties'] == 0 else 'it failed to rank the true next state alone in first place'
    # Every numerator and denominator comes from the preserved action census.
    out['census_text'] = (
        f"Among {used['n']} distinct moves used in successful paths, {used['n'] - used['correct']} had {error_label}. Among {unused['n']} unused moves, {unused['n'] - unused['correct']} did. "
        f"The model made {agreement['executed_n']:,} moves across the 600 successful paths. "
        f"For {agreement['incorrect_prediction_executed_n']:,} of those moves, {next_state_phrase}. "
        f"Yet {agreement['incorrect_prediction_bfs_optimal']:,} of those moves used an action on a shortest path to the goal. "
        f"Overall, {agreement['executed_correct']:,} of {agreement['executed_n']:,} executed actions were shortest-path actions. "
        "Getting the next state wrong and choosing a bad action are different errors."
    )
    singletons = [v for k, v in diagnostic['repairs'].items() if k.startswith('singleton_')]
    require(len(singletons) == 24 and all(v['n'] == 600 for v in singletons), 'Incomplete singleton repair panel')
    minimum = min(v['successes'] for v in singletons)
    maximum = max(v['successes'] for v in singletons)
    harmed = sum(v['paired_vs_soft_feedback']['success_to_failure'] > 0 for v in singletons)
    joint = diagnostic['repairs']['joint_all24']['successes']
    native = diagnostic['execution_2x2']['soft_feedback']['successes']
    full_distribution_matters = native > diagnostic['execution_2x2']['argmax_feedback']['successes']
    feedback_matters = native > diagnostic['execution_2x2']['soft_blind']['successes']
    out['execution_text'] = ('Both the full probability distribution and fresh observations of the real state mattered in this saved system: removing either reduced navigation success.' if full_distribution_matters and feedback_matters else 'The table separates the effects of replacing distributions by their most likely states and replacing true-state feedback by model-state updates.')
    out['repair_text'] = (
        f"We also corrected each of the 24 held-out predictions, one at a time. "
        f"Navigation success ranged from {minimum} to {maximum} of 600 tasks; "
        f"{harmed} of the 24 single corrections turned at least one success into a failure. "
        f"Correcting all 24 together gave {joint} successes. "
        "A local correction can change routes elsewhere. These results test interactions in this model and planner; they do not identify a unique learned mechanism."
    )
    return out


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.duplicate_ids = [], set(), []
        self.h1s = 0
        self.captions = 0
        self.tables = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                self.duplicate_ids.append(attrs['id'])
            self.ids.add(attrs['id'])
        if 'href' in attrs:
            self.links.append(attrs['href'])
        self.h1s += tag == 'h1'
        self.captions += tag == 'caption'
        self.tables += tag == 'table'


def validate_html(path: Path) -> dict:
    parser = PageParser()
    parser.feed(path.read_text())
    require(not parser.duplicate_ids and parser.h1s == 1, 'Invalid document headings/IDs')
    require(parser.tables == parser.captions, 'Table missing caption')
    local_links = 0
    for href in parser.links:
        if href.startswith(('https://', 'http://', 'mailto:')):
            continue
        parsed = urlsplit(href)
        file_part, anchor = unquote(parsed.path), parsed.fragment
        target = (path.parent / file_part).resolve() if file_part else path
        require(target.is_file(), 'Broken local link: ' + href)
        if anchor:
            p = PageParser()
            p.feed(target.read_text())
            require(anchor in p.ids, 'Broken fragment: ' + href)
        local_links += 1
    return {'local_links_checked': local_links, 'tables_with_captions': parser.tables, 'single_h1': True}



def verify_weight_review(phase: str) -> Path:
    work = ROOT/'work/world-model-acquisition-v1'
    path = work/f'{phase.upper()}_WEIGHT_REVIEW.json'
    review = load(path)
    require(review['status'] == 'PASS_SAVED_WEIGHT_INFERENCE_REPLAY' and review['phase'] == phase, 'Saved-weight check has not passed')
    bindings = {
        OUT/'plan.json': review['plan_sha256'],
        OUT/phase/'COLLECTED.json': review['collected_sha256'],
        OUT/phase/'ACQUISITION_COMPLETE.json': review['acquisition_complete_sha256'],
        work/'check_weights.py': review['checker_sha256'],
        work/f'{phase.upper()}_STARTED_WEIGHT_REVIEW.json': review['started_sha256'],
    }
    if review['arithmetic_audit_sha256_if_present'] is not None:
        bindings[OUT/phase/'AUDIT.json'] = review['arithmetic_audit_sha256_if_present']
    verify_inventory(ROOT, {p.relative_to(ROOT).as_posix(): h for p, h in bindings.items()})
    expected = (4 if phase == 'engineering' else 8)*102
    require(review['checkpoints'] == len(review['rows']) == expected and review['all_float32_logits_byte_exact'], 'Incomplete saved-weight replay')
    for row in review['rows']:
        checkpoint = (OUT/row['checkpoint']).resolve()
        require(checkpoint.is_relative_to((OUT/phase).resolve()), 'Unexpected checkpoint review path')
        stem = checkpoint.with_suffix('')
        artifacts = {
            checkpoint: row['checkpoint_sha256'],
            stem.with_suffix('.npy'): row['prediction_sha256'],
            Path(str(stem)+'_batches.npy'): row['batches_sha256'],
            Path(str(stem)+'_training.json'): row['training_sha256'],
        }
        verify_inventory(ROOT, {p.relative_to(ROOT).as_posix(): h for p, h in artifacts.items()})
    return path


def publish(docs: Path, diagnostic_review: Path) -> dict:
    require((OUT/'plan.json').is_file(), 'Study plan is absent')
    plan = load(OUT/'plan.json')
    require(plan['schema'] == 'world-model-acquisition-v1', 'Unexpected schema')
    require(plan['budget']['new_provider_calls'] == plan['budget']['new_estimated_usd'] == 0, 'Unexpected study spending')
    require(len(plan['maps']) == 12, 'Incomplete fixed panel')
    completion = load(OUT/'COMPLETE.json')
    require(completion['status'] == 'COMPLETE_INDEPENDENTLY_CHECKED_DESCRIPTIVE_STUDY', 'Study is not complete')
    verify_inventory(ROOT, completion['inputs_sha256'])
    inputs = [OUT/'plan.json', OUT/'PREFLIGHT.json', OUT/'COMPLETE.json']
    phases = {}
    for phase in ('engineering', 'evaluation'):
        directory = OUT/phase
        require(not (directory/'STOP.json').exists(), 'Stopped phase cannot be presented as complete')
        audit = load(directory/'AUDIT.json')
        require(audit['status'] == 'PASS' and audit['phase'] == phase, 'Independent acquisition audit has not passed')
        require(audit['plan_sha256'] == sha(OUT/'plan.json') and audit['collected_sha256'] == sha(directory/'COLLECTED.json') and audit['report_sha256'] == sha(directory/'report.json'), 'Acquisition audit is not bound to these results')
        report = load(directory/'report.json')
        verify_inventory(directory, load(directory/'COLLECTED.json')['files_sha256'])
        verify_inventory(directory, load(directory/'ACQUISITION_COMPLETE.json')['files_sha256'])
        inputs.append(verify_weight_review(phase))
        phases[phase] = normalize_phase(phase, report)
        phases[phase]['native_report'] = report
        inputs.extend([directory/'AUDIT.json', directory/'report.json', directory/'FREEZE.json', directory/'COLLECTED.json'])
    require(not (DIAGNOSTIC/'STOP.json').exists(), 'Stopped diagnostic cannot be presented as complete')
    review = load(diagnostic_review)
    require(review['status'] == 'PASS_SEPARATE_NUMERICAL_CHECK', 'Separate diagnostic checker has not passed')
    require(review['plan_sha256'] == sha(DIAGNOSTIC/'plan.json') and review['summary_sha256'] == sha(DIAGNOSTIC/'summary.json') and review['collected_sha256'] == sha(DIAGNOSTIC/'COLLECTED.json'), 'Diagnostic check is not bound to these results')
    diagnostic_complete = load(DIAGNOSTIC/'COMPLETE.json')
    require(diagnostic_complete['status'] == 'COMPLETE_CHECKED_POST_HOC_DIAGNOSTIC' and diagnostic_complete['independent_check_sha256'] == sha(diagnostic_review), 'Diagnostic completion differs')
    verify_inventory(DIAGNOSTIC, diagnostic_complete['files_sha256'])
    peer_review_path = ROOT/'work/world-model-acquisition-v1/DIAGNOSTIC_INDEPENDENT_REVIEW.json'
    peer_review = load(peer_review_path)
    require(peer_review['status'] == 'PASS_DIAGNOSTIC_INDEPENDENT_REVIEW', 'Second-author diagnostic review has not passed')
    verify_inventory(ROOT, peer_review['source_sha256'])
    verify_inventory(ROOT, peer_review['artifact_sha256'])
    repair_review_path = ROOT/'work/world-model-acquisition-v1/DIAGNOSTIC_REPAIR_INDEPENDENT_REVIEW.json'
    repair_review = load(repair_review_path)
    require(repair_review['status'] == 'PASS_DIAGNOSTIC_REPAIR_INDEPENDENT_REVIEW', 'Independent diagnostic repair check has not passed')
    verify_inventory(ROOT, repair_review['source_sha256'])
    verify_inventory(ROOT, repair_review['artifact_sha256'])
    require(repair_review['prior_review_sha256'] == sha(peer_review_path), 'Diagnostic prior-review binding differs')
    diagnostic = load(DIAGNOSTIC/'summary.json')
    require(diagnostic['model'] == 'direct_11' and diagnostic['native_baseline_exact'], 'Diagnostic baseline mismatch')
    require(diagnostic['execution_2x2']['soft_feedback']['successes'] == 600, 'Diagnostic native baseline changed')
    verify_inventory(DIAGNOSTIC, load(DIAGNOSTIC/'COLLECTED.json')['files_sha256'])
    inputs.extend([DIAGNOSTIC/'plan.json', DIAGNOSTIC/'summary.json', DIAGNOSTIC/'COLLECTED.json', DIAGNOSTIC/'COMPLETE.json', diagnostic_review, peer_review_path, repair_review_path])
    summary = {
        'schema': 'world-model-acquisition-publication-v1',
        'status': 'INDEPENDENTLY_CHECKED_DESCRIPTIVE_STUDY',
        'study': 'Can four useful observations improve a world model?',
        'engineering': phases['engineering'], 'evaluation': phases['evaluation'],
        'diagnostic': describe_diagnostic(diagnostic),
        'spending': plan['budget'], 'novelty_established': False,
        'comparison_unit': 'map; average architectures and seeds within each map first',
        'limits': plan['limits'],
        'inputs_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        'publication_source_sha256': sha(Path(__file__)),
    }
    evidence = docs/'research-evidence'
    evidence.mkdir(exist_ok=True)
    artifacts = []
    summary_path = evidence/'world-model-acquisition-summary.json'
    dump(summary_path, summary)
    artifacts.append({'file': summary_path.name, 'label': 'All results and the diagnostic summary', 'sha256': sha(summary_path), 'bytes': summary_path.stat().st_size})
    for m in plan['maps']:
        directory = OUT/m['phase']/m['id']
        record = deterministic_zip(evidence/f'world-model-acquisition-{m["id"]}.zip', [p for p in directory.rglob('*') if p.is_file()])
        record['label'] = f'{m["id"]} — {m["phase"]} evidence'
        artifacts.append(record)
    common = [p for p in (OUT/'sources').rglob('*') if p.is_file()]
    common += [p for p in OUT.glob('*') if p.is_file()]
    for phase in ('engineering', 'evaluation'):
        common += [p for p in (OUT/phase).glob('*') if p.is_file()]
    common += [p for p in SOURCE.glob('*') if p.is_file() and p.suffix in {'.py', '.md'}]
    work = ROOT/'work/world-model-acquisition-v1'
    if work.exists():
        common += [p for p in work.glob('*') if p.is_file() and p.suffix in {'.py', '.json', '.md'} and not p.name.startswith(('PUBLICATION', 'PUBLIC_REVIEW'))]
    record = deterministic_zip(evidence/'world-model-acquisition-shared.zip', common)
    record['label'] = 'Shared sources, plans and audit records'
    artifacts.append(record)
    diagnostic_files = [p for p in DIAGNOSTIC.rglob('*') if p.is_file()]
    diagnostic_files += [diagnostic_review, peer_review_path, repair_review_path, ROOT/'outputs/world-model-planning/V1/plan.json', ROOT/'outputs/world-model-planning/V1/predictions-direct_11.npy']
    record = deterministic_zip(evidence/'world-model-acquisition-diagnostic.zip', diagnostic_files)
    record['label'] = 'Earlier predictor: all diagnostic evidence'
    artifacts.append(record)
    require(all(a['bytes'] < 95 * 2**20 for a in artifacts), 'Artifact exceeds safe Git file size')
    page = docs/'world-model-acquisition.html'
    page.write_text(render_page(summary, artifacts))
    hash_record = {
        'status': 'BUILT_FROM_CHECKED_EVIDENCE', 'publication_source_sha256': sha(Path(__file__)),
        'page_sha256': sha(page), 'inputs_sha256': summary['inputs_sha256'], 'artifacts': artifacts,
        'note': 'Archive entries are byte-for-byte copies; archive metadata is fixed. The shared site manifest is maintained separately.',
    }
    dump(evidence/'world-model-acquisition-hashes.json', hash_record)
    checks = validate_html(page)
    for record in artifacts:
        require(sha(evidence/record['file']) == record['sha256'], 'Published artifact hash changed')
    return {'status': 'BUILT_AND_VALIDATED', 'page': str(page.relative_to(ROOT)), 'page_sha256': sha(page),
            'artifacts': len(artifacts) + 1, 'bytes': sum(a['bytes'] for a in artifacts), **checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs', type=Path, default=ROOT/'docs')
    parser.add_argument('--diagnostic-review', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.docs.resolve(), args.diagnostic_review.resolve()), indent=2))
