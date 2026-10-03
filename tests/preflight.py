"""Run from repository root: .venv/bin/python tests/preflight.py

Architecture and evaluator-unit checks on fresh random weights and pilot-only
synthetic examples. Never calls evaluate.run/main or loads trained checkpoints.
Prints a JSON report to stdout without creating or modifying result files.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model import ModelConfig, TinyTransformer, generate_batch
from evaluate import FAMILIES, SITES, COMBINATIONS, make_donors, patch_tensor, measure, proportion_interval

torch.set_num_threads(1)
torch.manual_seed(734821)
torch.use_deterministic_algorithms(True)
model = TinyTransformer(ModelConfig()).eval()
generator = torch.Generator().manual_seed(834822)
tokens, targets, meta = generate_batch(11, generator, model.config)
other, _, _ = generate_batch(11, generator, model.config)
report = {"scope": "fresh random weights; pilot-only synthetic inputs; no trained checkpoint loaded", "torch_version": str(torch.__version__), "model_seed": 734821, "data_seed": 834822, "checks": {}}
checks = report["checks"]

def selected_positions(kind, a, b):
    if kind == 'query':
        return [8]
    if kind == 'wrong_values':
        return [2*j+1 for j in range(4) if j not in (a,b)]
    return [2*a + (kind != 'keys'), 2*b + (kind != 'keys')]

@torch.no_grad()
def evaluator_checks():
    # Test all ordered pair selections on rank-3 and rank-4 sentinel tensors.
    pairs = [(a,b) for a in range(4) for b in range(4) if a != b]
    a, b = (torch.tensor([pair[i] for pair in pairs]) for i in (0,1))
    remaining = torch.tensor([[j for j in range(4) if j not in pair] for pair in pairs])
    for shape in ((12,9,64),(12,9,4,16)):
        recipient = torch.arange(12*9*64).reshape(shape).float()
        donor = recipient + 100000
        for kind in ('query','values','keys','wrong_values'):
            actual = patch_tensor(recipient,donor,kind,a,b,remaining)
            expected = recipient.clone()
            for row, (x,y) in enumerate(pairs):
                for pos in selected_positions(kind,x,y):
                    expected[row,pos] = donor[row,pos]
            assert torch.equal(actual,expected), (shape,kind)
            assert torch.equal(patch_tensor(recipient,recipient,kind,a,b,remaining),recipient)
        assert torch.equal(donor,recipient+100000)
    checks['exhaustive_per_row_mixed_tensor_selection'] = {'status':'pass','ordered_pair_selections':12,'ranks':[3,4]}

    donor_generator = torch.Generator().manual_seed(834823)
    donors, other_pair, remaining = make_donors(tokens,meta,donor_generator)
    assert set(donors) == set(FAMILIES)
    baseline, recipient_cache = model(tokens,return_cache=True)
    max_batch_difference = 0.0
    max_joint_difference = 0.0
    family_caches = {}
    for family,(donor_tokens,x,y) in donors.items():
        token_offset = int(family.endswith('value'))
        for row in range(len(tokens)):
            orig_pair = int(meta['query_pair'][row])
            xrow, yrow = int(x[row]), int(y[row])
            assert xrow != yrow
            expected = tokens[row].clone()
            expected[2*xrow+token_offset] = tokens[row,2*yrow+token_offset]
            expected[2*yrow+token_offset] = tokens[row,2*xrow+token_offset]
            assert torch.equal(donor_tokens[row],expected)
            assert int(donor_tokens[row,-1]) == int(tokens[row,-1])
            matches = (donor_tokens[row,:-1:2] == donor_tokens[row,-1]).nonzero().flatten()
            assert len(matches) == 1
            donor_answer = int(donor_tokens[row,2*int(matches[0])+1])-model.config.n_keys
            expected_answer = int(targets[row]) if family.startswith('matched') else int(meta['values'][row,other_pair[row]])
            assert donor_answer == expected_answer
            if family.startswith('matched'):
                assert orig_pair not in (xrow,yrow)
            else:
                assert orig_pair == xrow
        _, donor_cache = model(donor_tokens,return_cache=True)
        family_caches[family] = donor_cache
        donor_saved = {k:v.clone() for k,v in donor_cache.items()}
        unused = torch.tensor([[j for j in range(4) if j not in (int(xx),int(yy))] for xx,yy in zip(x,y)])
        for site,(layer,component,kind) in SITES.items():
            mixed = patch_tensor(recipient_cache[layer,component],donor_cache[layer,component],kind,x,y,unused)
            patched = model(tokens,patch={(layer,component):(slice(None),mixed)})
            noop = patch_tensor(recipient_cache[layer,component],recipient_cache[layer,component],kind,x,y,unused)
            assert torch.equal(model(tokens,patch={(layer,component):(slice(None),noop)}),baseline)
            for row in range(len(tokens)):
                pos = selected_positions(kind,int(x[row]),int(y[row]))
                individual = model(tokens[row:row+1],patch={(layer,component):(pos,donor_cache[layer,component][row:row+1])})
                diff = (individual-patched[row:row+1]).abs().max().item()
                max_batch_difference = max(diff,max_batch_difference)
                assert torch.allclose(individual,patched[row:row+1],atol=2e-6,rtol=0), (family,site,row,diff)
            if site == 'l2_q_values':
                assert torch.equal(patched[:,-1],baseline[:,-1])
        for name,sites in COMBINATIONS.items():
            # Assemble the evaluator's sequential same-component updates, then
            # compare to an independent union-of-selected-positions oracle.
            joint, joint_noop = {}, {}
            for site in sites:
                layer,component,kind = SITES[site]
                key = (layer,component)
                current = joint.get(key,(None,recipient_cache[key]))[1]
                joint[key] = (slice(None),patch_tensor(current,donor_cache[key],kind,x,y,unused))
                current_noop = joint_noop.get(key,(None,recipient_cache[key]))[1]
                joint_noop[key] = (slice(None),patch_tensor(current_noop,recipient_cache[key],kind,x,y,unused))
            expected = {key:recipient_cache[key].clone() for key in joint}
            for row in range(len(tokens)):
                for site in sites:
                    layer,component,kind = SITES[site]
                    for pos in selected_positions(kind,int(x[row]),int(y[row])):
                        expected[layer,component][row,pos] = donor_cache[layer,component][row,pos]
            for key,value in expected.items():
                assert torch.equal(joint[key][1],value), (family,name,key)
            joint_logits = model(tokens,patch=joint)
            assert torch.equal(model(tokens,patch=joint_noop),baseline), (family,name)
            # Same-component disjoint position patches must commute.
            reverse = {}
            for site in reversed(sites):
                layer,component,kind = SITES[site]
                key = (layer,component)
                current = reverse.get(key,(None,recipient_cache[key]))[1]
                reverse[key] = (slice(None),patch_tensor(current,donor_cache[key],kind,x,y,unused))
            assert torch.equal(model(tokens,patch=reverse),joint_logits), (family,name)
            for row in range(len(tokens)):
                individual_patch = {}
                for key in joint:
                    positions = sorted({pos for site in sites if SITES[site][:2] == key
                                        for pos in selected_positions(SITES[site][2],int(x[row]),int(y[row]))})
                    individual_patch[key] = (positions,donor_cache[key][row:row+1])
                individual = model(tokens[row:row+1],patch=individual_patch)
                diff = (individual-joint_logits[row:row+1]).abs().max().item()
                max_joint_difference = max(diff,max_joint_difference)
                assert torch.allclose(individual,joint_logits[row:row+1],atol=2e-6,rtol=0), (family,name,row,diff)
        for key in donor_cache:
            assert torch.equal(donor_saved[key],donor_cache[key])
    checks['all_donor_families_and_answer_invariants'] = {'status':'pass','families':len(FAMILIES)}
    checks['all_evaluator_conditions_batched_vs_individual_and_noop'] = {'status':'pass','conditions':len(FAMILIES)*len(SITES),'examples_per_condition':len(tokens),'max_logit_difference':max_batch_difference,'tolerance':2e-6}
    checks['last_layer_nonquery_q_structural_negative_control'] = 'pass'
    checks['joint_patch_union_noop_order_and_individual_equivalence'] = {'status':'pass','conditions':len(FAMILIES)*len(COMBINATIONS),'examples_per_condition':len(tokens),'max_logit_difference':max_joint_difference,'tolerance':2e-6}
    _, active_a, active_b = donors['key']
    assert torch.equal(active_a,donors['value'][1]) and torch.equal(active_b,donors['value'][2])
    cross_patch = {}
    for family,component in [('key','k'),('value','v')]:
        mixed = patch_tensor(recipient_cache[1,component],family_caches[family][1,component],'values',active_a,active_b,remaining)
        cross_patch[1,component] = (slice(None),mixed)
        for row in range(len(tokens)):
            expected = recipient_cache[1,component][row].clone()
            pos = selected_positions('values',int(active_a[row]),int(active_b[row]))
            expected[pos] = family_caches[family][1,component][row,pos]
            assert torch.equal(mixed[row],expected)
    cross_logits = model(tokens,patch=cross_patch)
    max_cross_difference = 0.0
    both_swaps = donors['key'][0].clone()
    both_swaps[:,1::2] = donors['value'][0][:,1::2]
    for row in range(len(tokens)):
        original_mapping = dict(zip(tokens[row,:-1:2].tolist(),tokens[row,1::2].tolist()))
        both_mapping = dict(zip(both_swaps[row,:-1:2].tolist(),both_swaps[row,1::2].tolist()))
        assert both_mapping == original_mapping
        assert torch.equal(both_swaps[row,-1],tokens[row,-1])
        pos = selected_positions('values',int(active_a[row]),int(active_b[row]))
        individual = model(tokens[row:row+1],patch={(1,'k'):(pos,family_caches['key'][1,'k'][row:row+1]),
                                                   (1,'v'):(pos,family_caches['value'][1,'v'][row:row+1])})
        diff = (individual-cross_logits[row:row+1]).abs().max().item()
        max_cross_difference = max(diff,max_cross_difference)
        assert torch.allclose(individual,cross_logits[row:row+1],atol=2e-6,rtol=0)
    assert torch.equal(model(tokens,patch={(1,'k'):(slice(None),recipient_cache[1,'k']),
                                          (1,'v'):(slice(None),recipient_cache[1,'v'])}),baseline)
    checks['cross_donor_keyK_valueV_selection_equivalence_noop_and_both_swap_map'] = {'status':'pass','max_logit_difference':max_cross_difference,'tolerance':2e-6}
    sample_logits = torch.tensor([[0.,1.,2.,3.],[4.,1.,2.,0.]])
    actual = measure(sample_logits,torch.tensor([3,0]),torch.tensor([1,2]))
    probabilities = sample_logits.softmax(-1)
    expected = torch.tensor([[probabilities[0,1]-probabilities[0,3],0.,1.],[probabilities[1,2]-probabilities[1,0],0.,1.]])
    assert torch.equal(torch.from_numpy(actual),expected)
    checks['probability_contrast_sign_and_accuracy_metric'] = 'pass'
    zero = proportion_interval([0.0]*100)
    one = proportion_interval([1.0]*100)
    half = proportion_interval([0.0,1.0]*50)
    assert abs(zero['ci95'][0]) < 1e-12 and 0 < zero['ci95'][1] < .1
    assert abs(one['ci95'][1]-1) < 1e-12 and .9 < one['ci95'][0] < 1
    assert abs(half['ci95'][0]+half['ci95'][1]-1) < 1e-12
    assert abs(zero['ci95'][1] - (1-one['ci95'][0])) < 1e-12
    checks['wilson_interval_boundary_non_degeneracy_and_symmetry'] = 'pass'

@torch.no_grad()
def run():
    logits, cache = model(tokens, return_cache=True)
    donor_logits, donor = model(other, return_cache=True)
    original_cache = {k:v.clone() for k,v in cache.items()}
    original_donor = {k:v.clone() for k,v in donor.items()}
    original_tokens = tokens.clone()
    original_other = other.clone()
    assert logits.shape == (11, 9, 16)
    assert torch.equal(targets, meta["values"][torch.arange(11), meta["query_pair"]])
    assert torch.equal(tokens[:, -1], meta["keys"][torch.arange(11), meta["query_pair"]])
    checks["task_shapes_and_target_indexing"] = "pass"

    max_causal_difference = 0.0
    for cut in range(1, 9):
        changed = tokens.clone()
        changed[:, cut:] = other[:, cut:]
        changed_logits = model(changed)
        diff = (changed_logits[:, :cut] - logits[:, :cut]).abs().max().item()
        max_causal_difference = max(max_causal_difference, diff)
        assert diff == 0, (cut, diff)
    for layer in range(2):
        pattern = cache[(layer, "attn_pattern")]
        assert torch.count_nonzero(pattern.triu(1)).item() == 0
        assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=2e-7, rtol=0)
    checks["causal_suffix_invariance_and_attention_mask"] = {"status":"pass", "max_difference":max_causal_difference}

    max_noop_difference = 0.0
    indexes = [4, [1, 5], slice(None), torch.tensor([2, 6])]
    for key, value in cache.items():
        for positions in indexes:
            patched = model(tokens, patch={key:(positions, value)})
            diff = (patched-logits).abs().max().item()
            max_noop_difference = max(max_noop_difference, diff)
            assert diff == 0, (key, positions, diff)
    checks["all_site_noop_and_index_forms"] = {"status":"pass", "sites":len(cache), "index_forms":len(indexes), "max_difference":max_noop_difference}

    for key, value in cache.items():
        positions = [1, 5]
        patched_logits, patched_cache = model(tokens, patch={key:(positions, donor[key])}, return_cache=True)
        axis = 2 if key[1] == "attn_pattern" else 1
        index = [slice(None)] * value.ndim
        index[axis] = positions
        expected = value.clone()
        expected[tuple(index)] = donor[key][tuple(index)]
        assert torch.equal(patched_cache[key], expected), key
    assert torch.equal(tokens, original_tokens)
    assert torch.equal(other, original_other)
    for key in cache:
        assert torch.equal(cache[key], original_cache[key]), key
        assert torch.equal(donor[key], original_donor[key]), key
    assert torch.equal(model(other), donor_logits)
    assert torch.equal(model(tokens), logits)
    checks["patch_exact_selection_cache_inputs_and_donor_isolation"] = "pass"

    for layer, block in enumerate(model.blocks):
        residual = cache[(layer,"resid_pre")]
        normalized = block.ln_attn(residual)
        projected = block.qkv(normalized)
        for i,name in enumerate(("q","k","v")):
            expected = projected[:,:,i*64:(i+1)*64].reshape(11,9,model.config.n_heads,64//model.config.n_heads)
            assert torch.equal(expected,cache[(layer,name)]), (layer,name)
        assert torch.equal(cache[(layer,"resid_mid")], residual + cache[(layer,"attn_out")])
        assert torch.equal(cache[(layer,"resid_post")], cache[(layer,"resid_mid")] + cache[(layer,"mlp_out")])
        scores = torch.einsum("bthd,bshd->bhts", cache[(layer,"q")], cache[(layer,"k")]) / ((64 // model.config.n_heads) ** 0.5)
        scores = scores.masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1), float("-inf"))
        assert torch.equal(scores.softmax(-1), cache[(layer,"attn_pattern")])
        combined = torch.einsum("bhts,bshd->bthd", cache[(layer,"attn_pattern")], cache[(layer,"v")]).reshape(11,9,64)
        assert torch.equal(block.attn_projection(combined), cache[(layer,"attn_out")])
    checks["qkv_pre_norm_scaled_attention_residual_semantics"] = "pass"

    post = model(tokens, patch={(0,"resid_post"):([1,5],donor[(0,"resid_post")])})
    pre = model(tokens, patch={(1,"resid_pre"):([1,5],donor[(1,"resid_pre")])})
    assert torch.equal(post,pre)
    assert (post-logits).abs().max().item() > 0
    q, qcache = model(tokens, patch={(1,"q"):(8,donor[(1,"q")])}, return_cache=True)
    assert torch.equal(qcache[(1,"k")],cache[(1,"k")])
    assert torch.equal(qcache[(1,"v")],cache[(1,"v")])
    assert torch.equal(qcache[(1,"resid_pre")],cache[(1,"resid_pre")])
    assert (q-logits).abs().max().item() > 0
    checks["layer_boundary_equivalence_and_component_locality"] = "pass"
    try:
        model(tokens, patch={(2,"q"):(8,donor[(1,"q")])})
    except KeyError:
        pass
    else:
        raise AssertionError("Invalid site accepted")
    try:
        model(tokens, patch={(1,"q"):(8,donor[(1,"q")][:1])})
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid donor shape accepted")
    checks["invalid_site_and_shape_rejection"] = "pass"

for n_heads in (1,4):
    torch.manual_seed(734821)
    model = TinyTransformer(ModelConfig(n_heads=n_heads)).eval()
    checks = {}
    run()
    evaluator_checks()
    report["checks"][f"n_heads_{n_heads}"] = checks
report["files_sha256"] = {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ("model.py","train.py","evaluate.py","score.py","protocol.json","tests/preflight.py")}
print(json.dumps(report,indent=2))
