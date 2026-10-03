"""Synthetic implementation tests only: never load trained checkpoints/data."""

import unittest
from unittest.mock import patch

import torch

from model import TinyTransformer as OriginalModel
from experiment2.model import ModelConfig, TinyTransformer, generate_batch
from experiment2.runner import calibrate, run, validate_conditions


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(817)
        cls.model = TinyTransformer(ModelConfig(embedding_std=1.0)).eval()
        generator = torch.Generator().manual_seed(981)
        tokens, targets, _ = generate_batch(11, generator)
        donor, _, _ = generate_batch(11, generator)
        cls.arrays = {"tokens": tokens, "targets": targets, "donor_tokens": donor,
                      "matched_donor_tokens": tokens.clone()}
        cls.calibration, _, _ = generate_batch(23, generator)
        cls.means = calibrate(cls.model, cls.calibration, batch_size=7)

    @staticmethod
    def condition(name, layer=0, heads=None, kind="zero", **extra):
        op = {"layer": layer, "component": "z", "positions": [1, 3, 5, 7] if layer == 0 else [8],
              "heads": [0] if heads is None else heads, "kind": kind, **extra}
        return {"id": name, "ops": [op]}

    def assert_metrics_close(self, first, second):
        for metric in first:
            torch.testing.assert_close(torch.tensor(first[metric]), torch.tensor(second[metric]),
                                       rtol=1e-6, atol=1e-7)

    def test_original_state_and_logits_unchanged(self):
        original = OriginalModel(self.model.config).eval()
        original.load_state_dict(self.model.state_dict(), strict=True)
        with torch.no_grad():
            expected = original(self.arrays["tokens"])
            actual, cache = self.model(self.arrays["tokens"], return_cache=True)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(tuple(cache[(0, "z")].shape), (11, 9, 4, 16))
        for layer in range(2):
            projection = self.model.blocks[layer].attn_projection(cache[(layer, "z")].reshape(11, 9, 64))
            self.assertTrue(torch.equal(projection, cache[(layer, "attn_out")]))

    def test_calibration_is_per_position_and_frozen(self):
        with torch.no_grad():
            _, cache = self.model(self.calibration, return_cache=True)
        for key, mean in self.means.items():
            expected = cache[key].double().mean(0, keepdim=True).float()
            torch.testing.assert_close(mean, expected, rtol=1e-6, atol=1e-7)
        before = {key: value.clone() for key, value in self.means.items()}
        run(self.model, self.arrays, [self.condition("m", kind="mean")], self.means)
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in self.means.items()))

    def test_noop_and_rescue_extremes(self):
        conditions = [{"id": "clean", "ops": []}]
        for layer in range(2):
            conditions.extend([
                self.condition(f"noop{layer}", layer, heads=[0, 1, 2, 3], kind="noop"),
                self.condition(f"restore{layer}", layer, heads=[0, 1, 2, 3], kind="rescue", corruption="mean"),
                self.condition(f"none{layer}", layer, heads=[], kind="rescue", corruption="mean"),
                self.condition(f"all{layer}", layer, heads=[0, 1, 2, 3], kind="mean")])
        results = run(self.model, self.arrays, conditions, self.means)
        for layer in range(2):
            self.assertEqual(results["clean"], results[f"noop{layer}"])
            self.assertEqual(results["clean"], results[f"restore{layer}"])
            self.assertEqual(results[f"none{layer}"], results[f"all{layer}"])

    def test_rescue_equals_complement_ablation(self):
        for layer in range(2):
            for kind in ("zero", "mean", "resample"):
                conditions = [self.condition("ablate", layer, heads=[1, 2, 3], kind=kind),
                              self.condition("rescue", layer, heads=[0], kind="rescue", corruption=kind)]
                results = run(self.model, self.arrays, conditions, self.means)
                self.assertEqual(results["ablate"], results["rescue"])

    def test_crosslayer_against_independent_live_hook(self):
        condition = {"id": "joint", "ops": [self.condition("a")["ops"][0],
                                                self.condition("b", 1, heads=[2])["ops"][0]]}
        result = run(self.model, self.arrays, [condition])["joint"]
        tokens, targets = self.arrays["tokens"], self.arrays["targets"]
        with torch.no_grad():
            _, clean = self.model(tokens, return_cache=True)
            first = clean[(0, "z")].clone()
            first[:, [1, 3, 5, 7], 0] = 0
            upstream_patch = {(0, "z"): ([1, 3, 5, 7], first)}
            _, upstream = self.model(tokens, patch=upstream_patch, return_cache=True)
        # Independent oracle: intercept the projection's live input, not a cache.
        observed = {}
        original_forward = self.model.blocks[1].attn_projection.forward

        def live_projection(flat):
            live = flat.reshape(len(tokens), 9, 4, 16).clone()
            observed["before"] = live.clone()
            live[:, 8, 2] = 0
            observed["after"] = live.clone()
            return original_forward(live.reshape(len(tokens), 9, 64))

        with torch.no_grad(), patch.object(self.model.blocks[1].attn_projection, "forward", live_projection):
            final = self.model(tokens, patch=upstream_patch)[:, -1]
        rows = torch.arange(len(tokens))
        others = final.clone()
        others[rows, targets] = -torch.inf
        expected = {"target_probability": final.softmax(-1)[rows, targets].tolist(),
                    "accuracy": (final.argmax(-1) == targets).long().tolist(),
                    "logit_margin": (final[rows, targets] - others.max(-1).values).tolist()}
        self.assert_metrics_close(result, expected)
        self.assertTrue(torch.equal(observed["before"], upstream[(1, "z")]))
        self.assertFalse(torch.equal(observed["before"][:, 8, [0, 1, 3]], clean[(1, "z")][:, 8, [0, 1, 3]]))
        self.assertTrue(torch.equal(observed["after"][:, 8, [0, 1, 3]], upstream[(1, "z")][:, 8, [0, 1, 3]]))

    def test_same_input_resample_and_mlp_control(self):
        conditions = [{"id": "clean", "ops": []},
                      self.condition("resample", kind="resample", donor="matched_donor_tokens"),
                      {"id": "mlp", "ops": [{"layer": 0, "component": "mlp_out", "positions": [1, 3, 5, 7], "kind": "zero"}]}]
        results = run(self.model, self.arrays, conditions)
        self.assertEqual(results["clean"], results["resample"])
        self.assertEqual(len(results["mlp"]["accuracy"]), len(self.arrays["tokens"]))

    def test_batching_invariance(self):
        conditions = [self.condition("a", kind="mean"), self.condition("b", 1, kind="resample")]
        small = run(self.model, self.arrays, conditions, self.means, batch_size=3)
        whole = run(self.model, self.arrays, conditions, self.means, batch_size=11)
        for name in small:
            self.assert_metrics_close(small[name], whole[name])

    def test_invalid_conditions_rejected(self):
        a = self.condition("a")
        with self.assertRaises(ValueError):
            validate_conditions([a, a], self.model.config)
        with self.assertRaises(ValueError):
            validate_conditions([{"id": "duplicate_site", "ops": a["ops"] * 2}], self.model.config)
        with self.assertRaises(ValueError):
            validate_conditions([self.condition("badhead", heads=[4])], self.model.config)
        with self.assertRaises(ValueError):
            run(self.model, self.arrays, [self.condition("missingmean", kind="mean")])


if __name__ == "__main__":
    unittest.main()
