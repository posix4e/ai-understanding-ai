# Public lab log

Dates use America/New_York unless explicitly marked UTC. Entries summarize methods
and evidence, not private conversations or internal model reasoning.

## 2026-10-03 — Setup and first pilot

- Verified authenticated GitHub account `posix4e`; created public repository
  `posix4e/ai-understanding-ai` with project-only Git author metadata.
- Hardware inspection: Apple M3, 8 logical CPUs, 16 GiB RAM, macOS 27.0.1.
  CPU execution chosen for this small workload; no TDX or cloud compute needed.
- Created isolated Python 3.14.8 environment; installed PyTorch 2.14.1 and NumPy
  2.5.3. Exact dependency versions are in requirements.txt. Sandbox access to uv's
  cache and GitHub initially failed; explicit host permissions resolved them.
- Assigned separate implementer and skeptical reviewer roles. The reviewer
  recommended separating second-layer K, Q, and V transfers, patching both swapped
  positions, and avoiding semantic claims from whole-residual swaps alone.
- Initial runs: seeds 0, 1, 2; 2,000 updates, batch 128, AdamW lr 0.001,
  four CPU threads. Final discovery-validation accuracies: 24.04%, 24.61%, 24.04%
  (5,120 cases each). These runs did not learn reliable lookup; all artifacts are
  preserved in outputs/checkpoints. Value-vocabulary chance is 6.25%; the stronger
  displayed-value-choice baseline is 25%.
- Longer same-architecture training was initiated as an exploratory recovery.
  Confirmatory data/results have not been generated. No predictions are frozen yet.

## 2026-10-03 — Recovery, discovery, and prospective test design

- Seed 0 remained near 25% after 12,000 updates. Separate 12,000-update runs with
  learning rates 0.003 and 0.01 also failed. Larger token and position embedding
  initialization SD (1.0 instead of 0.02), with projection SD held at 0.02,
  reached 100% pilot accuracy. We selected this setup, then trained all three
  seeds for a fixed 2,000 updates; all reached 100% on reused pilot validation.
  See outputs/training_recovery.md for every run and command.
- Discovery used 1,024 new recipients shared across the three trained models.
  All achieved 100% accuracy. Key-swap donors transferred their answer through
  L1 value-position residuals and L2 keys at value positions (contrast changes
  about 1.998 on a [-2,2] scale). Query Q transfers were near zero. Value swaps
  transferred through L2 values. These are exploratory findings.
- After discovery, prospectively added four same-donor joint patch conditions
  per donor family and one cross-donor routing/content composition test. No
  trained model has been run on these joint conditions. The cross test combines
  L2 K from a key swap with L2 V from a value swap; it probes composition beyond
  repeating previously observed single-site effects. This design choice was
  informed by discovery and is disclosed rather than called independently chosen.
- A separate Astra prediction role received source and discovery summaries only.
  Confirmation outputs do not yet exist. Blinding is procedural; filesystem
  access is shared. The prediction role is preparing numeric forecasts and gates.
- Skeptical runtime review on random weights found no donor/indexing defects.
  The proportion interval was improved to Wilson before confirmation, since a
  normal interval degenerates at 100%. Discovery's original output remains intact.
- GitHub SSH push failed host-key verification; authenticated HTTPS succeeded.
  No host-key checks were disabled. The initial public commit includes failed runs.

## 2026-10-03 08:40–08:42 — Original registration and aborted confirmation

- Published original preregistration commit
  `e3ea745929e44ed594667ce68330e7eda8b16acd`. GitHub API readback at
  12:40:52 UTC verified the manifest and all 34 frozen Git blob identities.
  The original manifest and outputs/remote_lock.json are retained.
- Confirmation started at 12:41:45 UTC and stopped at the exact input audit,
  before any confirmatory model forward pass. Among 12,288 recipient/donor
  inputs, four training-overlap incidences were found (2 seed-0, 1 seed-1,
  1 seed-2). No discovery, validation, or recipient-duplicate overlaps occurred.
  outputs/confirmatory contains only start and failed audit artifacts.
- This is an **aborted registered test**, not a successful confirmation. The
  finite dictionary space made separate RNG seeds insufficient to guarantee
  disjoint inputs; the exact guard worked. No predictions were scored.

## 2026-10-03 — Separately registered v2 recovery

- Revision v2 constructs an exact forbidden set of covered training streams,
  reused validation, discovery recipients/donors, and trained-model smoke inputs.
  Using the same candidate RNG streams, it rejects an entire recipient bundle
  when any recipient/donor (including the both-swaps comparator) overlaps, then
  continues deterministically until 2,048 accepted recipients. Duplicate accepted
  recipients are also rejected. The filter uses no model outcomes.
- Frozen NPZ includes recipients, all five donor arrays, original/alternative
  targets, query positions and selected other pairs. Metadata records exact
  rejection reasons, candidate indices and forbidden-set hash. This changes the
  sampled population slightly and is a disclosed protocol amendment.
- The blinded predictor reaffirmed all 159 numbers and decision gates unchanged.
  Original predictions.json SHA-256 remains
  `a869ea8eb4fdaa023cddb6de7ea3484ce624922ea31c78cb5d3e83254f3d004d`.
  V2 will have a new manifest, remotely verified commit and output directory.

## 2026-10-03 08:45–08:49 — V2 confirmation

- Published v2 commit `b2d3bdd55db2dd9a8f2f3f2548dabe9fe8f53b37` and read
  back its exact manifest and all 45 frozen file identities at 12:45:04 UTC.
  Confirmation's start record is 12:48:07 UTC; the exact audit passed before
  model forwards. Original v1 evidence was included in the new manifest.
- All three models answered 2,048/2,048 recipient lookups correctly. All 159
  mean forecasts passed the fixed absolute-error bound 0.15; largest error
  0.002686. Key/K-minus-query/Q primary contrasts were 1.998133, 1.997834 and
  1.998322; every bootstrap lower bound exceeded the threshold 1.5.
- Cross-donor cancellation original-answer accuracies were 100%, 99.951%,
  99.951%. Seed 1 and seed 2 each made one error to a third value. Small nonzero
  effects are retained; exact invariance was not observed. Value/V transfer also
  failed on four seed-1 cases and one seed-2 case.
- The explanation beat zero, full-donor transfer and empirical/additive baselines
  overall. Its advantage over empirical/additive forecasts is driven by the
  single prospectively selected cross-donor test. The empirical discovery-mean
  baseline was slightly better on familiar interventions in all three seeds.
- No forecasts, scoring code, checkpoints, conditions or gates were changed
  after v2 freeze. Post-outcome build_report.py only formats saved results and
  evaluates the existing gates. All outcomes and controls remain public.
- Independent skeptical review reproduced the key arithmetic, gates, hashes,
  input identities and timing. Its post hoc diagnostic excluding the cross test
  reverses the overall AI advantage in all seeds; this limitation is explicit
  in outputs/SKEPTICAL_REVIEW.md.
