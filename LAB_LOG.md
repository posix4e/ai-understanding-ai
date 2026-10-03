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

## 2026-10-03 — Experiment 2 design and discovery

- Continued locally with head-specific ablation and rescue. Experiment 1 frozen
  files remain unchanged. Added a head-output hook in an isolated model copy;
  its clean outputs match the original implementation exactly.
- Trained additional seeds 3, 4 and 5 with the unchanged final training recipe
  (2,000 updates, embedding SD 1.0). All reached 100% reused pilot-validation
  accuracy. E2 uses six trained models; new seeds are reported separately.
- Prepared identity-audited calibration (512), discovery (1,024) and confirmation
  (2,048) bundles, disjoint from 2,838,765 forbidden E1/training inputs and from
  one another. One discovery candidate was rejected; no confirmation candidates
  were rejected. Donors preserve query key/location; active donors change the
  answer and matched donors preserve it.
- Fixed the complete condition grid before discovery: 53 discovery/control
  conditions and 92 reserved compositions (36 head pairs, 24 sole-head rescues,
  32 cross-layer combinations). The reviewer noted that sole-head rescue equals
  corruption of the complementary three heads, so no rescue was shown in discovery.
- Discovery clean accuracy was 100% in all six seeds. L1 single-head effects
  vary strongly by seed; L2 single-head ablations leave substantial performance
  that collapses under all-head corruption. These observations motivate testing
  redundancy and head interactions, rather than assuming singleton effects add.
- Primary scoring weights the three reserved families equally; controls do not
  inflate it. Five initial baselines include additive, multiplicative, and
  linear/logit head-count interpolation. Before registration, two paired-case
  log-odds baselines were added to match the predictor's access to raw discovery
  measurements. All seven formulas and forecasts precede confirmation.
- Reviewed recent primary literature at the user's request; see
  outputs/RECENT_PAPERS.md. CHIVE's lack of tool uplift and HyVE's validation
  failures reinforce the need for stronger baselines and independently tested code.

## 2026-10-03 10:00–10:02 — Experiment 2 confirmation

- Published registration `f4b3f1d73952e4fa449b2d4bf1f139769b0d75ad`.
  Exact remote manifest/tree readback verified all 62 frozen file identities at
  14:00:33 UTC, before the confirmation start at 14:00:49 UTC.
- Evaluated 145 conditions on 2,048 cases for each of six models, including 92
  reserved combinations (552 primary model-condition forecasts). All six clean
  accuracies were 100%; no-op target-probability discrepancies were exactly zero.
- All six models passed fixed predictive-adequacy gates and the paired comparison
  against the best of seven baselines. AI family-balanced RMSE ranged 0.042–0.075.
  Counts within 0.10 were 84, 83, 78, 85, 88, 84 out of 92, totaling 502/552.
  The 50 misses remain visible; passing the registered aggregate criterion does
  not imply per-condition or per-case perfection.
- Cross-layer forecasts equal the case-logit-additive baseline by construction;
  improvement over that baseline comes from within-layer interaction allocation.
  The frozen explanation is a numerical surrogate informed by discovery, not
  proof of unique semantic circuitry or natural head necessity.
- Scoring, thresholds, forecasts, calibration tensors, inputs and checkpoints
  were unchanged after registration. Report and raw results are under
  outputs/experiment2; the original experiment remains intact.
- Independent numerical review confirmed raw-array means, all 62 frozen hashes,
  timing and paired intervals. Misses comprise 19 pair and 31 triple/rescue
  conditions; the largest error is 0.25264. Triple/rescue family RMSE exceeds
  0.10 in seeds 1 and 2 even though the predeclared balanced score passes.
- As an explicitly post hoc diagnostic, the AI's point-estimate advantage
  survives removing any single novel condition or any whole family in each
  seed. No new confidence intervals were registered for this diagnostic. Unlike
  E1, the overall advantage is not carried by one exceptional condition.

## 2026-10-03 — Experiment 3 prospective design

- Continued toward a stronger structural finding. E3 asks whether externally
  supplied position coordinates control which key and value become associated.
  It does not reuse E2's fitted response surrogate or claim spontaneous transfer
  to a new task format. No shifted-model discovery measurements were collected.
- Kept all six existing checkpoints, all nine tokens, query index 8 and physical
  causal masking fixed. Grouped all keys before all values. Enumerated every
  four-slot permutation under coherent key/value coordinate changes and under
  value-only changes: 50 total conditions, including references and a no-op.
  Both manipulated families preserve key/value parity and use the same trained
  position IDs exactly once. The value-only hypothesis predicts the specific
  inverse-permutation answer, rather than generic disruption.
- Primary grid: all nine derangements and their matched coherent controls.
  The blinded role fixed 1,200 numerical forecasts without inspecting E3 inputs
  or executing models. Main gates require every primary accuracy cell to pass
  plus a paired confidence-separation gate in every seed. Numeric misses will
  be reported independently, without changing their fixed .15 tolerance.
- Selected 2,048 new association dictionaries from 2,127 candidates. Rejected
  79 if any of their 96 pair-order/query variants matched historical data.
  Independent enumeration found zero overlap for all 196,608 accepted variants
  against 2,849,517 audited prior inputs. Query indices are balanced at 512 each.
- Synthetic runner tests use random untrained weights only. Before freezing,
  review aligned a draft bootstrap-seed mismatch and a logit-versus-probability
  no-op wording mismatch. Full class probabilities were added to support the
  promised secondary comparisons. No E3 trained-model outcome informed these
  changes. E1-v2 and E2 frozen file hashes remain intact.
- Existing literature already studies position dependence and positional
  generalization. The narrower question here is prospective, answer-specific
  rebinding using only trained in-range coordinates; a successful test would
  establish a result in these models, not priority over the entire literature.
- A targeted primary-source search before confirmation found an especially
  close precedent in Tang, Lake and Jazayeri (PLOS ONE, February 2026, Fig. 12).
  The related-work audit therefore calls E3 a controlled replication/extension
  candidate. Known positional steering is not being presented as a first discovery.

## 2026-10-03 14:25–14:29 UTC — Experiment 3 confirmation

- Published registration `42fd4236ecfa0535dacf9ddf6c824c41bd66f02e` and
  remotely verified all 42 file identities at 14:25:09 UTC. Confirmation began
  at 14:28:14 UTC and the complete report was generated at 14:28:55 UTC.
- All six original-layout accuracies were 100% and no-op discrepancies zero.
  Nevertheless every one of the 108 primary accuracy gates failed. The
  all-six mechanistic conjunction failed; no thresholds or predictions changed.
- Grouped-native accuracy was 19.3–20.9%; canonical position-label repair
  produced only 28.9–58.8%. Nine-derangement mean slot-minus-original probability
  margins ranged .107–.487, far below the registered lower-CI requirement .80.
  Partial position-guided redirection does not rescue the strong claim.
- Only 263 of 1,200 forecasts met .15 tolerance; all 937 misses are retained.
  The largest error was .697. The stronger structural explanation failed even
  though E2's fitted numerical surrogate passed its intervention-grid test.
- A prospective next hypothesis follows from the changed causal predecessor
  sets: grouped values receive keys from later logical slots that were masked
  in training. Restoring key visibility may repair binding. This is not yet a
  verified explanation; it will require a new registration and new inputs.

## 2026-10-03 — Experiment 4 prospective visibility repair

- Developed a distinct, falsifiable repair after E3's failure. In grouped
  canonical layout, Vi gains access to keys Kj with j>i that were masked during
  training. The correct guard removes those six key-edge types in every L1
  head, without changing weights or transplanting native activations.
- The L1 value and query restoration follows algebraically from their restored
  predecessor sets. It is explicitly an implementation check, not a discovery.
  Final-answer recovery remains empirical because L1 key-node states still
  differ and can influence the second layer.
- Enumerated the complete nine masks that preserve each true key and exactly
  match per-value removal counts (3,2,1,0). The correct guard competes with the
  mean of all eight alternatives; every alternative and the best one will be
  reported. Oracle native-state patches separate value repair from key repair.
  Total: 15 fixed conditions in all six existing checkpoints.
- Frozen decisions require correct-guard accuracy at least .95 in every query
  stratum, paired probability improvement over unguarded grouping above .30,
  and improvement over the mean eight matched masks above .05, using confidence
  bounds in every seed. No superiority to every alternative is assumed.
- Prepared 2,048 new association dictionaries from 2,114 candidates, rejecting
  66 for prior-data overlap under any of their 96 variants. Independent checks
  found zero overlap against 2,853,613 prior inputs including E3. No E4 trained
  forward has run. The predictor committed 180 forecasts using E3 only as
  disclosed discovery evidence; oracle identities do not count as new findings.
