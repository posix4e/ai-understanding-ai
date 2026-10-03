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
