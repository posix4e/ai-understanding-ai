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
