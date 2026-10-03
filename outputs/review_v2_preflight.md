# V2 prospective recovery review

**Ready for a separate v2 preregistration, subject to successful remote verification before any model evaluation.** No critical implementation blocker was found. V1 remains an aborted preregistered test, not a successful confirmation. This review performed only static inspection and input-only checks: no checkpoint was loaded, model instantiated, model forward performed, or confirmatory outcome inspected.

## Preserved original test

The original preregistration commit is `e3ea745929e44ed594667ce68330e7eda8b16acd`. The current original `preregistration_manifest.json` is byte-identical to that commit's manifest. `outputs/remote_lock.json` still names that original commit. The original `outputs/confirmatory` directory contains only `start.json` and `holdout_audit.json`; it has no model-result artifacts. At review time `outputs/confirmatory_v2` did not exist.

The amendment changes input sampling after an exact identity audit stopped v1. It does not revise the numeric forecasts or select using model outcomes. The modified sampling distribution and the failed original attempt must remain explicit in the final report.

## Input-only verification

Reviewed `prepare_holdout.py`, `audit_holdout.py`, the prepared-input branch of `evaluate.py`, `protocol.json`, the v2 paths in `preregister.py`, and the scorer's output-directory routing.

The selected artifact contains exactly 2,048 recipient rows and five corresponding donor arrays. The generator considered 2,052 candidates and rejected four whole recipient/donor groups, at zero-based candidate indices 172, 648, 1244 and 1253. Rejection reasons respectively identify the matched-key donor, cross donor, cross donor and value donor. No accepted recipient is duplicated.

Independent input checks passed for every accepted row:

* Nine-token sequence shape, token-vocabulary ranges, distinct dictionary keys and values, distinct selected pairs, and query-token identity.
* The original target equals the queried recipient value; the alternative equals the other selected recipient value and differs from the original.
* Each active key/value donor is exactly its specified two-token transposition and changes the queried answer to the alternative.
* Each matched donor swaps the two remaining pairs and preserves the original queried answer.
* The cross donor is exactly the combined key/value swap and preserves the entire original key-to-value mapping.
* The evaluator's prepared-batch donor helper reconstructs exactly the frozen donor bytes.
* Every stored array, including targets and metadata, exactly reproduces the declared candidate streams after dropping the four recorded indices. No additional rejection or reordering is hidden in the selected artifact.

Reconstructed the complete forbidden set using the frozen stream specification. It contains **2,057,974 distinct full input sequences**. All six accepted input arrays—recipient, four ordinary donors, and cross donor—have **zero forbidden-set matches**, and the accepted recipients have zero duplicates. The reconstructed forbidden-set identity and selected-file hash match the preparation metadata.

| Artifact | SHA-256 |
|---|---|
| Selected `outputs/holdout_v2.npz` bytes | `836b9d400594eca82e3d2a124735622b78e73a4fcd1f57ea137be6018d67146f` |
| Sorted little-endian uint64 forbidden-input identities | `efed96e6bf25fa068ff33488646ba34dd9ba9490ebb6b8c12b43f5a53e59398c` |

The forbidden coverage is training streams through the disclosed audit budgets, reused training validation, discovery recipients and donors, and the trained-model smoke inputs. Random-weight preflight inputs are outside that coverage. Exact identity uses all nine tokens; this is not a prohibition on shared individual keys, values, or partial sequences.

## Freeze and execution boundaries

The preparation code refuses to overwrite the selected NPZ. V2 registration uses `preregistration_manifest_v2.json` and `outputs/remote_lock_v2.json`, preserving the original paths. The evaluator reads the v2 lock, verifies frozen hashes and the remote commit, creates the distinct v2 result directory, runs the input audit, and only then enters model evaluation. The scorer reads the configured v2 result directory. The prepared branch consumes frozen recipient and donor arrays, comparing donor reconstruction to the frozen bytes.

Freeze the selected NPZ, preparation metadata, revised code and protocol, unchanged forecasts, predictor reaffirmation, this review, and the preserved v1 failure evidence. Remote tree/blob readback remains required; this review does not itself perform or certify that external step. The preparation metadata is the authoritative record of the four exclusions under the newly registered v2 sampling rule. They must never be described as exclusions permitted by the original v1 protocol.
