# E6 preconfirmation review

Before any reordered or repaired pretrained GPT-2 forward:

- The implementation agent verified the installed Transformers 4.57.6 attention
  signature and causal-mask handling. Six random-weight adapter tests passed.
- Five condition tests passed, including exhaustive enumeration of the nine
  matched masks, whole-chunk edge counts, own-key retention, physical causality,
  and correct-mask equality to native first-block value predecessor sets.
- Nine scorer tests passed, including saved-file integration, exact threshold
  boundaries, paired stratified resampling, frozen input labels, and secondary
  success that cannot override primary failure.
- Static integration review found and corrected a list-versus-object input
  contract mismatch before confirmation. It also added verification of the
  registration manifest's own bytes and the actual tokenizer answer mapping.
- A separate random-weight 12-block integration check exercised all fifteen
  conditions and the runner's inverse-permutation/state diagnostics. Maximum
  no-op probability error was zero; value and query-suffix state discrepancies
  were 3.73e-9; unedited-key discrepancy was zero. Parameter hash was unchanged.
- The 512 confirmation maps were reconstructed from the declared generator.
  They are unique and exclude every calibration map in every pair/query order.
  Each query stratum has 128 examples. Exact tokenization/chunks were checked.
- The forecast agent received the chosen format name and N, but no calibration
  scores, inputs, or pretrained outcomes. This is procedural role separation in
  a shared filesystem, not security-enforced isolation.

The root agent observed the native-only calibration to choose the format under
the registered rule. No agent observed shifted GPT-2 outcomes before the
confirmation registration. No frozen E1–E5 file was changed.

The first public confirmation registration (`f2b8bd9`) was superseded before
execution. Final static review caught a helper name shadowed by a local tensor
name in the added tokenizer check. The corrected runner and refreshed manifest
were publicly registered before any shifted/repaired trained-model forward.
No outcome, threshold, input or scientific condition changed.

The next registration (`5093cb6`) stopped on the first native batch because the
runtime's float32 softmax over 50,257 tokens produced sums of 1.000039–1.000059.
No batch results or shifted/repaired forwards occurred. The start record and a
native-only numerical diagnostic are retained in the outputs. Float64 softmax
of the same float32 logits gave sums within 9e-15 of one; maximum individual
probability difference was 1.69e-5. A new public registration changes only the
final softmax precision and documentation. It retains every validity tolerance,
behavioral threshold, input, model parameter, condition and forecast. Calibration
remains as originally recorded; its format choice used untied answer accuracies.

## Interpretation limits fixed in advance

The claim concerns one checkpoint, one calibrated lookup format and one grouped
serialization. It does not isolate scale, establish universal repair, infer its
own pair structure, or compare pretrained architectures. Restricted answer
ranking and unrestricted generation remain separate. The first-block identity
is not a guarantee about twelve-block behavior. Controls match edge counts,
not attention mass. All failed gates and controls will remain in the record.
