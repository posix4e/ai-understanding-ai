# The first-layer repair did not transfer to GPT-2

**The repair restored the checked value states in the first layer. It did not restore correct answers.** This is a limit of the earlier small-model result. It is not evidence that all repairs fail in language models.

## What we tested

We used pretrained GPT-2 small: **124,439,808 parameters and 12 layers**. It has about 1,760 times as many parameters as our earlier 70,720-parameter, two-layer model. All model computation ran locally on a CPU. We did not train GPT-2 again.

The task still used four label–value pairs. We selected one of four planned text formats using 64 separate calibration inputs, with no reordered-input tests during selection. The selected format includes one worked example. We then tested 512 generated dictionaries, with 128 questions about each pair position, under all 15 planned conditions. We excluded calibration dictionaries. We cannot audit overlap with GPT-2's pretraining data.

We moved the exact same text tokens into a grouped order. Their original position tags stayed attached. This makes an artificial text sequence, not a natural paraphrase. The repair was given the correct pairs and original positions.

## The checked result

These numbers come from the full float64 precision repeat. It reused the original 512 inputs.

| Condition | Correct among 16 possible answers | Correct next token across the full vocabulary |
| --- | ---: | ---: |
| Original order | 477/512 — **93.2%** | 445/512 — **86.9%** |
| Grouped order, original position tags | 53/512 — **10.4%** | 0/512 — **0%** |
| Grouped order, first-layer repair | 49/512 — **9.6%** | 0/512 — **0%** |
| Separate test: guard in all 12 layers | 160/512 — **31.25%** | 20/512 — **3.9%** |

The first-layer repair failed **all three planned repair requirements**:

- Its change in accuracy was −0.78 percentage points. The paired 95% interval was −1.95 to +0.39 points. The required lower bound was above +5 points.
- Its accuracy remained 83.59 points below the original order. The interval was 80.27 to 86.72 points. The allowed upper bound was 5 points.
- Its advantage over the mean of eight alternative masks was about 0.003 percentage points of conditional answer probability. The interval included zero. The required lower bound was above 2 points.

The all-layer result was a separate, secondary test. It did not satisfy or replace the first-layer claim. The sixth-layer-only test reached 10.5% restricted accuracy. [All 15 conditions, query groups, and intervals](precision_repeat/RESULTS.md) remain available. These intervals describe variation across the sampled dictionaries, not across a population of models.

## Why there are two runs

The original float32 run narrowly failed two numerical checks. Its maximum first-layer value-state error was 0.00001335; its query-state error was 0.00001144. The fixed limit was 0.00001. Its official classification stays **implementation_invalid**.

After inspecting that result, we publicly registered a repeat using more precise arithmetic. We kept the same weights, inputs, conditions, and pass requirements. We promoted the saved parameter values exactly to float64 and verified that they stayed unchanged. The repeat passed every implementation check. Both state errors fell to about **0.0000000000000213**. Its official classification is **repair_failed**.

This is an outcome-informed numerical check on reused inputs. It is **not an independent confirmation**. The primary answer counts were unchanged. Across all 7,680 input–condition combinations, two restricted predictions and two unrestricted predictions changed. These counts can include the same input under multiple conditions. The largest saved candidate-probability change was 0.00004659. [Compare every condition](precision_repeat/comparison.csv).

A separate audit program checked 128 frozen files, model file hashes, 64 raw batch files, and all 15 merged conditions. It independently recomputed 865 numerical entries and agreed within 0.0000000001. Hidden-state checks remain measurements from the runner; the saved answer probabilities cannot independently reconstruct them. This is a separate code audit within the project, not an external replication. [Audit record](precision_repeat/independent_audit.json).

## What changed our expectations

The forecast expected about 50% restricted accuracy with the first-layer repair. The observed result was 9.6%. Only two of six frozen numerical forecasts met the stated error tolerance. The failure was much larger than the forecast expected. [All forecasts and errors](precision_repeat/forecasts_vs_results.csv).

The result shows that restoring selected early value and query states was insufficient here. It does not identify the cause. The repair leaves other states changed, and later layers still process the reordered input. Model size, depth, training, vocabulary, and text format also differ from the earlier study. We cannot attribute the failure to size alone.

## Do other researchers use GPT-2 for this kind of work?

Yes. Wang and colleagues used causal interventions to study how GPT-2 small identifies who receives an object in a sentence. [Their paper](https://arxiv.org/abs/2211.00593).

Hanna and colleagues identified internal components that help GPT-2 small complete dates with a later year. [Their paper](https://arxiv.org/abs/2305.00586).

These establish precedent for the model and method. They do not establish that our exact repair test has already been done. We have not established field-wide novelty for this test.

## What to test next

Our next proposed experiment would separate two possible causes: altered label states from the first layer, and altered information paths in later layers. Test each correction alone and both together on fresh inputs, with rules fixed in advance. Full restoration would be a diagnostic control; it would supply internal information that a practical repair would not normally have.

Additional pretrained models, longer lists, and natural text tasks are also needed. Moving beyond small synthetic models follows broad directions in the [earlier papers](../PAPER_NEXT_STEPS.md). The exact four-condition diagnostic is our proposal.

This negative result is useful to report because it prevents a successful small-model repair from becoming an unsupported claim about pretrained language models. It strengthens the limits of the evidence. It is a research follow-up, not a peer-reviewed result.

## Evidence and reproduction

- [Original run](RESULTS.md), [raw outputs](confirmatory), and [original independent audit](independent_audit.json)
- [Precision-repeat results](precision_repeat/RESULTS.md), [raw outputs](precision_repeat/confirmatory), and [decoded predictions](precision_repeat/all_predictions.csv)
- [Original protocol](../../experiment6/PROTOCOL.md), [precision plan](../../experiment6/PRECISION_PLAN.md), and [all condition definitions](../../experiment6/conditions.py)
- [Original public registration](https://github.com/posix4e/ai-understanding-ai/commit/6ae8866e675cead908ae40c1862e81ee269eaf42) and [precision registration](https://github.com/posix4e/ai-understanding-ai/commit/6edaaac7c9a1eab41f4cc38ab700d212dcf866dc)
- [Repeat or audit the study](REPRODUCE.md)

The [original small-model paper](../paper/paper.pdf) remains a separate document. This follow-up does not change its recorded results.
