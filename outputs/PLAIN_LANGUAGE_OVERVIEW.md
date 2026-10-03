# What we found and why it may be worth publishing

We now have a measured result that may be worth publishing. It is too early to claim that the repair works in large AI systems.

## The problem

We trained six small AI models to match labels with values. For example, a list can contain “box A: 7” and “box B: 4.” The AI must return the value for the requested label.

The AI gave correct answers when we used the original input order. When we changed the order, the AI often gave wrong answers.

Changing the order also changed which information could reach each internal part of the AI. The position labels alone did not prevent this problem.

## The repair

We blocked selected internal connections. This limited which labels each value could receive information from. We kept the trained model unchanged.

The repair used the known label relationships. The AI did not find those relationships for itself during this test.

## The new result

The main test used six newly trained models and 105 input orders. All six models passed the requirements that we had set before the test.

Five models gave correct answers in every main test case. The sixth model made 15 errors across 107,520 combinations of inputs and orders. Those errors involved two inputs.

The lowest accuracy for any input order and query position was 99.6%. We kept all errors in the results.

A second test used 2,415 other input orders and a smaller set of 256 inputs. Two connection rules gave correct answers in every tested case. One rule let each value receive information only from itself and its own label.

This result shows that a correct answer does not always require the original internal state. It does not prove that the rule will always work.

## Why the work may be worth publishing

The useful result is a specific explanation that passed a new test. It tells us which connections to change and what result to expect.

The evidence has these strengths:

- We specified the test and its requirements before we saw the new results.
- We used six newly trained models.
- We included every input order in the defined test groups.
- We made 624 comparisons with alternative connection patterns across the input orders.
- We kept the errors, failed predictions, code, and data.

Other researchers can repeat the work and check the explanation. The comparison with alternative rules helps show that the choice of connections matters.

Earlier research already shows that information order and internal connections can affect AI answers. Our possible contribution is this complete, planned test of a specific repair. We do not claim to have discovered a new general law of AI.

## What remains uncertain

These models are small. Their task has only four label-value pairs. The repair receives the correct pair relationships and original position labels.

One further prediction did not meet its requirement in two models. We predicted a minimum effect when we restored a blocked connection. The results did not show that minimum effect with the required confidence in those models.

The next useful steps are an external repeat of the test, larger models, and longer lists. We must also test repairs that find the pair relationships without being given them.

The results support a focused research paper. They do not guarantee that a journal or conference will accept it.

[Full results](experiment5/RESULTS.md) · [Reproduction guide](experiment5/REPRODUCE.md) · [Public registration](https://github.com/posix4e/ai-understanding-ai/commit/de745644b5c71853755e8d53cc9b3dec2abdc2ed)

Writing approach: short sentences, consistent terms, and one main point per paragraph, following [ASD-STE100 writing principles](https://www.asd-ste100.org/STE_faq.html).
