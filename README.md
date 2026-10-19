# Assignment 3: Naive Bayes Spam Filter

In this assignment you will implement a multinomial Naive Bayes classifier that labels SMS messages as spam or ham (legitimate), and evaluate its accuracy on a held-out test set.

## Setup

To setup the environment, run the following commands.

```
conda env create -f environment.yml
conda activate ai-search
```

## Files

| File | |
| --- | --- |
| `naive_bayes.py` | The file that holds your naive bayes classifier. |
| `main.py` | Runs your classifier on the worksheet, the test set, or a message you type. |
| `tests/` | Sample tests. |
| `environment.yml` | Python package list. |
| `data/` | `worksheet.csv` and `worksheet_test.csv`, for solving by hand if wanted. `sms_train.csv` (100 messages) and `sms_test.csv` (40 messages). |
| `model.py`, `data.py` | The model and the message reader. Do not modify. |

## Tasks

1. Implement `train`.
2. Implement `prior` and `likelihood`.
3. Implement `log_score` and `classify`.
4. Implement `accuracy`, then run `python main.py evaluate` and examine the messages your classifier labels incorrectly.

## Background

Naive Bayes scores each candidate label using two quantities estimated from the training data:

- **Prior.** The proportion of training messages that carry the label. If 35 of 100 training messages are spam, the prior for spam is 0.35.
- **Likelihood.** For each word, the proportion of all words in that label's training messages that are this word. For example, "prize" occurs far more often in spam than in ham, so its likelihood is higher for spam.

A message's score for a label is the prior multiplied by the likelihood of each word in the message. The classifier assigns the label with the highest score. The method is called *naive* because it treats each word as independent evidence and ignores word order: "win cash" and "cash win" receive the same score.

Two refinements are needed in practice:

- **Laplace smoothing.** A word that never appears with a label in training would have a likelihood of 0 for that label, which would make the whole product 0 regardless of the other words. Adding a constant `k` (typically 1) to every count ensures that no likelihood is zero.
- **Log probabilities.** The product of many small probabilities quickly becomes smaller than the smallest positive number a floating-point value can represent, and it underflows to 0. Because `log(a × b) = log(a) + log(b)` and the logarithm is increasing, comparing sums of log probabilities gives the same ranking as comparing the products, without underflow. Use the natural logarithm, `math.log`.

### Relation to AIMA

This assignment implements the Naive Bayes model described in AIMA 4th edition, Section 12.6 (Naive Bayes Models) and Section 20.2.2 (Naive Bayes models). In the book's notation, with class variable *C* and observed features *x*₁, …, *x*ₙ, the model is

```
P(C | x₁, …, xₙ) = α P(C) ∏ᵢ P(xᵢ | C)
```

where α is the normalising constant that makes the probabilities sum to 1. The functions in this assignment correspond to that formula as follows:

| AIMA | This assignment |
| --- | --- |
| class variable *C*, with value *c* | a label, `"spam"` or `"ham"` |
| features *x*₁, …, *x*ₙ | the words of a message |
| P(*C* = *c*) | `prior(model, c)` |
| P(*xᵢ* \| *C* = *c*) | `likelihood(model, w, c, k)` |
| log( P(*c*) ∏ᵢ P(*xᵢ* \| *c*) ) | `log_score(model, words, c, k)` |
| argmax over *c* of P(*c* \| *x*₁, …, *xₙ*) | `classify(model, words, k)` |
| estimating P(*C*) and P(*xᵢ* \| *C*) from counts (maximum likelihood, Section 20.2) | `train(examples)` |

The assignment differs from the book in three respects:

- **Normalisation.** `classify` compares labels without computing α. Because α is the same for every label, it does not change which label scores highest.
- **Feature model.** AIMA's text classification example in Section 12.6 uses one Boolean variable per word, recording whether that word occurs in the document. This assignment uses the *multinomial* variant: each occurrence of a word contributes a factor, and P(*xᵢ* \| *C*) is the word's share of all words seen with that label.
- **Smoothing.** The maximum-likelihood estimates from Section 20.2 assign probability 0 to unseen events. The `k` in `likelihood` applies add-*k* (Laplace) smoothing to avoid this. With `k=0`, `likelihood` returns the unsmoothed maximum-likelihood estimate.

## The model

`train` returns a `Model` (from `model.py`) with four fields:

| Field | Holds | Worksheet example |
| --- | --- | --- |
| `class_counts[c]` | how many training messages have label `c` | `class_counts["spam"] == 2` |
| `word_counts[c][w]` | how many times word `w` appeared in messages labelled `c` | `word_counts["spam"]["cash"] == 2` |
| `total_words[c]` | how many words in total appeared in messages labelled `c` | `total_words["spam"] == 6` |
| `vocabulary` | the set of every word seen in training | 9 words |

The counts are `collections.Counter` objects, which return 0 for any key not yet counted, so `model.class_counts[label] += 1` needs no initialisation. `model.labels` returns the labels in alphabetical order, and `model.num_examples` returns the number of training messages.

Each message is represented as a list of words. `data.py` lowercases and tokenizes the text before it reaches your code, so `"Win $1000 NOW!"` becomes `["win", "1000", "now"]`.

## Pseudocode

```
function TRAIN(examples) returns a model
    model ← a new, empty Model
    for each (words, label) in examples do
        class_counts[label] ← class_counts[label] + 1
        for each w in words do
            word_counts[label][w] ← word_counts[label][w] + 1
            total_words[label] ← total_words[label] + 1
            add w to vocabulary
    return model

function PRIOR(model, c) returns a probability
    return class_counts[c] / num_examples

function LIKELIHOOD(model, w, c, k) returns a probability
    return (word_counts[c][w] + k) / (total_words[c] + k × |vocabulary|)

function LOG-SCORE(model, words, c, k) returns a number
    score ← log PRIOR(model, c)
    for each w in words do
        if w is in vocabulary then
            score ← score + log LIKELIHOOD(model, w, c, k)
    return score

function CLASSIFY(model, words, k) returns a label
    best, best-score ← null, −∞
    for each c in labels do
        s ← LOG-SCORE(model, words, c, k)
        if s > best-score then
            best, best-score ← c, s
    return best

function ACCURACY(model, examples, k) returns a fraction
    correct ← 0
    for each (words, label) in examples do
        if CLASSIFY(model, words, k) = label then
            correct ← correct + 1
    return correct / |examples|
```

`|vocabulary|` means the number of words in the vocabulary, `len(model.vocabulary)`. `|examples|` means `len(examples)`.

## Worked example

`data/worksheet.csv` has five training messages:

| Label | Message |
| --- | --- |
| spam | win cash now |
| spam | free cash prize |
| ham | lunch now |
| ham | free for lunch |
| ham | call me now |

Compute the following values by hand before you begin implementing, and confirm that your implementation reproduces them:

- `prior(model, "spam")` = 2 / 5 = 0.4
- `likelihood(model, "cash", "spam")` = (2 + 1) / (6 + 9) = 0.2
- `likelihood(model, "cash", "ham")` = (0 + 1) / (8 + 9) ≈ 0.0588
- `classify(model, ["free", "cash", "now"])` = `"spam"`

`python main.py worksheet` prints every count, prior and likelihood for this example.

## Requirements

The tests check exact numbers, so follow the pseudocode exactly:

- Count every occurrence of a word. "money money money" adds 3 to `word_counts["spam"]["money"]`.
- Use the whole vocabulary in the likelihood's denominator, not just the words seen with label `c`.
- Use the `k` argument passed in. Do not hard-code 1.
- In `log_score`, skip words that are not in the vocabulary.
- Add logarithms. Do not multiply probabilities. Use `math.log`, not `math.log10`.
- In `classify`, try labels in the order of `model.labels`, and keep the first one on a tie (`>`, not `≥`).
- Call your own functions. `classify` should use `log_score`, and `log_score` should use `prior` and `likelihood`.

## Testing

The tests in `tests/` are a representative sample. They cover the worked example and the SMS data, but not the edge cases. Grading uses a more extensive hidden test suite that checks every rule in Requirements, so passing the sample tests does not guarantee full marks.

Before submitting, identify the cases the sample tests do not cover and verify your implementation's behaviour on them, either by adding tests to `tests/test_naive_bayes.py` or by using `python main.py classify`. Consider, in particular, words absent from the training vocabulary, very long messages, ties between labels, and values of `k` other than 1.

```
pytest                              # the sample tests
pytest -k TestTraining              # one part: also TestProbabilities, TestClassification, TestEvaluation
python main.py worksheet            # counts, priors and likelihoods on the worked example
python main.py evaluate             # accuracy on sms_test, and the messages it gets wrong
python main.py --k 0.1 evaluate     # the same, with less smoothing
python main.py classify "Claim your free prize now"
python main.py words                # the words that most strongly indicate each label
```

## Submission

Submit `naive_bayes.py`.

## Grading

| Component | Points |
| --- | --- |
| Training | 25 |
| Probabilities | 30 |
| Classification | 30 |
| Evaluation | 15 |
| **Total** | **100** |
