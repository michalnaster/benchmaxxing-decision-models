<p align="center">
  <img src="assets/logo.png" alt="BenchMaxx logo" width="320">
</p>

# Benchmaxxing Decision Models

**benchmaxx** turns a frozen open LLM into a decision model you can fit to your own data: a small correction head
learns from the LLM's hidden states, and a domain gate keeps it at about the LLM's own level on data unlike yours.
With a few thousand labelled decisions it trains in minutes on a laptop, after one pass of the LLM over the data.

This repository holds the model architecture and the training recipe behind **benchmaxx-4B** (one gated head per
benchmark) and **benchmaxx-4B-generalist** (one head for everything), both on a frozen
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).

## How it works

A decision is one question about one document, answered by choosing one of its options. Most LLM-based decision
models ask an LLM for one letter. benchmaxx keeps that LLM frozen and fixes what reading it naively gets wrong, one
step per problem:

| step | problem | module |
|---|---|---|
| 1. Read the LLM twice | the answer depends on the order of the options | `prompt.py` |
| 2. Keep what the LLM saw | its answer alone throws away what it worked out | `reader.py` |
| 3. Learn a correction | the LLM doesn't know your domain | `correction_head.py` |
| 4. Add a domain gate | anything fitted to your data gets worse on data unlike it | `domain_gate.py` |
| 5. Train it all | the gate needs honest answers, without extra held-out data | `training.py` |

**1. Read twice.** Each decision becomes a chat prompt whose assistant turn is prefilled with `{"answer": "`, so the
next token is an option code. It is read with the options in their order and reversed.

**2. Keep the hidden states.** From each reading, at layers 12, 20, 28 and 32: the *answer vector* (the hidden state
at the last prompt token, where the LLM is about to answer) and one *option vector* per option (its tokens' hidden
states, averaged), plus the LLM's probability for each option code. The two readings are averaged.

**3. Correction head** (2.24M parameters). Each option becomes a token: its option vector, projected, plus five
numbers on how the LLM rated it. The answer vector enters as one token per layer, so each option's attention chooses
which layers to read (*attention over depth*). Two cross-attention blocks give each option a correction, and a gate on
statistics of the LLM's answer decides how much of it to apply. The gate starts almost closed: before training, the
head *is* the LLM.

**4. Domain gate.** Per decision, a weight `w` mixes the corrected answer with the LLM's:
`final = w · corrected + (1 − w) · LLM`. It is a logistic model on how far the answer vector lies from the training
data's manifold (residual outside its top 64 principal directions, distance to the 10 nearest training decisions) and
on how the two answers look.

**5. Cross-fitting.** Two heads, each trained on one half of the data (split by case) and early-stopped on the other
by KL divergence; then one head refit on everything. The gate is fitted on the halves' answers on the data they never
trained on (in domain) and on the shipped head's answers on other domains (out of domain).

The final answer's top probability is the model's confidence. No temperature or confidence model is fitted afterwards.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                     # creates .venv with the package and the dev tools (pytest, ruff)
```

Python 3.11 or newer. Reading needs a GPU or Apple silicon with room for the LLM: Qwen3.5-4B takes about 9 GB in
bfloat16, plus activations for the longest documents (up to 8,192 tokens).

## Data

Decisions are JSON Lines, one decision per line:

```json
{"case_id": "ticket-17",
 "state": "I was charged twice, and the app crashes when I open the invoice.",
 "question": {"text": "Which team should handle this?", "type": "choice",
              "options": [{"key": "billing", "text": "payments and invoices"},
                          {"key": "technical", "text": "product faults"},
                          {"key": "sales", "text": "new business"}]},
 "gold": [0.7, 0.25, 0.05],
 "dataset": "tickets"}
```

- `type` is `choice`, `yes_no` (options default to *no*, *yes*) or `score` (levels from lowest to highest).
- `gold` is a probability per option; `"label": 0` can be given instead. Neither is needed for prediction.
- `case_id` groups the questions asked about one document; cases are never split during training.
- `dataset` names the source, used for per-dataset weighting and gold softening.

See [`examples/decisions.jsonl`](examples/decisions.jsonl).

## Usage

### Command line

```bash
# 1-2. read the training decisions once with the frozen LLM (the slow part)
uv run benchmaxx read --decisions train.jsonl --out features/train

# 3-5. train the correction head; with out-of-domain features, also fit the domain gate
uv run benchmaxx read --decisions other_domain.jsonl --out features/other
uv run benchmaxx train --features features/train --out models/mine --out-of-domain other=features/other

# evaluate on labelled test features, or answer new decisions
uv run benchmaxx read --decisions test.jsonl --out features/test
uv run benchmaxx evaluate --model models/mine --features features/test
uv run benchmaxx predict --model models/mine --decisions new.jsonl --out predictions.jsonl
```

`--config` takes a JSON file of training settings; [`configs/`](configs) holds those of the benchmaxx-4B specialists
and of benchmaxx-4B-generalist (small datasets repeated within each epoch, early stopping on the KL averaged per
dataset, softened gold for generated data).

### Python

```python
from benchmaxx import Benchmaxx, TrainingConfig, read_decisions, train
from benchmaxx.reader import LLMReader

reader = LLMReader("Qwen/Qwen3.5-4B")
features = reader.read(read_decisions("train.jsonl"))
other = reader.read(read_decisions("other_domain.jsonl"))

model = train(features, TrainingConfig(), out_of_domain={"other": [other]})
model.save("models/mine")

model = Benchmaxx.load("models/mine")
for prediction in model.predict(read_decisions("new.jsonl"), reader):
    print(prediction.answer, prediction.confidence)
```

## Choosing what to automate

The confidence estimates how much of the gold's probability falls on the answer, so with soft gold it is lower than
the probability of being "right". Pick the automation threshold from the accuracy you measure at each coverage on
held-out decisions of your own (the risk-coverage curve), not from the confidence value.

## Repository layout

```
src/benchmaxx/
  schema.py           decisions and questions; JSON Lines input
  prompt.py           the prompt and its exact token spans
  reader.py           runs the frozen LLM and pools its hidden states
  features.py         feature storage and padded batches
  correction_head.py  the correction head
  domain_gate.py      the manifold, the gate module and its fit
  losses.py           log loss + Brier score / ranked probability score
  metrics.py          accuracy, KL, ECE, AURC
  training.py         cross-fitting, refit, gate fitting
  model.py            the trained model: head + gate, save and load
  cli.py              the benchmaxx command
assets/               the logo
configs/              training settings of the published models
examples/             a small decisions file
tests/                unit tests (no downloads needed)
```

## Reproducibility

Inference reproduces the models of the article exactly: with the same weights, the prompts, the features and the
answers match the code they were produced with. Training follows the same recipe with one simplification: the
projector's standardisation statistics are computed from the vectors the head reads (answer and option vectors),
so a model retrained here matches the published ones closely but not bit for bit. Results on Apple MPS and on CPU can
differ slightly, because MPS computes float32 matrix products at lower precision.

## Testing

```bash
uv run pytest
uv run ruff check .
```
