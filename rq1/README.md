# RQ1: identify the LLM from its response

## Run locally

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-rq1.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python rq1/run_rq1.py --output-dir rq1/results/seed42
```

The runner defaults to both models and selects CUDA, Apple MPS, or CPU in that
order according to availability. Use `--device cpu` to force CPU execution.
The tested dependency versions are in `requirements-rq1.txt`; `requirements.txt`
retains the team's broader dependency list. The existing `.venv` is ready on
Amal's Mac and is ignored by Git.

Results directories are never overwritten. For a later independent seed, use
both `--seed 43` and `--output-dir rq1/results/seed43`. A quick pipeline check
can use `--epochs 1 --output-dir /tmp/rq1-smoke`; do not report it as a full run.

## What the code does

1. Read the fixed source-grouped split: 1,280 training, 160 validation, and 160
   test responses. Only `generation` is used as classifier input; the label is
   `model`. IDs, domains, prompts, and split names are never classifier features.
2. Tokenize into words, punctuation, and newline tokens, preserving case. Build
   the vocabulary from training text only, keep tokens occurring at least twice,
   and cap it at 20,000 entries including padding and unknown tokens.
3. Encode the first 256 tokens of each response; unseen tokens become `<UNK>`.
   Pad within a batch and retain the actual lengths. Each run records how many
   responses were truncated. These are tokens from our regex, not LLM tokens.
4. Learn randomly initialized 128-dimensional embeddings together with the
   classifier. No pretrained LLM or pretrained word embedding is used.
5. Train with AdamW, learning rate 0.001, weight decay 0.0001, batch size 32,
   dropout 0.3, and gradient clipping at norm 1.0. Train at most 15 epochs, with
   patience 4. Select the checkpoint with highest validation macro F1 (lower
   validation loss breaks ties), restore it, and then evaluate the test split.

The **CNN** uses 128 filters at each of widths 3, 4, and 5. ReLU and max pooling
produce one value per filter; their 384 values feed dropout and a four-class
linear layer. Pooling excludes windows beyond the text. Responses shorter than
a filter receive one zero-padded window.

The **RNN is an LSTM**, with one forward layer and 128 hidden units. Mean pooling
over real token states summarizes the response, followed by dropout and a
four-class linear layer. Padding states do not enter the mean. Because the LSTM
is forward-only, later padding cannot change earlier real-token states.

Both models return raw logits to cross-entropy loss. Softmax is used only when
saving prediction probabilities. Implementation is in `src/common/`; the RQ1
runner orchestrates the experiment. The shared API accepts three sets of rows
so RQ3 can later supply domain-held-out splits and fit fresh vocabularies/models.

## Saved evidence

Each model's output directory contains:

- `config.json`: all hyperparameters, device, package versions, source-code
  hashes, dataset hash, source IDs, vocabulary size, and truncation counts.
- `vocabulary.json` and `best.pt`: preprocessing and selected model state.
  Checkpoints remain local because the repository ignores `*.pt`.
- `history.json` and `learning_curves.png`: train loss and validation metrics.
- `test_metrics.json`: accuracy, macro F1, per-model precision/recall/F1,
  confusion matrix, and per-domain scores.
- `test_predictions.csv`: one prediction and class-probability vector per test
  response, linked back to its generation/source ID.
- `confusion_matrix.png`: true models on rows and predicted models on columns.

The run directory also contains a summary CSV. Class order is saved explicitly.
Random seeds are fixed, but floating-point results can vary across devices and
library versions. A single seed is an initial result, not an estimate of
training variability. The test has 160 responses from 40 sources, so the four
responses sharing a source are not independent observations.

Balanced chance accuracy is 25%. Compare both architectures on the same split;
do not choose hyperparameters or rerun selectively based on test scores.
RQ3 still requires a separate domain-held-out experiment; these results are
within the mixed-domain RQ1 setting.

PyTorch references: [LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)
and [MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html).

## Initial results: seed 42

These are the completed local MPS run in `results/seed42`, using the unchanged
hyperparameters above. Checkpoint selection used validation only.

| Model | Test accuracy | Test macro F1 | Selected epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| CNN | 70.625% (113/160) | 0.7012 | 6 | 10 |
| LSTM | 57.500% (92/160) | 0.5536 | 10 | 14 |

| Test domain | CNN accuracy | LSTM accuracy | Responses |
| --- | ---: | ---: | ---: |
| News | 67.5% | 35.0% | 40 |
| Poetry | 52.5% | 50.0% | 40 |
| Recipes | 85.0% | 80.0% | 40 |
| Reviews | 77.5% | 65.0% | 40 |

Both classifiers exceeded the 25% chance-accuracy reference on this split.
The CNN performed better in this run; this does not establish that CNNs are
always better than LSTMs. Training loss continued falling while validation
performance fluctuated, indicating overfitting and supporting early stopping.
Recipes were easiest for both classifiers. These per-domain results still use
training data from every domain and therefore do not answer RQ3.

The 256-token cap truncated 875/1,280 training, 106/160 validation, and 115/160
test responses. The classifiers therefore used only the beginning of most
responses. This is a limitation to disclose, alongside the small test set,
single training seed, and four older generator classes. No hyperparameters
were changed after observing these test scores.
