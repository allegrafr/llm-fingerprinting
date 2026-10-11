# Shared RAID dataset

Build from the repository root with Python 3.9 or later:

```bash
python3 scripts/build_dataset.py
python3 -m unittest discover -s tests -v
```

The builder uses the Python standard library. The first run downloads about
802 MB to `data/raw/train_none.csv`; later runs reuse that file. Raw data is
ignored by Git. The output is CSV so it can be inspected and loaded with pandas
without adding a Parquet dependency.

## Source and selection

- [RAID dataset card](https://huggingface.co/datasets/liamdugan/raid)
- [Official code and downloads](https://github.com/liamdugan/raid)
- [Dugan et al., ACL 2024](https://aclanthology.org/2024.acl-long.674/)
- Input: the labeled RAID **train** partition without adversarial attacks.
  RAID's official test partition has hidden labels and is not used here.
- Models: `chatgpt`, `llama-chat`, `mistral-chat`, `cohere-chat`.
- Domains: `news`, `reviews`, `recipes`, `poetry`.
- Generation condition: `attack=none`, `decoding=sampling`,
  `repetition_penalty=no` for every model.
- Seed: 42; 100 matched sources per domain, four responses per source.

Only complete sources with a nonempty response from every model and exactly
identical prompts are eligible. Duplicate model rows for a source cause an
error. Across different sources, repeated source IDs, normalized prompts, or
normalized response texts are removed before sampling, retaining one complete
group in a reproducible hash order. Normalization is whitespace collapsing and
case folding, solely for duplicate detection. The saved text retains original
capitalization, punctuation, and formatting. No length filtering, truncation,
or vocabulary fitting happens during dataset construction.

This is a controlled sample of four older RAID generators and four writing
domains; findings should not be generalized to all current LLMs or coding/math.
Identical responses from different models to the same source are retained:
they are valid examples of ambiguity, and the whole source stays in one split.

## Outputs and evaluation contract

`data/processed/raid_balanced.csv` contains 1,600 responses from 400 sources.
Its fields are `id`, `source_id`, `domain`, `model`, `prompt`, `generation`,
`decoding`, `repetition_penalty`, `attack`, and `split`.

| Split | Sources per domain | Total responses | Responses per model |
| --- | ---: | ---: | ---: |
| train | 80 | 1,280 | 320 |
| validation | 10 | 160 | 40 |
| test | 10 | 160 | 40 |

`data/processed/dataset_manifest.json` records source and output SHA-256
checksums, the builder checksum, filters, seed, curation counts, and all source
IDs in each split. Selection and splitting use separate deterministic hash
orders; changing CSV row order does not change the result.

- RQ1: train CNN/LSTM on `generation`, predict `model`, and use the saved split.
- RQ2: reuse exactly the same rows and splits for the three input conditions.
  With identical prompts and all four labels present per source, a deterministic
  input-only classifier has exactly 25% accuracy on a complete evaluation split.
- RQ3: create separate domain-held-out folds; never use the held-out domain for
  vocabulary fitting, training, early stopping, or hyperparameter selection.
  The saved mixed-domain `split` column alone is not an RQ3 evaluation.
- Fit tokenizers/vocabularies on each experiment's training data only. Use
  validation for model selection and reserve test data for final evaluation.

Four-class balanced chance accuracy is 25%. The planned RQ3 comparison is
leave-one-domain-out, with all models retrained for each held-out domain.
The shared CNN/LSTM implementation and initial RQ1 results are documented in
`rq1/README.md`. RQ3 domain-held-out experiments are next.

The Hugging Face dataset card lists the dataset under MIT. Retain attribution
to RAID in the report. The manifest identifies exactly which public source
file was used; the generated subset does not add new data collection.
