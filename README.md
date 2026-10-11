# MIDTERM Project 1 - LLM Fingerprinting LLM Identifying LLMs from Their Responses

CMPSC 448 Midterm Project

## Team Responsibilities

- RQ1 — LLM identification from generated responses: Amal
- RQ2 — Effect of user prompts on LLM identification: Allegra
- RQ3 — Cross-domain generalization of LLM fingerprints: Amal
- RQ4 — Characteristics distinguishing LLMs: Allegra

## Dataset

We use a balanced subset of [RAID](https://huggingface.co/datasets/liamdugan/raid) consisting of 1,600
LLM-generated responses.

Dataset structure:

- 4 LLM families
- 4 domains
- 100 matched prompts per domain
- each prompt answered by all 4 LLMs

Build the subset with `python3 scripts/build_dataset.py`. See
[data/README.md](data/README.md) for the exact filters, reproducibility manifest,
and source-grouped 80/10/10 train/validation/test split.
The dataset and shared CNN/LSTM pipeline are implemented. See
[rq1/README.md](rq1/README.md) for setup, training, and the first RQ1 results.
RQ3 domain-held-out experiments are next.

## Repository Structure

- `data/` — raw and processed datasets
- `scripts/` — dataset construction scripts
- `src/common/` — shared CNN, RNN, training, and evaluation code
- `rq1/` — RQ1 experiments
- `rq2/` — RQ2 experiments
- `rq3/` — RQ3 experiments
- `rq4/` — RQ4 experiments
- `report/` — final project report

## AI Use Disclosure

As permitted by the project guidelines, LLM assistance, specifically OpenAI Codex and ChatGPT, was used in the implementation of this project.
