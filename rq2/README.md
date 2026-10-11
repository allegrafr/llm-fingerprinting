# RQ2: Does prompt context help overall or in certain domains?

These scripts import the team's existing common.train.TrainConfig/train_experiment and therefore reuse TextCNN, TextLSTM, tokenizer, optimizer, checkpoint selection, metrics and plots. They do not modify src/common, data, scripts/build_dataset.py, rq1 or rq3.

## Exact team dataset

Use data/processed/raid_balanced.csv at repository base commit 2850ea1e9503695d19fc63780b996a40198356d4. Its SHA-256 is f95ce226d32562d0f8b7e18946931b20be61ecdeab47d13af32a15c580685ca4. There are 1,600 responses, four source-model labels (chatgpt, llama-chat, mistral-chat, cohere-chat), and news/poetry/recipes/reviews domains. Each domain has 100 matched prompts. The original 80/10/10 source split is retained: 1,280 train, 160 validation and 160 test responses. All examples use sampling, repetition_penalty=no, attack=none. One source model represents each of four distinct families.

## Run

From the repository root, after installing its requirements:

```bash
python rq2/run_rq2.py --device cpu
```

For a small execution check use --smoke-test. To avoid replacing saved results use --output-dir rq2/results/new_run. Existing completed directories are protected against overwrite. CPU is the default; --device auto/mps/cuda are available but scores may differ across hardware.

## Controlled comparison

Run CNN and LSTM separately for input only, output only, and input+output. On copies of rows, retain first128 prompt tokens and first256 response tokens using the existing tokenizer. Output-only and combined conditions see exactly the same response tokens; combined adds the same prompt portion plus the shared newline separator (max_length385). Input-only max_length128, output-only max_length256. IDs, labels, domains and split assignments are unchanged. project_rows checks this token equivalence and split leakage.

All conditions fit their own vocabulary on visible training text only, using the team's Vocabulary implementation. Unlike the original RQ1 runner, which fits vocabulary on complete training outputs before encoding truncation, this RQ2 baseline fits on the visible output window. Accordingly, refit the output baseline; do not copy RQ1 scores into this comparison. The code and model architecture are reused, not the trained weights. The response evidence budget is kept constant to avoid prompt length displacing outputs or giving combined input extra response text.

Shared defaults: embedding128, hidden/channels128, dropout0.3, AdamW lr0.001/weight_decay0.0001, batch32, 15 epochs maximum, patience4, gradient clipping1.0, seed42. LSTM uses masked mean pooling, not final-state pooling. Choose maximum validation macro-F1, with lower validation loss breaking ties, then evaluate test. No test tuning.

## Results

results/seed42/summary.csv contains six runs. Each condition/model directory follows the shared experiment format: config.json, vocabulary.json, history.json, test_metrics.json, test_predictions.csv, confusion_matrix.png, learning_curves.png. Checkpoints are ignored by Git.

prompt_benefit.csv compares input+output minus output-only accuracy and macro-F1 overall and within each domain. domain_scores.csv includes all three conditions by domain. prompt_benefit_by_domain.png shows accuracy differences and 95% paired percentile bootstrap intervals from 2,000 matched-source resamples. Ten independent test prompts per domain means estimates are imprecise. Intervals are conditional on trained models; they do not account for training-seed variation or multiple domain comparisons. Domain slicing is not RQ3 domain-held-out transfer.

Because each identical prompt has all four labels, deterministic input-only accuracy is exactly25% on complete matched groups. Macro-F1 need not equal25%. This controlled dataset cannot identify user-selected platforms. Positive gains indicate prompts helped in this run, negative gains indicate worse performance; both zero/no benefit and mixed domain results are valid findings.
