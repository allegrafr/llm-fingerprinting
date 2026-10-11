# RQ4: What characteristics distinguish the generators?

Uses the team dataset and shared CNN/LSTM training pipeline described in rq2/README.md.

```bash
python rq4/run_rq4.py --device cpu --ablations
```

Without --ablations, only training-response feature analysis runs. Use --smoke-test for a small functionality check, or --output-dir rq4/results/new_run to avoid overwriting completed results.

Analyze full original training responses only: character/word counts, average word length, raw type-token ratio, first100-word type-token ratio (missing for shorter outputs), approximate words per sentence, newlines/punctuation/bullets per100 words, heading lines, code fences and uppercase fraction. Outputs include feature_summary.csv (count/mean/median/std), features_by_domain.csv, frequent_words.csv, per-row training_features.csv and four distribution plots. Sentence splitting is heuristic; full-response features may differ from those in the classifier's visible window. Frequent words are not neural attribution.

For neural ablations, first select the same first256 response tokens as the RQ2 output baseline. Train shared CNN and LSTM on baseline, formatting_removed and token_order_shuffled. Formatting removal retains the visible word content/case while removing punctuation and line layout; it does not expose additional response text. Token shuffling is deterministic per response and preserves the visible token multiset. Split, architecture, training hyperparameters and seed are retained. Vocabulary is refitted on transformed training data; checkpoints are selected on validation data; test is evaluated afterwards. Negative F1 drops mean improvement. Single-seed results are exploratory.

Ablation outputs use the shared directory/file format under results/seed42/ablations/CONDITION/MODEL. ablation_summary.csv reports held-out accuracy, macro-F1 and F1 decrease from baseline. This baseline should reproduce the RQ2 output-only run on the same device and software environment.

Descriptive differences do not establish classifier reliance. Shuffling simultaneously changes syntax, local phrases and layout; removing formatting changes token counts. These are not isolated causal interventions. Raw lexical diversity depends on length; first100 excludes shorter outputs, which limits comparisons across generators and domains.
