# Add Allegra's RQ2/RQ4 using Amal's existing dataset and pipeline

This package applies to the existing repo at commit 2850ea1e9503695d19fc63780b996a40198356d4. It fills rq2/rq4, adds tests/test_rq2_rq4.py and a report contribution. It contains no data/, src/, rq1/, rq3/, root README or requirements replacements. No remote changes were pushed.

## Mac: fresh clone and branch

Use a fresh local folder to avoid mixing earlier downloaded projects:

```bash
cd ~/Desktop
git clone https://github.com/amalsony/llm-fingerprinting.git llm-fingerprinting-allegra
cd llm-fingerprinting-allegra
git switch -c allegra-rq2-rq4
```

If that Desktop folder already exists, open it and run git status before proceeding. Do not delete or force-reset existing work. If already on your contribution branch, do not create it again.

Download the NEW partner-compatible ZIP to Downloads. Keep the ZIP filename unchanged. Unpack its contents directly into the cloned repo:

```bash
unzip -o ~/Downloads/allegra_partner_RQ2_RQ4.zip -d ~/Desktop/llm-fingerprinting-allegra
python3 rq2/check_setup.py
```

The check verifies the exact data/processed/raid_balanced.csv and shared source files used for supplied results. The dataset stays in the clone and is not rebuilt. The ZIP replaces the empty RQ2/RQ4 placeholders and adds outputs. If you already added your own RQ2/RQ4 work, review that first before unpacking with -o.

## Optional: open locally in Codex/VS Code

Open VS Code > File > Open Folder > Desktop > llm-fingerprinting-allegra. A local Codex session can work with those files. Paste:

"Work in this repository on allegra-rq2-rq4. Review rq2/run_rq2.py and rq4/run_rq4.py. They must import the existing common.train pipeline and use data/processed/raid_balanced.csv with its saved splits. Preserve data, src/common, rq1 and rq3. Check python3 rq2/check_setup.py. Completed results are included, so do not retrain just to upload them. Explain any mismatches before changing shared files."

## Optional: verify/run code locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Full-run results are included, so rerunning is optional. To reproduce without overwriting:

```bash
python rq2/run_rq2.py --device cpu --output-dir rq2/results/reproduction
python rq4/run_rq4.py --device cpu --ablations --output-dir rq4/results/reproduction
```

## Commit and publish your branch

Review git status. It should show only rq2, rq4, tests/test_rq2_rq4.py, the new report contribution, and this guide as changed/new paths.

```bash
git status
git add rq2 rq4 tests/test_rq2_rq4.py report/allegra_RQ2_RQ4 START_HERE_ALLEGRA.md
git commit -m "Add RQ2 and RQ4 using existing team dataset and shared models"
git push -u origin allegra-rq2-rq4
```

GitHub requires collaborator permissions and authentication. If Git asks for credentials or rejects access, use the offered GitHub sign-in method; do not paste passwords or access tokens into chat. Then open the repo website and choose Compare & pull request > Create pull request. Your teammate reviews and merges it.

Your report contribution must be combined with team RQ1/RQ3, full member names, leader designation and confirmed lessons/experience. The supplement is not the complete team submission. Disclose LLM assistance and understand all submitted work.
