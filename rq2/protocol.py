"""RQ2/RQ4 adapters around the existing team pipeline; no dataset mutation."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common.data import read_dataset,check_splits,tokenize
PROMPT_TOKENS=128
RESPONSE_TOKENS=256


def load_team_data(path,smoke=False):
    rows=read_dataset(path)
    splits=[[r for r in rows if r['split']==s] for s in ('train','validation','test')]
    check_splits(*splits)
    groups=defaultdict(list)
    for r in rows: groups[r['source_id']].append(r)
    labels=sorted({r['model'] for r in rows})
    if len(labels)!=4 or len(rows)!=1600 or len(groups)!=400:
        raise ValueError('Expected the team 1,600-row, four-family, 400-source RAID subset')
    for group in groups.values():
        if len(group)!=4 or {r['model'] for r in group}!=set(labels):raise ValueError('Incomplete matched source')
        for field in ('prompt','domain','split'):
            if len({r[field] for r in group})!=1:raise ValueError(f'Mismatched {field} within a source')
    domains=sorted({r['domain'] for r in rows})
    if domains!=['news','poetry','recipes','reviews']:raise ValueError(f'Unexpected domains: {domains}')
    info={'dataset_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'rows':len(rows),'sources':len(groups),'models':labels,'domains':domains,
          'rows_by_split':dict(Counter(r['split'] for r in rows)), 'prompt_budget':PROMPT_TOKENS,'response_budget':RESPONSE_TOKENS}
    for split,count in [('train',80),('validation',10),('test',10)]:
        for domain in domains:
            if len({r['source_id'] for r in rows if r['split']==split and r['domain']==domain})!=count:
                raise ValueError('Expected the original 80/10/10 source splits per domain')
    if smoke:
        kept=set()
        for split in ('train','validation','test'):
            for domain in domains:
                kept.update(sorted({r['source_id'] for r in rows if r['split']==split and r['domain']==domain})[:2])
        rows=[r for r in rows if r['source_id'] in kept]
    return rows,info


def clip_text(text,budget):
    tokens=tokenize(text)
    return text if len(tokens)<=budget else ' '.join(tokens[:budget])


def project_rows(rows):
    # Same visible response content across output-only and combined experiments.
    projected=[dict(r,prompt=clip_text(r['prompt'],PROMPT_TOKENS),generation=clip_text(r['generation'],RESPONSE_TOKENS)) for r in rows]
    for original,projected_row in zip(rows,projected):
        assert tokenize(projected_row['generation'])==tokenize(original['generation'])[:RESPONSE_TOKENS]
        assert tokenize(projected_row['prompt'])==tokenize(original['prompt'])[:PROMPT_TOKENS]
    check_splits(*split_rows(projected))
    return projected


def split_rows(rows):return [[r for r in rows if r['split']==s] for s in ('train','validation','test')]


def save_protocol(directory,info):
    directory.mkdir(parents=True,exist_ok=True)
    info=dict(info,baseline_note='Output-only refits training vocabulary on visible first-256-token responses. It reuses team models, training settings, dataset and splits; it is not copied RQ1 metrics.',
              controlled_comparison='Input-only: first128 prompt tokens; output-only: first256 response tokens; both: those same components separated by the shared newline, maximum385 tokens. Vocabularies use visible training text only.')
    (directory/'protocol.json').write_text(json.dumps(info,indent=2)+'\n')
