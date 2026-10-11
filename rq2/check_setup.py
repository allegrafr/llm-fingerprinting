"""Check dataset/shared code against the supplied completed run, without ML packages."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
config_path=ROOT/'rq2/results/seed42/output/cnn/config.json'
if not config_path.exists():
    sys.exit('Copy the completed RQ2 package into the repository first.')
metadata=json.loads(config_path.read_text());errors=[]
data=ROOT/'data/processed/raid_balanced.csv'
if not data.exists() or hashlib.sha256(data.read_bytes()).hexdigest()!=metadata['provenance']['dataset_sha256']:
    errors.append('Dataset differs from the one used for the supplied results.')
for relative,expected in metadata['code_sha256'].items():
    path=ROOT/'src/common'/relative
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        errors.append(f'Shared code differs: src/common/{relative}')
if errors:sys.exit('\n'.join(errors)+'\nReview the changes with your teammate before using the saved scores.')
print('PASS: exact team dataset and shared model/training code match the completed RQ2 run.')
print('No dataset replacement or model reimplementation is needed.')
