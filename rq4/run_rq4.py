"""RQ4: characteristics and retraining ablations using the existing team pipeline."""
import argparse,csv,hashlib,json,random,re,sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rq2'))
from protocol import load_team_data,project_rows,split_rows,save_protocol
from common.data import tokenize
from common.train import TrainConfig,train_experiment


def features(text):
    words=re.findall(r'\b\w+\b',text.lower());n=len(words);lines=text.splitlines()
    return {'characters':len(text),'words':n,'mean_word_length':float(np.mean([len(w) for w in words])) if n else 0,
        'type_token_ratio':len(set(words))/n if n else 0,'first100_type_token_ratio':len(set(words[:100]))/100 if n>=100 else np.nan,
        'approx_words_per_sentence':n/max(sum(bool(s.strip()) for s in re.split(r'[.!?]+',text)),1),
        'newlines_per100_words':text.count('\n')/max(n,1)*100,'punctuation_per100_words':len(re.findall(r'[^\w\s]',text))/max(n,1)*100,
        'bullet_lines_per100_words':sum(bool(re.match(r'^\s*(?:[-*+]\s|\d+[.)]\s)',line)) for line in lines)/max(n,1)*100,
        'heading_lines':sum(bool(re.match(r'^\s*#{1,6}\s',line)) for line in lines),'code_fences':text.count('```'),
        'uppercase_letter_fraction':sum(c.isupper() for c in text)/max(sum(c.isalpha() for c in text),1)}


def transform(text,condition):
    if condition=='baseline':return text
    if condition=='formatting_removed':return ' '.join(re.findall(r'\w+',text))
    if condition=='token_order_shuffled':
        tokens=tokenize(text);random.Random(int(hashlib.sha256(text.encode()).hexdigest()[:16],16)).shuffle(tokens)
        return ' '.join(tokens)
    raise ValueError(condition)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed/raid_balanced.csv')
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--ablations',action='store_true');p.add_argument('--smoke-test',action='store_true');args=p.parse_args()
    out=args.output_dir or ROOT/'rq4'/('results_smoke' if args.smoke_test else f'results/seed{args.seed}')
    if out.exists() and any(out.iterdir()):p.error('Results already exist; choose another --output-dir')
    originals,info=load_team_data(args.data,args.smoke_test);info['smoke_test']=args.smoke_test;save_protocol(out,info)
    training=[r for r in originals if r['split']=='train']
    f=pd.DataFrame([dict(id=r['id'],source_id=r['source_id'],model=r['model'],domain=r['domain'],**features(r['generation'])) for r in training])
    f.to_csv(out/'training_features.csv',index=False)
    columns=[c for c in f if c not in ('id','source_id','model','domain')]
    f.groupby('model')[columns].agg(['count','mean','median','std']).to_csv(out/'feature_summary.csv')
    f.groupby(['domain','model'])[columns].mean().to_csv(out/'features_by_domain.csv')
    labels=sorted(f.model.unique())
    for column in ['words','first100_type_token_ratio','newlines_per100_words','punctuation_per100_words']:
        fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([f.loc[f.model==label,column].dropna().values for label in labels],showfliers=False)
        ax.set_xticks(range(1,len(labels)+1),labels);plt.setp(ax.get_xticklabels(),rotation=20,ha='right')
        ax.set(ylabel=column,title='Training responses: '+column);fig.tight_layout();fig.savefig(out/f'{column}.png',dpi=160);plt.close(fig)
    words=[]
    for label in labels:
        counts=Counter(w for r in training if r['model']==label for w in re.findall(r'\b\w+\b',r['generation'].lower()));total=sum(counts.values())
        words.extend(dict(model=label,word=w,count=n,per1000_words=n/max(total,1)*1000) for w,n in counts.most_common(30))
    pd.DataFrame(words).to_csv(out/'frequent_words.csv',index=False)
    if args.ablations:
        base=project_rows(originals);summary=[]
        for condition in ('baseline','formatting_removed','token_order_shuffled'):
            rows=[dict(r,generation=transform(r['generation'],condition)) for r in base]
            config=TrainConfig(seed=args.seed,epochs=1 if args.smoke_test else args.epochs,input_mode='output',max_length=256,device=args.device)
            provenance={'research_question':'RQ4','condition':condition,'dataset_sha256':info['dataset_sha256'],'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'smoke_test':args.smoke_test}
            for model in ('cnn','lstm'):
                result=train_experiment(model,*split_rows(rows),out/'ablations'/condition/model,config,provenance)
                summary.append(dict(model=model,condition=condition,accuracy=result['accuracy'],macro_f1=result['macro_f1'],best_epoch=result['best_epoch'],test_responses=result['n'],smoke_test=args.smoke_test))
        s=pd.DataFrame(summary);b=s[s.condition=='baseline'].set_index('model')
        s['macro_f1_drop_from_baseline']=[b.loc[r.model,'macro_f1']-r.macro_f1 for r in s.itertuples()]
        s.to_csv(out/'ablation_summary.csv',index=False)
    (out/'analysis_notes.json').write_text(json.dumps({'descriptive_split':'train','descriptive_text':'full original response','classification_text':'first256 tokens before any ablation',
        'ablations_run':args.ablations,'limits':['Descriptive differences do not prove classifier reliance.','Raw lexical diversity depends on length; first100 excludes shorter responses.','Formatting removal preserves word content of the visible baseline window but changes token counts and layout.','Shuffling preserves the visible token multiset but alters syntax, phrases and line layout.','Each ablation refits training vocabulary and retrains; fixed single-seed results are exploratory.']},indent=2))
    print(f'RQ4 finished: {out}',flush=True)


if __name__=='__main__':main()
