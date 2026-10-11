"""Paired RQ2 comparisons conditional on fixed trained models."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score


def score(rows,column,labels):
    return accuracy_score(rows.true_model,rows[column]),f1_score(rows.true_model,rows[column],labels=labels,average='macro',zero_division=0)


def matrix_score(m):
    den=m.sum(0)+m.sum(1)
    return np.trace(m)/m.sum(),np.divide(2*np.diag(m),den,out=np.zeros(len(m)),where=den!=0).mean()


def bootstrap(part,labels):
    ids={x:i for i,x in enumerate(labels)};matrices=[]
    for _,group in part.groupby('source_id'):
        pair=[]
        for col in ('output_prediction','both_prediction'):
            m=np.zeros((len(labels),len(labels)))
            for r in group.itertuples():m[ids[r.true_model],ids[getattr(r,col)]]+=1
            pair.append(m)
        matrices.append(pair)
    arrays=np.asarray(matrices);rng=np.random.default_rng(42);deltas=[]
    for _ in range(2000):
        sampled=arrays[rng.integers(0,len(arrays),len(arrays))].sum(0)
        a=matrix_score(sampled[0]);b=matrix_score(sampled[1]);deltas.append([b[0]-a[0],b[1]-a[1]])
    return np.quantile(deltas,[.025,.975],axis=0)


def summarize(root):
    root=Path(root);domain_scores=[];benefits=[]
    for model in ('cnn','lstm'):
        predictions={mode:pd.read_csv(root/mode/model/'test_predictions.csv') for mode in ('input','output','both')}
        labels=sorted(predictions['output'].true_model.unique())
        for mode,frame in predictions.items():
            for domain,part in frame.groupby('domain'):
                acc,f1=score(part,'predicted_model',labels)
                domain_scores.append(dict(model=model,input_mode=mode,domain=domain,accuracy=acc,macro_f1=f1,test_responses=len(part),sources=part.source_id.nunique()))
        a=predictions['output'].rename(columns={'predicted_model':'output_prediction'})
        b=predictions['both'][['id','true_model','predicted_model']].rename(columns={'true_model':'true_check','predicted_model':'both_prediction'})
        joined=a.merge(b,on='id',validate='one_to_one')
        if not joined.true_model.equals(joined.true_check):raise ValueError('Mismatched test labels')
        for domain,part in [('overall',joined)]+list(joined.groupby('domain')):
            acc0,f10=score(part,'output_prediction',labels);acc1,f11=score(part,'both_prediction',labels);ci=bootstrap(part,labels)
            benefits.append(dict(model=model,domain=domain,test_responses=len(part),sources=part.source_id.nunique(),output_accuracy=acc0,both_accuracy=acc1,accuracy_gain=acc1-acc0,accuracy_gain_ci_low=ci[0,0],accuracy_gain_ci_high=ci[1,0],output_macro_f1=f10,both_macro_f1=f11,macro_f1_gain=f11-f10,macro_f1_gain_ci_low=ci[0,1],macro_f1_gain_ci_high=ci[1,1]))
    pd.DataFrame(domain_scores).to_csv(root/'domain_scores.csv',index=False)
    gains=pd.DataFrame(benefits);gains.to_csv(root/'prompt_benefit.csv',index=False)
    domains=sorted(gains.loc[gains.domain!='overall','domain'].unique());fig,ax=plt.subplots(figsize=(8,4))
    for i,model in enumerate(('cnn','lstm')):
        rows=gains[(gains.model==model)&(gains.domain!='overall')].set_index('domain').loc[domains]
        x=np.arange(len(domains))+(i-.5)*.2;color=['tab:blue','tab:orange'][i]
        ax.scatter(x,rows.accuracy_gain*100,label=model.upper(),color=color)
        ax.vlines(x,rows.accuracy_gain_ci_low*100,rows.accuracy_gain_ci_high*100,color=color)
    ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(range(len(domains)),domains)
    ax.set(ylabel='Accuracy gain from prompt (percentage points)',title='RQ2: prompt benefit within each test domain');ax.legend();fig.tight_layout();fig.savefig(root/'prompt_benefit_by_domain.png',dpi=160);plt.close(fig)
    (root/'comparison_notes.json').write_text(json.dumps({'bootstrap_unit':'source_id matched group','resamples':2000,'seed':42,'interval':'95% paired percentile bootstrap','limits':'Conditional on fixed trained models; no training-seed uncertainty or multiple-comparison adjustment. Ten test prompts per domain. Domain slicing is not RQ3 domain-held-out transfer.'},indent=2))
