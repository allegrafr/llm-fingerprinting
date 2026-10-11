"""RQ2: prompt benefit overall and by domain using the team's shared CNN/LSTM."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
from protocol import ROOT,load_team_data,project_rows,split_rows,save_protocol
from common.train import TrainConfig,train_experiment
from analyze import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/processed/raid_balanced.csv')
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke-test',action='store_true');args=p.parse_args()
    out=args.output_dir or ROOT/'rq2'/('results_smoke' if args.smoke_test else f'results/seed{args.seed}')
    for mode in ('input','output','both'):
        for model in ('cnn','lstm'):
            target=out/mode/model
            if target.exists() and any(target.iterdir()):p.error(f'Existing results at {target}; choose another --output-dir')
    original,info=load_team_data(args.data,args.smoke_test);rows=project_rows(original)
    info['smoke_test']=args.smoke_test;save_protocol(out,info)
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit='unavailable'
    summary=[]
    for mode,limit in [('input',128),('output',256),('both',385)]:
        config=TrainConfig(seed=args.seed,epochs=1 if args.smoke_test else args.epochs,input_mode=mode,max_length=limit,device=args.device)
        provenance={'research_question':'RQ2','dataset_sha256':info['dataset_sha256'],'base_commit':commit,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_mode':mode,'smoke_test':args.smoke_test,'prompt_budget':128,'response_budget':256}
        for model in ('cnn','lstm'):
            result=train_experiment(model,*split_rows(rows),out/mode/model,config,provenance)
            summary.append({'model':model,'input_mode':mode,'seed':args.seed,'accuracy':result['accuracy'],'macro_f1':result['macro_f1'],'best_epoch':result['best_epoch'],'test_responses':result['n'],'smoke_test':args.smoke_test})
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    summarize(out)
    print(f'RQ2 finished: {out}',flush=True)
    if args.smoke_test:print('Smoke scores are not reportable results.')


if __name__=='__main__':main()
