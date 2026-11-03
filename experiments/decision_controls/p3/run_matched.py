"""Run one frozen model's full P3 source tasks with matched legal readouts."""
import argparse,json,os
from pathlib import Path
from task_runtime import MODEL_IDS,OriginalPromptModel,configure_root,run_horizon,run_wcst,write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=MODEL_IDS,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',choices=['horizon','wcst'],default=['horizon','wcst'])
    p.add_argument('--readouts',nargs='+',choices=['argmax','sample'],default=['argmax','sample'])
    p.add_argument('--conditions',nargs='+',choices=['horizon:argmax','horizon:sample','wcst:argmax','wcst:sample'],
                   help='Explicit missing conditions, avoiding a model reload for a non-Cartesian condition set.')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh output required; prior products are retained.')
    args.output.mkdir(parents=True)
    configure_root(args.root)
    model=OriginalPromptModel(args.model,args.root,args.device)
    write_json(args.output/'model_metadata.json',model.metadata)
    completed=[]
    conditions = ([c.split(':') for c in args.conditions] if args.conditions
                  else [(task,readout) for task in args.tasks for readout in args.readouts])
    if len({tuple(c) for c in conditions}) != len(conditions):
        raise ValueError('Do not schedule a condition twice.')
    for task, readout in conditions:
        out=args.output/f'{task}_{readout}'
        fn=run_horizon if task=='horizon' else run_wcst
        completed.append(fn(model,args.root,out,readout))
    write_json(args.output/'summary.json',{'status':'complete','model':args.model,'conditions':completed})

if __name__=='__main__': main()
