import argparse,json
from pathlib import Path
from slgsea.config import load_config
from slgsea.engine import run_training

def main():
    p=argparse.ArgumentParser(description='Train the full SLGSea method; no ablation queue')
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--data-root',type=Path);p.add_argument('--text-cache',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--init-weights',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--device');p.add_argument('--workers',type=int)
    p.add_argument('--max-steps',type=int,help='Maximum optimizer updates for this run')
    p.add_argument('--skip-eval',action='store_true',help='Skip validation explicitly, e.g. a short smoke run')
    args=p.parse_args();cfg=load_config(args.config)
    for arg,key in [('data_root','data_root'),('text_cache','text_cache'),('output','output_dir')]:
        value=getattr(args,arg)
        if value is not None:cfg[key]=str(value.resolve())
    if args.device:cfg['device']=args.device
    if args.workers is not None:cfg['solver']['workers']=args.workers
    if args.init_weights and args.resume:p.error('Choose --init-weights or --resume, not both')
    print(json.dumps(run_training(cfg,args.init_weights,args.resume,args.max_steps,args.skip_eval),indent=2))

if __name__=='__main__':main()
