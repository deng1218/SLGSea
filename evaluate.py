import argparse,json
from pathlib import Path
from slgsea.config import project_path
from slgsea.checkpoint import read_checkpoint,restore_model
from slgsea.model import SLGSea
from slgsea.data import SARPanopticDataset
from slgsea.evaluation import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path);p.add_argument('--output',type=Path,default=Path('outputs/evaluation'))
    p.add_argument('--device',default='cuda');p.add_argument('--max-images',type=int,default=None,help='Smoke-test subset only; omit for reported metrics')
    a=p.parse_args();checkpoint=read_checkpoint(a.checkpoint);cfg=checkpoint['config']
    model=SLGSea(cfg,checkpoint['text_cache']).to(a.device);restore_model(model,checkpoint['model'])
    dataset=SARPanopticDataset(a.data_root,'val',cfg['input_size'],False)
    report=evaluate(model,dataset,a.output,a.max_images);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
