"""Single-image inference: only a checkpoint and an image are required."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from slgsea.checkpoint import read_checkpoint,restore_model
from slgsea.model import SLGSea
from slgsea.panoptic.utils import id2rgb

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('outputs/prediction'))
    p.add_argument('--device',default='cuda');a=p.parse_args()
    checkpoint=read_checkpoint(a.checkpoint);cfg=checkpoint['config']
    model=SLGSea(cfg,checkpoint['text_cache']).to(a.device);restore_model(model,checkpoint['model']);model.eval()
    with Image.open(a.image) as im:
        im=im.convert('RGB');w,h=im.size
        arr=np.asarray(im.resize((cfg['input_size'],cfg['input_size']),Image.Resampling.BILINEAR)).copy()
    with torch.inference_mode():ids,segments=model([{'image':torch.from_numpy(arr).permute(2,0,1).float(),'height':h,'width':w}])[0]
    ids=ids.cpu().numpy();a.output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(id2rgb(ids).astype(np.uint8)).save(a.output/'panoptic.png')
    colors=np.zeros((*ids.shape,3),dtype=np.uint8)
    palette={0:[240,185,35],1:[55,135,200],2:[130,165,110]}
    for s in segments:colors[ids==s['id']]=palette[s['category_id']]
    Image.fromarray(colors).save(a.output/'preview.png')
    (a.output/'segments.json').write_text(json.dumps({'categories':{'0':'ship','1':'sea','2':'land'},'segments_info':segments},indent=2),encoding='utf-8')
    print(a.output.resolve())

if __name__=='__main__':main()
