"""Download pretrained model weights."""
import argparse,json,urllib.request
from pathlib import Path
from slgsea.config import ROOT
from slgsea.language.encoder import file_sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['coco_panoptic_r50','openai_clip_rn50'],default='coco_panoptic_r50')
    p.add_argument('--output-dir',type=Path,default=Path('weights'));a=p.parse_args()
    spec=json.loads((ROOT/'assets/pretrained_weights.json').read_text(encoding='utf-8'))[a.model]
    a.output_dir.mkdir(parents=True,exist_ok=True);dest=a.output_dir/spec['filename']
    if dest.exists():
        if file_sha256(dest)!=spec['sha256']:raise ValueError(f'Existing file hash mismatch: {dest}')
        print('Verified existing weights:',dest);return
    part=dest.with_suffix(dest.suffix+'.part')
    print('Downloading weights:',spec['url'],flush=True)
    urllib.request.urlretrieve(spec['url'],part)
    if file_sha256(part)!=spec['sha256']:raise ValueError(f'Download SHA-256 mismatch; incomplete/unverified file retained as {part}')
    part.replace(dest);print('Verified:',dest)

if __name__=='__main__':main()
