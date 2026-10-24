"""COCO-panoptic data with training-only image-aligned language."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .prompts import training_prompts

def rgb2id(rgb):
    x=np.asarray(rgb,dtype=np.int64)
    return x[...,0]+256*x[...,1]+65536*x[...,2]

def load_text_rows(root,split='train'):
    files=list((Path(root)/'text_library').glob(f'*_{split}.json'))
    if len(files)!=1:raise ValueError(f'Expected one text_library/*_{split}.json under {root}, found {len(files)}')
    rows=json.loads(files[0].read_text(encoding='utf-8-sig'))
    if len({str(x['image_id']) for x in rows})!=len(rows):raise ValueError('Duplicate image IDs in text library')
    return {str(x['image_id']):x for x in rows}

class SARPanopticDataset(Dataset):
    def __init__(self,root,split,input_size=512,training=False):
        self.root=Path(root);self.split=split;self.size=input_size;self.training=training
        self.json_path=self.root/'annotations'/f'panoptic_{split}2017.json'
        self.gt=json.loads(self.json_path.read_text(encoding='utf-8-sig'))
        self.images=self.gt['images'];self.anns={str(a['image_id']):a for a in self.gt['annotations']}
        names={'ship':0,'sea':1,'land':2}
        self.category_to_contiguous={c['id']:names[c['name'].lower()] for c in self.gt['categories']}
        self.contiguous_to_category={v:k for k,v in self.category_to_contiguous.items()}
        if set(self.category_to_contiguous.values())!={0,1,2}:raise ValueError('Exactly ship/sea/land categories required')
        self.mask_dir=self.root/'annotations'/f'panoptic_{split}2017'
        self.text=load_text_rows(root) if training else None
        if training:
            missing={str(x['id']) for x in self.images}-set(self.text)
            if missing:raise ValueError(f'Missing training text for {len(missing)} images')

    def __len__(self):return len(self.images)

    def __getitem__(self,index):
        info=self.images[index];ann=self.anns[str(info['id'])]
        path=self.root/f'{self.split}2017'/info['file_name']
        with Image.open(path) as im:
            im=im.convert('RGB');w,h=im.size
            image=np.asarray(im.resize((self.size,self.size),Image.Resampling.BILINEAR)).copy()
        record={'image':torch.from_numpy(image).permute(2,0,1).float(),'image_id':info['id'],
                'height':h,'width':w,'file_name':info['file_name'],'panoptic_file_name':ann['file_name']}
        if self.training:
            with Image.open(self.mask_dir/ann['file_name']) as im:
                ids=rgb2id(im.convert('RGB').resize((self.size,self.size),Image.Resampling.NEAREST))
            masks=[];labels=[]
            for segment in ann['segments_info']:
                if segment.get('iscrowd',0):continue
                mask=ids==segment['id']
                if mask.any():masks.append(torch.from_numpy(mask));labels.append(self.category_to_contiguous[segment['category_id']])
            record['target']={'labels':torch.tensor(labels,dtype=torch.long),
                              'masks':torch.stack(masks) if masks else torch.zeros(0,self.size,self.size,dtype=torch.bool)}
            record['prompts']=training_prompts(self.text[str(info['id'])])
        return record

def collate(records):return records
