"""Full validation with bundled COCO panoptic metric implementation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .panoptic.evaluation import pq_compute_single_core
from .panoptic.utils import id2rgb

@torch.inference_mode()
def evaluate(model,dataset,output_dir,max_images=None):
    was_training=model.training;model.eval()
    out=Path(output_dir);pred_dir=out/'panoptic';pred_dir.mkdir(parents=True,exist_ok=True)
    predictions=[];pairs=[]
    n=len(dataset) if max_images is None else min(max_images,len(dataset))
    try:
        for i in range(n):
            record=dataset[i]
            ids,segments=model([record])[0]
            ids=ids.cpu().numpy()
            for s in segments:
                s['category_id']=dataset.contiguous_to_category[s['category_id']]
                s['area']=int((ids==s['id']).sum())
            name=record['panoptic_file_name']
            (pred_dir/name).parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(id2rgb(ids).astype(np.uint8)).save(pred_dir/name)
            ann={'image_id':record['image_id'],'file_name':name,'segments_info':segments}
            predictions.append(ann);pairs.append((dataset.anns[str(record['image_id'])],ann))
            if (i+1)%50==0:print(f'Validated {i+1}/{n}',flush=True)
        (out/'predictions.json').write_text(json.dumps({'annotations':predictions,'categories':dataset.gt['categories']}),encoding='utf-8')
        categories={c['id']:c for c in dataset.gt['categories']}
        stat=pq_compute_single_core(0,pairs,str(dataset.mask_dir),str(pred_dir),categories)
        metrics={};valid_categories={}
        for suffix,isthing in [('',None),('_th',True),('_st',False)]:
            result,_=stat.pq_average(categories,isthing=isthing)
            valid_categories[suffix or 'all']=result['n']
            for key in ['pq','sq','rq']:metrics[key.upper()+suffix]=100*result[key]
        report={'metrics':metrics,'valid_category_counts':valid_categories,'images':n,'complete_validation':n==len(dataset),
                'inference_language':'fixed task-level cached CLIP embeddings; no image-aligned validation text',
                'checkpoint_selection':'caller controlled; no best-score selection here'}
        (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return report
    finally:model.train(was_training)
