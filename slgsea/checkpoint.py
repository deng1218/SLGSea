"""Portable checkpoints and explicit, audited COCO visual initialization."""
import pickle
from pathlib import Path
import numpy as np
import torch

def load_visual_initialization(model,path):
    path=Path(path)
    # The public COCO .pkl is a Python pickle: only load trusted pretrained files.
    if path.suffix=='.pkl':
        with path.open('rb') as f:payload=pickle.load(f,encoding='latin1')
    else:payload=torch.load(path,map_location='cpu',weights_only=True)
    source=payload.get('model',payload.get('state_dict',payload))
    target=model.state_dict();selected={};skipped=[];renamed={}
    for k,v in source.items():
        k=k.removeprefix('module.')
        if k=='sem_seg_head.predictor.static_query.weight':
            renamed[k]='sem_seg_head.predictor.query_feat.weight'
            k=renamed[k]
        if k in target and tuple(v.shape)==tuple(target[k].shape):
            selected[k]=torch.from_numpy(v) if isinstance(v,np.ndarray) else v
        else:skipped.append(k)
    expected=[k for k in target if k.startswith(('backbone.','sem_seg_head.')) and '.class_embed.' not in k]
    missing=[k for k in expected if k not in selected]
    if missing:raise ValueError('Incomplete COCO visual initialization; missing/incompatible tensors: '+str(missing[:20]))
    # Assign directly to avoid historic checkpoint-version migration hooks.
    with torch.no_grad():
        for k,v in selected.items():target[k].copy_(v)
    return {'loaded_tensors':len(selected),'skipped_source_keys':skipped,'visual_missing':missing,'legacy_key_migration':renamed}

def save_checkpoint(path,model,optimizer,scaler,step,config,text_cache,rng_state=None,data_state=None):
    value={'format':'slgsea-standalone-v1','model':model.state_dict(),'optimizer':optimizer.state_dict(),
           'scaler':scaler.state_dict(),'step':step,'config':config,'text_cache':text_cache,
           'rng_state':rng_state,'data_state':data_state}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);temporary.replace(path)

def read_checkpoint(path):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value.get('format')!='slgsea-standalone-v1':
        raise ValueError('Invalid SLGSea checkpoint format.')
    return value

def restore_model(model,state):
    target=model.state_dict()
    if set(target)!=set(state):raise ValueError('Checkpoint keys do not match the model')
    with torch.no_grad():
        for k,v in state.items():
            if target[k].shape!=v.shape:raise ValueError('Checkpoint shape mismatch: '+k)
            target[k].copy_(v)
