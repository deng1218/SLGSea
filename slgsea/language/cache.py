"""Offline frozen text features; trainable projections stay inside the model."""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from ..prompts import SPEC,fixed_prompts

def template_hash():
    return hashlib.sha256(json.dumps(SPEC,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def load_cache(path):
    value=torch.load(path,map_location='cpu',weights_only=True)
    validate_cache(value)
    return value

def validate_cache(value,allow_test=False):
    if value.get('synthetic_test_only',False) and not allow_test:
        raise ValueError('Synthetic smoke-test text cache cannot be used for training or evaluation.')
    if value['template_sha256']!=template_hash():raise ValueError('Templates changed; regenerate the text cache.')
    features=value['features']
    if features.ndim!=2 or len(value['texts'])!=features.shape[0] or not torch.isfinite(features).all():
        raise ValueError('Invalid text-cache features')
    if len(set(value['texts']))!=len(value['texts']):raise ValueError('Duplicate cache strings')
    absent=set(t for bank in fixed_prompts().values() for t in bank)-set(value['texts'])
    if absent:raise ValueError('Fixed inference prompts missing from cache: '+str(absent))

class TextFeatureBank(nn.Module):
    def __init__(self,cache,allow_test=False):
        super().__init__();validate_cache(cache,allow_test)
        self.register_buffer('features',cache['features'].float(),persistent=False)
        self.index={t:i for i,t in enumerate(cache['texts'])}
        self.raw_dim=self.features.shape[1]

    def get(self,prompt_batch):
        result={}
        for key in ['scale_topology','integrated','scattering','sqc']:
            try:ids=[[self.index[t] for t in record[key]] for record in prompt_batch]
            except KeyError as e:raise ValueError(f'Prompt absent from cache; run prepare_text.py: {e}') from e
            result[key]=self.features[torch.tensor(ids,device=self.features.device)]
        return result
