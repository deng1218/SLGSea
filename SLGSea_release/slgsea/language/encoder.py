"""Frozen CLIP text encoder."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from .transformer import Transformer,LayerNorm
from .tokenizer import SimpleTokenizer

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

class FrozenTextEncoder(nn.Module):
    def __init__(self,state):
        super().__init__()
        width=state['ln_final.weight'].shape[0]
        layers=len({k.split('.')[2] for k in state if k.startswith('transformer.resblocks.')})
        self.context_length=state['positional_embedding'].shape[0]
        self.token_embedding=nn.Embedding(state['token_embedding.weight'].shape[0],width)
        self.positional_embedding=nn.Parameter(torch.empty(self.context_length,width))
        mask=torch.full((self.context_length,self.context_length),float('-inf')).triu_(1)
        self.transformer=Transformer(width,layers,width//64,attn_mask=mask)
        self.ln_final=LayerNorm(width)
        self.text_projection=nn.Parameter(torch.empty_like(state['text_projection'],dtype=torch.float32))
        required=self.state_dict()
        missing=set(required)-set(state)
        if missing:raise ValueError('Missing CLIP text tensors: '+str(sorted(missing)))
        self.load_state_dict({k:state[k].float() for k in required},strict=True)
        self.requires_grad_(False);self.eval()

    def forward(self,tokens):
        x=self.token_embedding(tokens)+self.positional_embedding
        x=self.transformer(x.permute(1,0,2)).permute(1,0,2)
        x=self.ln_final(x)
        return x[torch.arange(x.shape[0],device=x.device),tokens.argmax(-1)]@self.text_projection

def load_encoder(path,device='cpu'):
    # Only use a trusted pretrained checkpoint. Both official TorchScript and state dicts work.
    try:
        state=torch.jit.load(str(path),map_location='cpu').state_dict()
    except (RuntimeError,ValueError):
        loaded=torch.load(path,map_location='cpu',weights_only=True)
        state=loaded.get('state_dict',loaded)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    return FrozenTextEncoder(state).to(device)

def tokenize(texts,context_length=77):
    tok=SimpleTokenizer()
    result=torch.zeros(len(texts),context_length,dtype=torch.long)
    for i,text in enumerate(texts):
        ids=[tok.encoder['<|startoftext|>']]+tok.encode(text)+[tok.encoder['<|endoftext|>']]
        if len(ids)>context_length:raise ValueError(f'Text exceeds CLIP context ({len(ids)}): {text}')
        result[i,:len(ids)]=torch.tensor(ids)
    return result
