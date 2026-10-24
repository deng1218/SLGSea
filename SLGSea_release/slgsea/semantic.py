"""Spatial semantic field generation, scattering-aware queries, and mask alignment."""
import torch
from torch import nn
from torch.nn import functional as F


def simplex_projection(x):
    """Euclidean row projection onto {a >= 0, sum(a)=1}; Eq.(5)."""
    u=x.sort(dim=-1,descending=True).values
    cssv=u.cumsum(-1)-1
    k=torch.arange(1,x.shape[-1]+1,device=x.device,dtype=x.dtype)
    valid=(u-cssv/k)>0
    rho=valid.sum(-1,keepdim=True).clamp_min(1)
    theta=cssv.gather(-1,rho-1)/rho.to(x.dtype)
    return (x-theta).clamp_min(0)


class LSFG(nn.Module):
    """6 scale/topology, 3 integrated, 3 scattering tokens; output B,256,H,W."""
    def __init__(self, dim=256, *, gamma_st, gamma_integrated, gamma_scattering, initial_temperature):
        super().__init__()
        if initial_temperature<=0: raise ValueError('temperature must be positive')
        self.log_temperature=nn.Parameter(torch.tensor(float(initial_temperature)).log())
        self.gammas=(gamma_st,gamma_integrated,gamma_scattering)
        self.norms=nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.output=nn.Conv2d(dim,dim,1)

    def project_semantics(self,x,text):
        if text.ndim==2: text=text.unsqueeze(0).expand(x.shape[0],-1,-1)
        logits=F.normalize(x,dim=-1)@F.normalize(text,dim=-1).transpose(-1,-2)
        return (logits/self.log_temperature.exp()).softmax(-1)@text

    def forward(self,projected_visual,position_encoding,text_banks):
        if projected_visual.shape!=position_encoding.shape:
            raise ValueError('visual and positional features must have identical B,D,H,W shapes')
        b,d,h,w=projected_visual.shape
        x=(projected_visual+position_encoding).flatten(2).transpose(1,2)
        for name,norm,gamma,n in zip(['scale_topology','integrated','scattering'],self.norms,self.gammas,[6,3,3]):
            tokens=text_banks[name]
            if tokens.shape[-2:]!=(n,d):raise ValueError(f'{name} must end in ({n},{d})')
            x=norm(x+gamma*self.project_semantics(x,tokens))
        return self.output(x.transpose(1,2).reshape(b,d,h,w))


class SQC(nn.Module):
    def __init__(self,queries=100,dim=256,strength=0.5):
        super().__init__();self.strength=strength
        self.mixing=nn.Parameter(torch.zeros(queries,3))
        self.norm=nn.LayerNorm(dim)

    def forward(self,learnable_queries,projected_reference_bank):
        # Query input Q,D or B,Q,D; projected bank 3,D or B,3,D.
        weights=simplex_projection(self.mixing)
        prior=weights@projected_reference_bank
        return self.norm(learnable_queries+self.strength*prior)


def mask_readout(mask_logits,features):
    """Eqs.(7),(9): one normalized spatial weight per query, including unmatched."""
    if features.shape[-2:]!=mask_logits.shape[-2:]:
        # Match the feature resolution to the predicted masks.
        features=F.interpolate(features,size=mask_logits.shape[-2:],mode='bilinear',align_corners=False)
    weights=mask_logits.sigmoid().flatten(2)
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
    return weights@features.flatten(2).transpose(1,2)


class SFMA(nn.Module):
    def __init__(self,dim=256,temperature=0.07,layer_decay=0.7,feedback_strength=0.03):
        super().__init__();self.temperature=temperature;self.layer_decay=layer_decay;self.beta=feedback_strength
        self.visual_proj=nn.Linear(dim,dim);self.semantic_proj=nn.Linear(dim,dim)
        self.modulation=nn.Sequential(nn.Linear(3*dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.norm=nn.LayerNorm(dim)

    def feedback(self,queries,mask_logits,semantic_field):
        g=mask_readout(mask_logits,semantic_field)
        return self.norm(queries+self.beta*self.modulation(torch.cat([queries,g,queries*g],dim=-1)))

    def alignment_loss(self,mask_logits_by_layer,visual_by_layer,semantic_field):
        if not mask_logits_by_layer or len(mask_logits_by_layer)!=len(visual_by_layer):
            raise ValueError('supply equally many mask and visual layers')
        losses=[];weights=[];L=len(mask_logits_by_layer)
        for i,(m,p) in enumerate(zip(mask_logits_by_layer,visual_by_layer)):
            v=F.normalize(self.visual_proj(mask_readout(m,p)),dim=-1)
            g=F.normalize(self.semantic_proj(mask_readout(m,semantic_field)),dim=-1)
            scores=(v@g.transpose(-1,-2))/self.temperature  # B,Q,Q; never cross-image pairs
            b,q,_=scores.shape
            labels=torch.arange(q,device=scores.device).expand(b,q)
            losses.append(F.cross_entropy(scores.reshape(b*q,q),labels.reshape(-1)))
            weights.append(self.layer_decay**(L-1-i))
        return sum(w*l for w,l in zip(weights,losses))/sum(weights)

