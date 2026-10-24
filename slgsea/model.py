"""SLGSea panoptic segmentation model."""
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from .vision.pixel_decoder import MSDeformAttnPixelDecoder
from .vision.position_encoding import PositionEmbeddingSine
from .vision.matcher import HungarianMatcher
from .vision.criterion import SetCriterion
from .decoder import SLGSeaDecoder
from .semantic import LSFG,SQC,SFMA
from .language.cache import TextFeatureBank
from .prompts import fixed_prompts

class SLGSea(nn.Module):
    def __init__(self,config,text_cache,allow_test_cache=False):
        super().__init__();self.config=config
        m=config['model'];d=m['hidden_dim']
        cfg=get_cfg();cfg.MODEL.BACKBONE.NAME='build_resnet_backbone';cfg.MODEL.BACKBONE.FREEZE_AT=0
        cfg.MODEL.RESNETS.DEPTH=50;cfg.MODEL.RESNETS.STRIDE_IN_1X1=False
        cfg.MODEL.RESNETS.OUT_FEATURES=['res2','res3','res4','res5'];cfg.MODEL.RESNETS.NORM='FrozenBN'
        self.backbone=build_backbone(cfg)
        self.sem_seg_head=nn.Module()
        self.sem_seg_head.pixel_decoder=MSDeformAttnPixelDecoder(
            self.backbone.output_shape(),transformer_dropout=0.,transformer_nheads=m['num_heads'],
            transformer_dim_feedforward=1024,transformer_enc_layers=m['pixel_encoder_layers'],
            conv_dim=d,mask_dim=d,norm='GN',transformer_in_features=['res3','res4','res5'],common_stride=4)
        self.sem_seg_head.predictor=SLGSeaDecoder(d,True,num_classes=3,hidden_dim=d,num_queries=m['num_queries'],
            nheads=m['num_heads'],dim_feedforward=m['feedforward_dim'],dec_layers=m['decoder_layers'],
            pre_norm=False,mask_dim=d,enforce_input_project=False)
        self.text_bank=TextFeatureBank(text_cache,allow_test_cache)
        self.text_projection=nn.Linear(self.text_bank.raw_dim,d)
        self.sqc_projection=nn.Linear(self.text_bank.raw_dim,d)
        self.visual_projection=nn.Conv2d(d,d,1)
        self.field_position=PositionEmbeddingSine(d//2,normalize=True)
        self.lsfg=LSFG(d,gamma_st=m['lsfg_gamma_st'],gamma_integrated=m['lsfg_gamma_integrated'],
            gamma_scattering=m['lsfg_gamma_scattering'],initial_temperature=m['lsfg_initial_temperature'])
        self.sqc=SQC(m['num_queries'],d,m['lambda_sca'])
        self.sfma=SFMA(d,m['sfma_temperature'],m['layer_decay'],m['feedback_strength'])
        weights={'loss_ce':2.,'loss_mask':5.,'loss_dice':5.}
        for i in range(m['decoder_layers']):weights.update({f'{k}_{i}':v for k,v in [('loss_ce',2.),('loss_mask',5.),('loss_dice',5.)]})
        matcher=HungarianMatcher(cost_class=2.,cost_mask=5.,cost_dice=5.,num_points=m['point_samples'])
        self.criterion=SetCriterion(3,matcher=matcher,weight_dict=weights,eos_coef=.1,losses=['labels','masks'],
            num_points=m['point_samples'],oversample_ratio=3.,importance_sample_ratio=.75)
        self.register_buffer('pixel_mean',torch.tensor([123.675,116.280,103.530]).view(1,3,1,1),persistent=False)
        self.register_buffer('pixel_std',torch.tensor([58.395,57.120,57.375]).view(1,3,1,1),persistent=False)

    def forward(self,records,return_debug=False):
        device=self.pixel_mean.device
        image=torch.stack([r['image'] for r in records]).to(device)
        features=self.backbone((image-self.pixel_mean)/self.pixel_std)
        mask_features,_,multi=self.sem_seg_head.pixel_decoder.forward_features(features)
        # Evaluation uses fixed task-level prompts.
        prompts=[r['prompts'] for r in records] if self.training else [fixed_prompts() for _ in records]
        with torch.autocast(device_type=device.type,enabled=False):
            raw=self.text_bank.get(prompts)
            banks={k:self.text_projection(v.float()) for k,v in raw.items() if k!='sqc'}
            field=self.lsfg(self.visual_projection(mask_features.float()),self.field_position(mask_features).float(),banks)
            learned=self.sem_seg_head.predictor.query_feat.weight.unsqueeze(0).expand(len(records),-1,-1)
            queries=self.sqc(learned,self.sqc_projection(raw['sqc'].float()))
        outputs=self.sem_seg_head.predictor(multi,mask_features,field,queries,self.sfma)
        if self.training:
            targets=[{k:v.to(device) for k,v in r['target'].items()} for r in records]
            criterion_input={k:v for k,v in outputs.items() if k!='stage_masks'}
            losses=self.criterion(criterion_input,targets)
            losses={k:v*self.criterion.weight_dict[k] for k,v in losses.items() if k in self.criterion.weight_dict}
            with torch.autocast(device_type=device.type,enabled=False):
                visual=[multi[i%3].float() for i in range(len(outputs['stage_masks']))]
                masks=[F.interpolate(v.float(),size=p.shape[-2:],mode='bilinear',align_corners=False) for v,p in zip(outputs['stage_masks'],visual)]
                losses['loss_sfma']=self.config['model']['lambda_sfma']*self.sfma.alignment_loss(masks,visual,field)
            return losses
        if return_debug:
            return {'pred_logits':outputs['pred_logits'],'pred_masks':outputs['pred_masks'],'field':field,'queries':queries}
        results=[]
        for r,cls,mask in zip(records,outputs['pred_logits'],outputs['pred_masks']):
            mask=F.interpolate(mask[None].float(),size=(r['height'],r['width']),mode='bilinear',align_corners=False)[0]
            results.append(self.panoptic_inference(cls.float(),mask))
        return results

    def panoptic_inference(self,classes,masks):
        scores,labels=classes.softmax(-1).max(-1)
        keep=(labels!=3)&(scores>self.config['model']['object_threshold'])
        scores=scores[keep];labels=labels[keep];masks=masks[keep].sigmoid()
        result=torch.zeros(masks.shape[-2:],dtype=torch.int32,device=masks.device)
        info=[];stuff_ids={};current_id=0
        if len(masks)==0:return result,info
        assignment=(scores[:,None,None]*masks).argmax(0)
        for k in range(len(masks)):
            category=int(labels[k]);assigned=assignment==k
            original=masks[k]>=.5
            area=int(assigned.sum());original_area=int(original.sum())
            selected=assigned&original
            if not area or not original_area or not selected.any():continue
            if area/original_area<self.config['model']['overlap_threshold']:continue
            if category!=0 and category in stuff_ids:
                result[selected]=stuff_ids[category];continue
            current_id+=1;result[selected]=current_id
            if category!=0:stuff_ids[category]=current_id
            info.append({'id':current_id,'category_id':category,'isthing':category==0})
        return result,info
