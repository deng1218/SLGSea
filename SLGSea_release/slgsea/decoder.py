"""Nine-stage visual decoding with scattering initialization and field feedback."""
import torch
from .vision.attention import MultiScaleMaskedTransformerDecoder

class SLGSeaDecoder(MultiScaleMaskedTransformerDecoder):
    def forward(self,x,mask_features,field,scattering_queries,feedback):
        src=[];pos=[];sizes=[]
        for i in range(3):
            sizes.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i]).flatten(2).permute(2,0,1))
            src.append((self.input_proj[i](x[i]).flatten(2)+self.level_embed.weight[i][None,:,None]).permute(2,0,1))
        batch=mask_features.shape[0]
        query_pos=self.query_embed.weight.unsqueeze(1).expand(-1,batch,-1)
        output=scattering_queries.transpose(0,1)
        classes=[];masks=[];stages=[]
        cls,mask,attention=self.forward_prediction_heads(output,mask_features,sizes[0])
        classes.append(cls);masks.append(mask)
        for i in range(self.num_layers):
            # Current mask reads semantic regions before refining the next query state.
            with torch.autocast(device_type=output.device.type,enabled=False):
                output=feedback.feedback(output.transpose(0,1).float(),mask.float(),field.float()).transpose(0,1)
            attention[attention.all(-1)]=False
            level=i%3
            output=self.transformer_cross_attention_layers[i](output,src[level],memory_mask=attention,pos=pos[level],query_pos=query_pos)
            output=self.transformer_self_attention_layers[i](output,query_pos=query_pos)
            output=self.transformer_ffn_layers[i](output)
            cls,mask,attention=self.forward_prediction_heads(output,mask_features,sizes[(i+1)%3])
            classes.append(cls);masks.append(mask);stages.append(mask)
        return {'pred_logits':classes[-1],'pred_masks':masks[-1],
                'aux_outputs':[{'pred_logits':a,'pred_masks':b} for a,b in zip(classes[:-1],masks[:-1])],
                'stage_masks':stages}
