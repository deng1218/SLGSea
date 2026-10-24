"""Forward, backward, and inference checks using synthetic or real input data."""
import argparse,importlib.abc,json,sys,time
from pathlib import Path

class RepositoryImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'mask2former','clip','open_clip','panopticapi','MultiScaleDeformableAttention'}:
            raise ImportError('Unexpected module import: '+fullname)
        return None
sys.meta_path.insert(0,RepositoryImportGuard())

import torch
from slgsea.config import load_config,ROOT
from slgsea.language.cache import load_cache,template_hash
from slgsea.prompts import fixed_prompts
from slgsea.model import SLGSea
from slgsea.data import SARPanopticDataset
from slgsea.checkpoint import load_visual_initialization

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--size',type=int,default=64)
    p.add_argument('--data-root',type=Path);p.add_argument('--text-cache',type=Path);p.add_argument('--init-weights',type=Path)
    p.add_argument('--report',type=Path,default=Path('outputs/smoke_report.json'));a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0)
    cfg=load_config(ROOT/'configs/sarssp.yaml');cfg['input_size']=a.size;cfg['device']=a.device
    # Full 9-layer/100-query architecture retained; fewer sampled loss points only for smoke speed.
    cfg['model']['point_samples']=256
    if bool(a.data_root)!=bool(a.text_cache):p.error('Provide both --data-root and --text-cache for real-data smoke testing')
    if a.data_root:
        cache=load_cache(a.text_cache);data=SARPanopticDataset(a.data_root,'train',a.size,True);record=data[0]
    else:
        texts=sorted(set(t for bank in fixed_prompts().values() for t in bank))
        cache={'texts':texts,'features':torch.randn(len(texts),1024),'template_sha256':template_hash(),'synthetic_test_only':True}
        masks=torch.zeros(3,a.size,a.size,dtype=torch.bool);masks[0,a.size//4:a.size//2,a.size//4:a.size//2]=True
        masks[2,:a.size//5]=True;masks[1]=~(masks[0]|masks[2])
        record={'image':torch.rand(3,a.size,a.size)*255,'height':a.size,'width':a.size,'prompts':fixed_prompts(),
                'target':{'labels':torch.arange(3),'masks':masks}}
    model=SLGSea(cfg,cache,allow_test_cache=not a.data_root).to(a.device)
    initialization=load_visual_initialization(model,a.init_weights) if a.init_weights else None
    tic=time.time();model.train();losses=model([record]);loss=sum(losses.values())
    assert torch.isfinite(loss),losses
    loss.backward()
    checked={}
    for name in ['text_projection.weight','lsfg.output.weight','sqc.mixing','sfma.modulation.2.weight']:
        param=dict(model.named_parameters())[name]
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
        checked[name]=float(param.grad.abs().sum())
        assert checked[name]>0,name
    torch.nn.utils.clip_grad_norm_(model.parameters(),.01,error_if_nonfinite=True)
    torch.optim.SGD(model.parameters(),lr=1e-4).step()
    model.eval()
    with torch.inference_mode():
        only_image={k:record[k] for k in ['image','height','width']}
        debug=model([only_image],return_debug=True)
        poisoned={**only_image,'prompts':None,'target':'must not be read'}
        again=model([poisoned],return_debug=True)
        assert torch.equal(debug['pred_logits'],again['pred_logits'])
        originals=model.text_bank.features.clone();model.text_bank.features.zero_()
        changed=model([only_image],return_debug=True)
        assert not torch.allclose(debug['field'],changed['field'])
        model.text_bank.features.copy_(originals)
        panoptic,segments=model([only_image])[0]
        assert tuple(panoptic.shape)==(record['height'],record['width'])
        assert set(int(v) for v in panoptic.unique())<=({0}|{s['id'] for s in segments})
    forbidden=[x for x in sys.modules if x.split('.')[0] in {'mask2former','clip','open_clip','panopticapi','MultiScaleDeformableAttention'}]
    assert not forbidden,forbidden
    report={'status':'passed','device':a.device,'input_size':a.size,'source':'one real training image' if a.data_root else 'synthetic',
            'decoder_layers':cfg['model']['decoder_layers'],'queries':cfg['model']['num_queries'],
            'initialization':initialization,'loss':float(loss.detach()),'finite_nonzero_gradients':checked,
            'inference_ignores_image_specific_prompts_and_gt':True,'inference_uses_fixed_language_field':True,
            'seconds':time.time()-tic,'smoke_loss_point_samples':256}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
