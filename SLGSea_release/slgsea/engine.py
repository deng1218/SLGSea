"""Single-device training with reproducible batch indexing and explicit resume."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import SARPanopticDataset,collate
from .evaluation import evaluate
from .checkpoint import save_checkpoint,restore_model,load_visual_initialization,read_checkpoint
from .language.cache import load_cache
from .config import project_path
from .model import SLGSea
from .language.encoder import file_sha256

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def rng_state():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python':random.getstate(),'numpy_name':np.random.get_state()[0],
            'numpy_keys':np.random.get_state()[1].tolist(),'numpy_position':np.random.get_state()[2],
            'numpy_has_gauss':np.random.get_state()[3],'numpy_cached_gaussian':np.random.get_state()[4]}

def restore_rng(value):
    if not value:return
    torch.set_rng_state(value['torch'])
    if value['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(value['cuda'])
    random.setstate(value['python'])
    np.random.set_state((value['numpy_name'],np.array(value['numpy_keys'],dtype=np.uint32),
                        value['numpy_position'],value['numpy_has_gauss'],value['numpy_cached_gaussian']))

class UpdateBatchSampler:
    """Batch k depends only on seed/k/dataset length, independent of worker prefetch."""
    def __init__(self,n,batch,seed,start,total):
        self.n=n;self.batch=batch;self.seed=seed;self.start=start;self.total=total
        if n<1:raise ValueError('Empty training dataset')
    def __len__(self):return self.total-self.start
    def __iter__(self):
        cached={}
        for step in range(self.start,self.total):
            batch=[]
            for position in range(step*self.batch,(step+1)*self.batch):
                epoch,offset=divmod(position,self.n)
                if epoch not in cached:
                    cached={epoch:torch.randperm(self.n,generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()}
                batch.append(cached[epoch][offset])
            yield batch

def run_training(cfg,init_weights=None,resume=None,max_steps=None,skip_eval=False):
    seed_everything(cfg['seed']);device=torch.device(cfg['device']);solver=cfg['solver']
    if solver['schedule']!='constant':raise ValueError('Supported learning-rate schedule: constant')
    out=project_path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if not resume and (out/'training.jsonl').exists():
        raise FileExistsError('Output contains a previous run. Use --resume or a new --output directory.')
    checkpoint=read_checkpoint(resume) if resume else None
    if checkpoint:
        old=checkpoint['config']
        for key in ['model','input_size','dataset','seed']:
            if old[key]!=cfg[key]:raise ValueError(f'Resume config mismatch: {key}')
        for key in ['batch_size','lr','weight_decay','gradient_clip','amp','amp_dtype','schedule']:
            if old['solver'][key]!=solver[key]:raise ValueError(f'Resume solver mismatch: {key}')
    cache=checkpoint['text_cache'] if checkpoint else load_cache(project_path(cfg['text_cache']))
    model=SLGSea(cfg,cache).to(device)
    initialization=None
    if checkpoint:restore_model(model,checkpoint['model'])
    elif init_weights:initialization=load_visual_initialization(model,init_weights)
    else:raise ValueError('Provide --init-weights for COCO initialization or --resume for checkpoint loading')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=solver['lr'],weight_decay=solver['weight_decay'])
    amp=solver['amp'] and device.type=='cuda'
    dtype={'float16':torch.float16,'bfloat16':torch.bfloat16}[solver['amp_dtype']]
    if amp and dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support BF16. Set solver.amp_dtype to float16, or amp to false.')
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    start=0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer']);scaler.load_state_dict(checkpoint['scaler']);start=checkpoint['step']
        restore_rng(checkpoint.get('rng_state'))
    total=solver['iterations'] if max_steps is None else min(solver['iterations'],start+max_steps)
    if total<=start:raise ValueError('No updates remaining')
    dataset=SARPanopticDataset(project_path(cfg['data_root']),'train',cfg['input_size'],True)
    data_state={'annotation_sha256':file_sha256(dataset.json_path),'images':len(dataset)}
    if checkpoint:
        previous=checkpoint.get('data_state') or {}
        for key,value in data_state.items():
            if previous.get(key)!=value:raise ValueError(f'Resume dataset mismatch: {key}')
    # Separate loader RNG avoids perturbing network randomness when resuming.
    loader=DataLoader(dataset,batch_sampler=UpdateBatchSampler(len(dataset),solver['batch_size'],cfg['seed'],start,total),
                      collate_fn=collate,num_workers=solver['workers'],pin_memory=device.type=='cuda',
                      generator=torch.Generator().manual_seed(cfg['seed']))
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    metadata={'initialization':initialization,'text_cache_weight_sha256':cache['weight_sha256'],
              'template_sha256':cache['template_sha256'],'full_training_iterations':solver['iterations'],
              'this_run_stop_step':total,'subset_debug_run':total<solver['iterations'],
              'selection_rule':'final scheduled checkpoint; no validation-based best selection'}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    model.train();log_path=out/'training.jsonl'
    for step,batch in enumerate(loader,start=start+1):
        tic=time.time();retry_rng=rng_state()
        for attempt in range(12):
            if attempt:restore_rng(retry_rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):losses=model(batch)
            loss=sum(losses.values())
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at update {step}')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),solver['gradient_clip'],error_if_nonfinite=False)
            if torch.isfinite(norm):
                scaler.step(optimizer);scaler.update();break
            if not scaler.is_enabled():raise FloatingPointError(f'Non-finite gradients at update {step}')
            # GradScaler skips the update and reduces its scale; retry the same batch.
            # Thus reported steps count actual successful optimizer updates.
            scaler.step(optimizer);scaler.update()
        else:raise FloatingPointError(f'FP16 overflow persisted at update {step}')
        record={'step':step,'loss':float(loss.detach()),'grad_norm_before_clip':float(norm),
                'seconds':time.time()-tic,'amp_overflow_retries':attempt,**{k:float(v.detach()) for k,v in losses.items()}}
        with log_path.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        if step%20==0 or step==start+1 or step==total:print(f'update {step}/{solver["iterations"]}: loss={record["loss"]:.5f}',flush=True)
        if not skip_eval and (step%solver['eval_period']==0 or step==total):
            val=SARPanopticDataset(project_path(cfg['data_root']),'val',cfg['input_size'],False)
            evaluate(model,val,out/'validation'/f'step_{step:07d}')
        if step%solver['checkpoint_period']==0 or step==total:
            state=rng_state()
            save_checkpoint(out/f'model_{step:07d}.pt',model,optimizer,scaler,step,cfg,cache,state,{**data_state,'processed_updates':step})
            if step==solver['iterations']:
                save_checkpoint(out/'model_final.pt',model,optimizer,scaler,step,cfg,cache,state,{**data_state,'processed_updates':step})
    return {'completed_step':total,'paper_schedule_complete':total==solver['iterations'],'output':str(out)}
