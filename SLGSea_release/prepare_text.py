"""Prepare authentic frozen CLIP features once; image-specific text is train-only."""
import argparse,json
from pathlib import Path
import torch
from slgsea.data import load_text_rows
from slgsea.prompts import fixed_prompts,training_prompts
from slgsea.language.cache import template_hash
from slgsea.language.encoder import file_sha256,load_encoder,tokenize

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--clip-weights',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path,nargs='+',help='One or more dataset roots with text_library/*_train.json')
    p.add_argument('--output',type=Path,default=Path('assets/text_cache.pt'))
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args();torch.set_num_threads(min(torch.get_num_threads(),8))
    strings=set(t for bank in fixed_prompts().values() for t in bank)
    datasets=[]
    for root in args.data_root:
        rows=load_text_rows(root)
        for row in rows.values():strings.update(t for bank in training_prompts(row).values() for t in bank)
        textfile=next((root/'text_library').glob('*_train.json'))
        datasets.append({'name':root.name,'training_records':len(rows),'text_library_sha256':file_sha256(textfile)})
    texts=sorted(strings);encoder=load_encoder(args.clip_weights,args.device)
    tokens=tokenize(texts,encoder.context_length);embeddings=[]
    with torch.inference_mode():
        for i in range(0,len(texts),args.batch_size):embeddings.append(encoder(tokens[i:i+args.batch_size].to(args.device)).float().cpu())
    cache={'format':'slgsea-text-cache-v1','template_sha256':template_hash(),'texts':texts,'features':torch.cat(embeddings),
           'encoder_family':'OpenAI CLIP text transformer (QuickGELU)','weight_sha256':file_sha256(args.clip_weights),
           'context_length':encoder.context_length,'datasets':datasets,'synthetic_test_only':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(cache,args.output)
    summary={k:v for k,v in cache.items() if k not in ['features','texts']}
    summary.update(unique_texts=len(texts),feature_shape=list(cache['features'].shape))
    args.output.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
