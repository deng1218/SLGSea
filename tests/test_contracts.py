"""Tests for text caches, batch sampling, and panoptic evaluation."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from slgsea.language.cache import load_cache,validate_cache
from slgsea.language.encoder import tokenize
from slgsea.config import ROOT
from slgsea.prompts import fixed_prompts
from slgsea.engine import UpdateBatchSampler
from slgsea.evaluation import evaluate
from slgsea.panoptic.utils import id2rgb

class Contracts(unittest.TestCase):
    def test_cache_authentic_and_complete(self):
        cache=load_cache(ROOT/'assets/text_cache.pt')
        self.assertFalse(cache['synthetic_test_only'])
        self.assertEqual(tokenize(cache['texts']).shape[0],len(cache['texts']))
        self.assertEqual([len(v) for v in fixed_prompts().values()],[6,3,3,3])
        bad=dict(cache,synthetic_test_only=True)
        with self.assertRaises(ValueError):validate_cache(bad)
        bad=dict(cache,template_sha256='wrong')
        with self.assertRaises(ValueError):validate_cache(bad)

    def test_resume_batch_sequence(self):
        full=list(UpdateBatchSampler(13,8,42,0,10))
        resumed=list(UpdateBatchSampler(13,8,42,4,10))
        self.assertEqual(full[4:],resumed)
        self.assertTrue(all(len(x)==8 for x in full))

    def test_evaluator_perfect_and_category_remap(self):
        # IDs deliberately differ from class indices; verifies the wrapper's remapping.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);gt_dir=root/'gt';gt_dir.mkdir()
            ids=np.zeros((6,6),dtype=np.int32);ids[:2]=1;ids[2:4]=2;ids[4:]=3
            Image.fromarray(id2rgb(ids).astype(np.uint8)).save(gt_dir/'a.png')
            classes=[{'id':5,'name':'ship','isthing':1},{'id':7,'name':'sea','isthing':0},{'id':9,'name':'land','isthing':0}]
            gtsegments=[{'id':i+1,'category_id':c['id'],'area':12,'iscrowd':0} for i,c in enumerate(classes)]
            class Dataset:
                mask_dir=gt_dir;gt={'categories':classes};contiguous_to_category={0:5,1:7,2:9}
                anns={'1':{'image_id':1,'file_name':'a.png','segments_info':gtsegments}}
                def __len__(self):return 1
                def __getitem__(self,i):return {'image_id':1,'panoptic_file_name':'a.png'}
            class Model:
                training=True
                def train(self,mode=True):self.training=mode;return self
                def eval(self):return self.train(False)
                def __call__(self,records):
                    return [(torch.from_numpy(ids),[{'id':i+1,'category_id':i,'isthing':i==0} for i in range(3)])]
            model=Model();report=evaluate(model,Dataset(),root/'pred')
            self.assertTrue(model.training)
            self.assertTrue(report['complete_validation'])
            for value in report['metrics'].values():self.assertAlmostEqual(value,100.)

if __name__=='__main__':unittest.main()
