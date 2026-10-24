"""YAML configuration loading."""
import copy
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]

def merge(left,right):
    result=copy.deepcopy(left)
    for k,v in right.items():
        result[k]=merge(result[k],v) if k in result and isinstance(result[k],dict) and isinstance(v,dict) else copy.deepcopy(v)
    return result

def load_config(path):
    path=Path(path).resolve()
    value=yaml.safe_load(path.read_text(encoding='utf-8'))
    base=value.pop('_BASE_',None)
    return merge(load_config(path.parent/base),value) if base else value

def project_path(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p
