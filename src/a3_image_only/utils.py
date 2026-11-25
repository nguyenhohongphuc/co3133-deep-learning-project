import random, json, hashlib, os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed); np.random.seed(seed)

def metrics(y,p,n):
    return {'accuracy':float(accuracy_score(y,p)), 'macro_f1':float(f1_score(y,p,labels=list(range(n)),average='macro',zero_division=0))}

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def fingerprint(folder):
    return {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['train.csv','validation.csv','test.csv','labels.json']}

def save_checkpoint(path,state):
    temp=Path(str(path)+'.tmp'); torch.save(state,temp); os.replace(temp,path)
