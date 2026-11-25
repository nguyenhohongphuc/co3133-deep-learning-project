import time, random, platform, subprocess
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from config import Config
from dataset import load_splits, make_loader
from model import build_model
from engine import run_epoch
from utils import seed_all, write_json, fingerprint, save_checkpoint

def train(cfg, resume=False):
    seed_all(cfg.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames,labels=load_splits(cfg); n=len(labels)
    out=cfg.output_dir; out.mkdir(parents=True,exist_ok=True)
    fp=fingerprint(cfg.splits()); generator=torch.Generator().manual_seed(cfg.seed)
    train_loader=make_loader(cfg,frames['train'],True,device,generator)
    val_loader=make_loader(cfg,frames['validation'],False,device)
    model=build_model(cfg,n,pretrained=not resume).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=cfg.amp and device.type=='cuda')
    criterion=nn.CrossEntropyLoss(); history=[]; best=-1.; start=0
    settings={k:str(v) if hasattr(v,'__fspath__') else v for k,v in asdict(cfg).items()}
    if resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if ck['fingerprint']!=fp or ck['config']!=settings: raise ValueError('Resume requires identical config and split files')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        start=ck['epoch']+1; best=ck['best']; history=ck['history']; generator.set_state(ck['generator'])
        torch.set_rng_state(ck['torch_rng']); random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        if device.type=='cuda' and ck['cuda_rng'] is not None: torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif (out/'last.pt').exists(): raise FileExistsError('Existing run: use resume=True or change output_dir')
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: commit=None
    write_json(out/'run_config.json',{'config':settings,'fingerprint':fp,'labels':labels,'torch':torch.__version__,'torchvision':torchvision.__version__,'python':platform.python_version(),'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'git_commit':commit,'parameters':sum(p.numel() for p in model.parameters())})
    for epoch in range(start,cfg.epochs):
        lr=optimizer.param_groups[0]['lr']
        tr,_=run_epoch(model,train_loader,device,criterion,n,optimizer,scaler,cfg.amp)
        va,_=run_epoch(model,val_loader,device,criterion,n,amp=cfg.amp)
        scheduler.step()
        history.append({'epoch':epoch+1,'lr':lr,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()}})
        improved=va['macro_f1']>best
        if improved: best=va['macro_f1']
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'history':history,'config':settings,'fingerprint':fp,'labels':labels,'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        if improved: save_checkpoint(out/'best.pt',state)
        save_checkpoint(out/'last.pt',state)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f'Epoch {epoch+1}/{cfg.epochs}: train loss {tr["loss"]:.4f}, val acc {va["accuracy"]:.4f}, val F1 {va["macro_f1"]:.4f}, best {best:.4f}')
    return pd.DataFrame(history)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--resume',action='store_true'); p.add_argument('--workers',type=int,default=4); args=p.parse_args()
    cfg=Config(workers=args.workers); train(cfg,args.resume)
