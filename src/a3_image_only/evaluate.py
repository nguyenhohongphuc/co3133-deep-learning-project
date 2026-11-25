import time
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from config import Config
from dataset import load_splits,make_loader
from model import build_model
from engine import run_epoch
from utils import fingerprint,write_json,seed_all

def evaluate(cfg):
    seed_all(cfg.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames,labels=load_splits(cfg); names=[k for k,v in sorted(labels.items(),key=lambda kv:kv[1])]; n=len(names)
    out=cfg.output_dir; ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    if ck['fingerprint']!=fingerprint(cfg.splits()) or ck['labels']!=labels: raise ValueError('Checkpoint/split mismatch')
    for k in ['backbone','dropout','image_size']:
        if ck['config'][k]!=getattr(cfg,k): raise ValueError(f'Checkpoint config mismatch: {k}')
    model=build_model(cfg,n,False).to(device); model.load_state_dict(ck['model'])
    loader=make_loader(cfg,frames['test'],False,device)
    scores,p=run_epoch(model,loader,device,nn.CrossEntropyLoss(),n,amp=cfg.amp)
    scores['checkpoint_epoch']=ck['epoch']+1; scores['parameters']=sum(x.numel() for x in model.parameters()); scores['training_seconds']=sum(r['train_seconds'] for r in ck['history']); scores['through_best_only']=True
    history_path=out/'history.csv'
    if history_path.exists(): scores['training_seconds']=float(pd.read_csv(history_path).train_seconds.sum()); scores['through_best_only']=False
    # Isolated forward latency: fixed batch, excluding disk and preprocessing.
    x=next(iter(loader))[0].to(device); model.eval(); enabled=cfg.amp and device.type=='cuda'
    sync=lambda: torch.cuda.synchronize() if device.type=='cuda' else None
    with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=enabled):
        for _ in range(5): model(x)
        sync(); t=time.perf_counter()
        for _ in range(20): model(x)
        sync(); scores['forward_ms_per_image']=1000*(time.perf_counter()-t)/(20*len(x)); scores['latency_batch_size']=len(x)
    write_json(out/'test_metrics.json',scores)
    report=classification_report(p['true'],p['pred'],labels=list(range(n)),target_names=names,output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(out/'classification_report.csv')
    pred=frames['test'].iloc[p['index']].copy(); pred['prediction']=p['pred']; pred['predicted_type']=[names[i] for i in p['pred']]; pred['confidence']=p['confidence']; pred['correct']=pred.label==pred.prediction
    pred.drop(columns=['local_path']).to_csv(out/'test_predictions.csv',index=False)
    pred.loc[~pred.correct].drop(columns=['local_path']).to_csv(out/'errors.csv',index=False)
    cm=confusion_matrix(p['true'],p['pred'],labels=list(range(n))); pd.DataFrame(cm,index=names,columns=names).to_csv(out/'confusion_matrix.csv')
    fig,ax=plt.subplots(figsize=(13,10)); ConfusionMatrixDisplay(cm,display_labels=names).plot(ax=ax,xticks_rotation=90,values_format='d',colorbar=False); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig)
    if history_path.exists():
        h=pd.read_csv(history_path)
        for metric in ['loss','accuracy','macro_f1']:
            fig,ax=plt.subplots(); ax.plot(h.epoch,h['train_'+metric],label='train'); ax.plot(h.epoch,h['val_'+metric],label='validation'); ax.set(xlabel='Epoch',ylabel=metric); ax.legend(); fig.tight_layout(); fig.savefig(out/f'{metric}_curve.png',dpi=160); plt.close(fig)
    print(scores); return scores,pred

if __name__=='__main__': evaluate(Config())
