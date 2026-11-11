"""Grad-CAM diagnostics for fixed A2 test examples; never updates model weights.

    python scripts/a2_gradcam.py

Uses the full-fine-tuning run selected by validation before test. Selects at most
twelve diagnostic images: Banh pia correct/wrong, H1/H2 confusions and crop errors.
Attribution is explanatory evidence, not proof of background dependence or label error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'dla261-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from src.a2.data import VNFoods, build_transforms
from src.a2.models import ResNet50


def gradcam(model, tensor, targets):
    captured = {}
    def capture(module, inputs, output):
        captured['activation'] = output
        output.retain_grad()
    hook = model.net.layer4[-1].register_forward_hook(capture)
    maps = {}
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        activation = captured['activation']
        for index, label in enumerate(targets):
            model.zero_grad(set_to_none=True)
            activation.grad = None
            logits[0, label].backward(retain_graph=index < len(targets)-1)
            weights = activation.grad.mean(dim=(2,3), keepdim=True)
            coarse = torch.relu((weights*activation).sum(dim=1,keepdim=True))
            enlarged = F.interpolate(coarse, size=tensor.shape[-2:], mode='bilinear', align_corners=False)[0,0]
            maximum = float(enlarged.max().detach())
            normalized = enlarged/maximum if maximum>0 else torch.zeros_like(enlarged)
            maps[label] = dict(map=normalized.detach().cpu().numpy(), max_before_normalization=maximum,
                               probability=float(logits.softmax(1)[0,label].detach()))
        return maps
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=12)
    parser.add_argument('--force',action='store_true',help='Recompute existing Grad-CAM images')
    args=parser.parse_args()
    torch.set_num_threads(5)
    directory=ROOT/'results/a2/error_analysis';directory.mkdir(parents=True,exist_ok=True)
    figures=ROOT/'results/a2/figures/test';figures.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'results/a2/tables/a2_test_protocol.json').read_text())
    run_id=protocol['representatives']['resnet50_full']['run_id']
    run=ROOT/'results/a2/runs'/run_id
    with (run/'evaluation/test/predictions.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    selected=[];paths=set()
    def add(candidates,reason,count):
        chosen=0
        for row in candidates:
            if row['path'] not in paths and len(selected)<args.limit:
                selected.append((row,reason));paths.add(row['path']);chosen+=1
                if chosen==count:break
    ordered=lambda candidates:sorted(candidates,key=lambda r:(-float(r['confidence']),r['path']))
    add(ordered([r for r in rows if r['true_class']=='Banh pia' and r['correct']=='0']),'Banh pia failure',2)
    add(ordered([r for r in rows if r['true_class']=='Banh pia' and r['correct']=='1']),'Banh pia correct',2)
    for names,reason in [({'Banh chung','Banh tet'},'H1 shape-pair confusion'),
                         ({'Bun bo Hue','Bun rieu','Banh canh'},'H2 soup-group confusion')]:
        add(ordered([r for r in rows if r['true_class'] in names and r['pred_class'] in names and r['correct']=='0']),reason,2)
    add(ordered([r for r in rows if (float(r['aspect_ratio'])>2 or float(r['aspect_ratio'])<.5) and r['correct']=='0']),
        'Extreme-aspect failure',2)
    add(ordered([r for r in rows if r['correct']=='0']),'High-confidence failure',args.limit)
    cfg=yaml.safe_load((run/'config.yaml').read_text())
    data_root=Path(cfg['data_dir']);data_root=data_root if data_root.is_absolute() else ROOT/data_root
    model_cfg=cfg['models'][cfg['model']]
    dataset=VNFoods(data_root,[dict(path=r['path'],label=int(r['true_label'])) for r,_ in selected],
                    build_transforms(cfg,model_cfg['normalize'],train=False))
    params=dict(model_cfg.get('params',{}));params['pretrained']=False
    model=ResNet50(**params)
    model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True),strict=True)
    model.eval()
    before={name:buffer.clone() for name,buffer in model.named_buffers()}
    checkpoint_hash=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
    metadata=[]
    for index,(row,reason) in enumerate(selected):
        image,_=dataset[index]
        stats=cfg['data']['stats'][model_cfg['normalize']]
        rgb=(image*torch.tensor(stats['std'])[:,None,None]+torch.tensor(stats['mean'])[:,None,None]).clamp(0,1)
        rgb=rgb.permute(1,2,0).numpy()
        predicted,truth=int(row['pred_label']),int(row['true_label'])
        targets=list(dict.fromkeys([predicted,truth]))
        filename=f"a2_test_gradcam_{int(row['index']):04d}.png"
        cache=directory/f"gradcam_{int(row['index']):04d}.json"
        if cache.exists() and (figures/filename).exists() and not args.force:
            entry=json.loads(cache.read_text())
            if entry['checkpoint_sha256']!=checkpoint_hash or entry['path']!=row['path']:
                raise ValueError('Grad-CAM cache provenance mismatch')
            metadata.append(entry);continue
        maps=gradcam(model,image.unsqueeze(0),targets)
        fig,axes=plt.subplots(1,1+len(targets),figsize=(4*(1+len(targets)),4.5),squeeze=False)
        axes=axes[0];axes[0].imshow(rgb);axes[0].set_title('Actual model input')
        for ax,label in zip(axes[1:],targets):
            cam=maps[label]['map'];overlay=plt.get_cmap('magma')(cam);overlay[:,:,3]=.55*cam
            ax.imshow(rgb);ax.imshow(overlay)
            name=row['pred_class'] if label==predicted else row['true_class']
            ax.set_title(('Predicted' if label==predicted else 'True')+f': {name}\nGrad-CAM target score {maps[label]["probability"]:.1%}')
        for ax in axes:ax.axis('off')
        fig.suptitle(f"#{row['index']} | {reason} | true: {row['true_class']}\n"
                     'Final residual block; brighter regions = larger positive attribution',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.88))
        fig.savefig(figures/filename,dpi=160,bbox_inches='tight',facecolor='white');plt.close(fig)
        entry=dict(run_id=run_id,path=row['path'],index=int(row['index']),reason=reason,
                   true_class=row['true_class'],predicted_class=row['pred_class'],
                   csv_confidence=float(row['confidence']),figure=filename,checkpoint_sha256=checkpoint_hash,
                   backend='CPU; explanation-only forward/backward; eval mode; no optimizer',
                   torch_version=torch.__version__,
                   targets=[dict(label=label,probability=maps[label]['probability'],
                                 max_before_normalization=maps[label]['max_before_normalization']) for label in targets])
        cache.write_text(json.dumps(entry,indent=2)+'\n',encoding='utf-8');metadata.append(entry)
        print('GRADCAM',index+1,'/',len(selected),row['path'],flush=True)
    assert all(torch.equal(before[name],buffer) for name,buffer in model.named_buffers()), 'BatchNorm buffers changed'
    assert hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()==checkpoint_hash
    (directory/'gradcam_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('GRADCAM_COMPLETE',len(metadata),'images; checkpoint and BN buffers unchanged',flush=True)


if __name__=='__main__':
    main()
