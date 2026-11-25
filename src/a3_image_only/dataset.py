import json
from pathlib import Path
from itertools import combinations
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch
from utils import seed_worker

class Letterbox:
    def __init__(self, size): self.size = size
    def __call__(self, im):
        im = ImageOps.contain(im, (self.size,self.size), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB',(self.size,self.size),'white')
        canvas.paste(im, ((self.size-im.width)//2,(self.size-im.height)//2))
        return canvas

def resolve_image(root, value):
    value = str(value).replace('\\','/')
    marker = 'images/small/'
    if marker not in value: raise ValueError(f'Unexpected path: {value}')
    relative = Path(value.split(marker,1)[1])
    if relative.is_absolute() or '..' in relative.parts: raise ValueError(value)
    return Path(root)/relative

def load_splits(cfg, check_files=True):
    folder = cfg.splits()
    labels = json.loads((folder/'labels.json').read_text(encoding='utf-8'))
    if sorted(labels.values()) != list(range(len(labels))) or len(labels)!=12:
        raise ValueError('Expected official 12-class mapping')
    frames = {}
    for name in ('train','validation','test'):
        df = pd.read_csv(folder/f'{name}.csv', dtype={'item_id':str,'image_id':str,'group_id':str,'phash':str})
        required = ['item_id','image_id','image_path','group_id','product_type','label']
        if df[required].isna().any().any(): raise ValueError(f'Missing fields in {name}')
        if not (df.product_type.map(labels)==df.label).all(): raise ValueError(f'Label mismatch: {name}')
        if set(df.label)!=set(labels.values()): raise ValueError(f'Incomplete classes: {name}')
        df['local_path'] = [str(resolve_image(cfg.image_root,p)) for p in df.image_path]
        if check_files:
            missing = [p for p in set(df.local_path) if not Path(p).is_file()]
            if missing: raise FileNotFoundError(f'{name}: {len(missing)} missing files, e.g. {missing[:3]}')
        frames[name] = df
    for a,b in combinations(frames,2):
        for col in ['item_id','image_id','image_path','group_id','image_sha256','pixel_sha256','phash','title']:
            if col in frames[a] and col in frames[b]:
                overlap = set(frames[a][col].dropna()) & set(frames[b][col].dropna())
                if overlap: raise ValueError(f'Leakage {a}/{b} {col}: {len(overlap)}')
    return frames,labels

def transform(cfg, train=False):
    ops = [Letterbox(cfg.image_size)]
    if train: ops += [T.RandomHorizontalFlip(p=.5), T.RandomAffine(5,translate=(.03,.03),scale=(.95,1.05),fill=255)]
    return T.Compose(ops+[T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])

class ABODataset(Dataset):
    def __init__(self, frame, tf): self.frame=frame.reset_index(drop=True); self.tf=tf
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        with Image.open(row.local_path) as im:
            x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
        return x, int(row.label), i

def make_loader(cfg, df, train=False, device=None, generator=None):
    return DataLoader(ABODataset(df,transform(cfg,train)),batch_size=cfg.batch_size,shuffle=train,num_workers=cfg.workers,pin_memory=(device is not None and device.type=='cuda'),worker_init_fn=seed_worker,generator=generator)
