import time
import torch
from tqdm.auto import tqdm
from utils import metrics

def run_epoch(model, loader, device, criterion, n_classes, optimizer=None, scaler=None, amp=True):
    training=optimizer is not None
    model.train(training)
    enabled=amp and device.type=='cuda'
    ys=[]; ps=[]; ids=[]; conf=[]; loss_sum=0.; count=0
    if device.type=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.set_grad_enabled(training):
        for x,y,i in tqdm(loader,desc='Train' if training else 'Evaluate',leave=False):
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=enabled):
                logits=model(x); loss=criterion(logits,y)
            if training:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            prob,p=logits.detach().float().softmax(1).max(1)
            loss_sum+=loss.item()*len(y); count+=len(y)
            ys.extend(y.cpu().tolist()); ps.extend(p.cpu().tolist()); ids.extend(i.tolist()); conf.extend(prob.cpu().tolist())
    if device.type=='cuda': torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    return {'loss':loss_sum/count,**metrics(ys,ps,n_classes),'seconds':seconds,'ms_per_sample':1000*seconds/count}, {'index':ids,'true':ys,'pred':ps,'confidence':conf}
