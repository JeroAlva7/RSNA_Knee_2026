"""EXP-011 | RSNA Knee 2026, original independent DINOv2-S fine-tuning.

Fine-tunes last two transformer blocks on report-derived weak labels.
Gold 58 are evaluation-only; never used for fitting or epoch selection.
Offline GPU Kaggle kernel: writes /kaggle/working/submission.csv + diagnostics.
No score guarantee. Time guard fails explicitly, never substitutes priors.
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

UID = 'StudyInstanceUID'
SID = 'SeriesInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
           'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
           'Synovitis', "Baker's", 'Contusion', 'Fracture']
PLANES = ['sagittal', 'coronal', 'axial']
SEED = 20261019
N_VIEWS = 12  # 2 series x 2 middle slices x 3 planes
IMAGE_SIZE = 224
EPOCHS = 4
ACCUMULATION = 8
CACHE_DIR = Path('/kaggle/temp/exp011')
OUT_DIR = Path('/kaggle/working')
MAX_TRAIN_EVAL_SECONDS = 7.25 * 3600   # reserve hidden inference
MAX_TOTAL_SECONDS = 8.60 * 3600
GOLD_SUBMIT_GATE = 0.82


def paths():
    candidates = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                  Path('/kaggle/input/rsna-knee-abnormality-detection')]
    root = next((p for p in candidates if (p / 'train.csv').is_file()), None)
    if root is None:
        raise FileNotFoundError('Missing RSNA competition dataset input')
    checkpoints = [Path('/kaggle/input/models/metaresearch/dinov2/pytorch/small/1'),
                   Path('/kaggle/input/dinov2/pytorch/small/1')]
    checkpoint = next((p for p in checkpoints if (p / 'config.json').is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError('Missing DINOv2 PyTorch Small model input')
    weak = Path('/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv')
    if not weak.is_file():
        raise FileNotFoundError(f'Missing report label dataset: {weak}')
    return root, checkpoint, weak


def normalized_plane(x):
    x = str(x).lower()
    return next((p for p in PLANES if p in x), 'other')


def boolean(x):
    return str(x).strip().lower() in {'true', '1', 'yes', 'y', 't'}


def select_series(df):
    data = df.copy()
    data['_plane'] = data['Anatomical_Plane'].map(normalized_plane)
    data['_fluid'] = data['Fluid_Sensitive'].map(boolean)
    data['_fat'] = data['Fat_Suppression'].map(boolean)
    data['_score'] = data['_fluid'].astype(int) * 2 + data['_fat'].astype(int)
    selected = {}
    for uid, group in data.groupby(UID, sort=False):
        byplane = {}
        for plane in PLANES:
            rows = group[group['_plane'] == plane].sort_values('_score', ascending=False, kind='stable')
            chosen, protocols = [], set()
            for _, row in rows.iterrows():
                protocol = (bool(row['_fluid']), bool(row['_fat']))
                series = str(row[SID])
                if protocol not in protocols:
                    chosen.append(series)
                    protocols.add(protocol)
                if len(chosen) == 2:
                    break
            for series in rows[SID].astype(str):
                if len(chosen) == 2:
                    break
                if series not in chosen:
                    chosen.append(series)
            byplane[plane] = chosen
        selected[str(uid)] = byplane
    return selected


def sort_dicoms(directory):
    records = []
    for order, path in enumerate(directory.glob('*.dcm')):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True,
                                 specific_tags=['ImagePositionPatient','ImageOrientationPatient','InstanceNumber'])
            ipp = getattr(ds, 'ImagePositionPatient', None)
            iop = getattr(ds, 'ImageOrientationPatient', None)
            if ipp is not None and iop is not None and len(iop) >= 6:
                normal = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:6], float))
                z = float(np.dot(normal, np.asarray(ipp[:3], float)))
            else:
                z = float(getattr(ds, 'InstanceNumber', order))
        except Exception:
            z = float(order)
        records.append((z, str(path)))
    records.sort(key=lambda x: x[0])
    return [Path(f) for _, f in records]


def decode_slice(path):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    arr = arr * float(getattr(ds,'RescaleSlope',1.0)) + float(getattr(ds,'RescaleIntercept',0.0))
    if str(getattr(ds,'PhotometricInterpretation','')).upper() == 'MONOCHROME1':
        arr = arr.max() - arr
    finite = np.isfinite(arr)
    if not finite.any():
        arr = np.zeros_like(arr)
    else:
        lo, hi = np.percentile(arr[finite], [1, 99])
        arr = np.clip((arr-lo)/max(float(hi-lo), 1e-6), 0, 1)
    spacing = getattr(ds, 'PixelSpacing', None)
    return (255*arr).astype(np.uint8), spacing


def crop_140mm(rgb, spacing):
    h,w = rgb.shape[:2]
    try:
        sy, sx = float(spacing[0]), float(spacing[1])
        hside, wside = min(h,int(round(140/sy))), min(w,int(round(140/sx)))
        hside = max(64, min(h, hside))
        wside = max(64, min(w, wside))
    except Exception:
        hside = int(h*0.9)
        wside = int(w*0.9)
    cy,cx = h//2,w//2
    return rgb[max(0,cy-hside//2):max(0,cy-hside//2)+hside,
               max(0,cx-wside//2):max(0,cx-wside//2)+wside]


def rgb_triplet(files, center):
    idx = [max(0,center-1),center,min(len(files)-1,center+1)]
    decoded = [decode_slice(files[i]) for i in idx]
    h = min(d[0].shape[0] for d in decoded)
    w = min(d[0].shape[1] for d in decoded)
    rgb = np.stack([d[0][:h,:w] for d in decoded], axis=2)
    rgb = crop_140mm(rgb, decoded[1][1])
    img = Image.fromarray(rgb).resize((IMAGE_SIZE,IMAGE_SIZE),resample=Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8).transpose(2,0,1)


def study_images(uid, selected, dicom_root):
    out = np.zeros((N_VIEWS,3,IMAGE_SIZE,IMAGE_SIZE),dtype=np.uint8)
    mask = np.zeros(N_VIEWS,dtype=bool)
    planes = selected.get(str(uid),{})
    for p,plane in enumerate(PLANES):
        for si, series_uid in enumerate(planes.get(plane,[])[:2]):
            directory = dicom_root / str(uid) / str(series_uid)
            if not directory.is_dir():
                continue
            files = sort_dicoms(directory)
            if len(files) < 2:
                continue
            centers = np.rint(np.array([0.36,0.64])*(len(files)-1)).astype(int)
            for wi, center in enumerate(centers):
                slot = p*4+si*2+wi
                try:
                    out[slot] = rgb_triplet(files,int(center))
                    mask[slot] = True
                except Exception:
                    pass
    if not mask.any():
        raise ValueError(f'Could not decode any DICOM views for study {uid}')
    return out,mask


class StudyDataset(Dataset):
    def __init__(self, image_path, masks, labels, indices, total_n):
        self.image_path = str(image_path)
        self.total_n = int(total_n)
        self.images = None
        self.masks = masks
        self.labels = labels
        self.indices = np.asarray(indices,dtype=np.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,j):
        if self.images is None:
            self.images = np.memmap(self.image_path,dtype='uint8',mode='r',
                                    shape=(self.total_n,N_VIEWS,3,IMAGE_SIZE,IMAGE_SIZE))
        idx = int(self.indices[j])
        return (np.array(self.images[idx],copy=True),
                self.masks[idx].copy(),self.labels[idx].copy())


def prep_cache(uids, chosen, root, start):
    CACHE_DIR.mkdir(parents=True,exist_ok=True)
    n = len(uids)
    bytes_needed = n*N_VIEWS*3*IMAGE_SIZE*IMAGE_SIZE
    available = shutil.disk_usage(CACHE_DIR).free
    if available < bytes_needed+1_000_000_000:
        raise RuntimeError(f'Not enough free disk for cache: {available/1e9:.1f} GB')
    cache_path = CACHE_DIR/'weak_images.dat'
    image_data = np.memmap(cache_path,dtype='uint8',mode='w+',shape=(n,N_VIEWS,3,IMAGE_SIZE,IMAGE_SIZE))
    masks = np.zeros((n,N_VIEWS),bool)
    root = root/'train_series'
    for i,uid in enumerate(uids):
        if time.monotonic()-start > MAX_TRAIN_EVAL_SECONDS:
            raise TimeoutError('Aborting before Kaggle 9-hour limit: slow DICOM extraction')
        arr,m = study_images(uid,chosen,root)
        image_data[i] = arr
        masks[i] = m
        if (i+1)%50==0 or i+1==n:
            image_data.flush()
            elapsed = (time.monotonic()-start)/3600
            print(f'CACHE: {i+1}/{n} | elapsed={elapsed:.2f} hours',flush=True)
    image_data.flush()
    del image_data
    return cache_path,masks


class KneeMIL(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(checkpoint, local_files_only=True)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        if not hasattr(self.backbone, 'encoder') or not hasattr(self.backbone.encoder, 'layer'):
            raise RuntimeError('Unexpected Dinov2Model structure; no encoder.layer')
        for layer in self.backbone.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad_(True)
        dim = int(self.backbone.config.hidden_size)
        self.slot = nn.Parameter(torch.randn(1,N_VIEWS,dim)*0.015)
        self.attention = nn.Sequential(nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,1))
        self.head = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,256),nn.GELU(),
                                  nn.Dropout(0.25),nn.Linear(256,len(TARGETS)))
    def forward(self,image,mask):
        # image: [batch,12,3,224,224], batch=1 by design
        batch,views,ch,height,width = image.shape
        pixels = image.reshape(batch*views,ch,height,width)
        features = self.backbone(pixel_values=pixels).last_hidden_state[:,0]
        features = features.reshape(batch,views,-1)+self.slot
        attention = self.attention(features).squeeze(-1)
        attention = attention.masked_fill(~mask,-1e4)
        attention = torch.softmax(attention,dim=1)
        pool = (features*attention.unsqueeze(-1)).sum(dim=1)
        return self.head(pool)


def to_device(batch, device):
    images,masks,targets = batch
    images = images.to(device,dtype=torch.float32,non_blocking=True)/255.0
    mean = images.new_tensor([0.485,0.456,0.406]).view(1,1,3,1,1)
    std = images.new_tensor([0.229,0.224,0.225]).view(1,1,3,1,1)
    images = (images-mean)/std
    return images,masks.to(device,dtype=torch.bool),targets.to(device,dtype=torch.float32)


def score(y,p):
    values = {}
    for j,t in enumerate(TARGETS):
        keep = np.isfinite(y[:,j]) & np.isfinite(p[:,j])
        labels = y[keep,j]
        if len(np.unique(labels)) < 2:
            values[t] = float('nan')
        else:
            values[t] = float(roc_auc_score(labels,p[keep,j]))
    valid = [v for v in values.values() if np.isfinite(v)]
    return float(np.mean(valid)) if valid else float('nan'),values


def evaluate(model,loader,device):
    model.eval()
    truth,preds=[],[]
    with torch.inference_mode():
        for batch in loader:
            imgs,mask,y = to_device(batch,device)
            with torch.autocast('cuda',dtype=torch.float16):
                pred = torch.sigmoid(model(imgs,mask)).float().cpu().numpy()
            truth.append(y.cpu().numpy())
            preds.append(pred)
    return np.concatenate(truth),np.concatenate(preds)


def train_model(model,cache,masks,labels,device,start):
    train_idx,val_idx = train_test_split(np.arange(len(labels)),test_size=0.13,random_state=SEED)
    tr = StudyDataset(cache,masks,labels,train_idx,len(labels))
    val = StudyDataset(cache,masks,labels,val_idx,len(labels))
    train_loader = DataLoader(tr,batch_size=1,shuffle=True,num_workers=0,pin_memory=True)
    val_loader = DataLoader(val,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
    finetune_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.slot.parameters()) if isinstance(model.slot,nn.Module) else [model.slot]
    head_params += list(model.attention.parameters())+list(model.head.parameters())
    optimizer = torch.optim.AdamW([{'params':finetune_params,'lr':1.2e-5},
                                   {'params':head_params,'lr':5.0e-4}],weight_decay=0.03)
    scaler = torch.amp.GradScaler('cuda')
    best_auc,best_epoch = -float('inf'),0
    best_ckpt = OUT_DIR/'exp011_best.pt'
    for epoch in range(1,EPOCHS+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for i,batch in enumerate(train_loader):
            if time.monotonic()-start > MAX_TRAIN_EVAL_SECONDS:
                raise TimeoutError('Training time limit; not generating unverified submission')
            img,mask,y = to_device(batch,device)
            with torch.autocast('cuda',dtype=torch.float16):
                logits = model(img,mask)
                loss_each = nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
                confidence = (0.5 + torch.abs(y-0.5))
                loss = (loss_each*confidence).mean()
            scaler.scale(loss/ACCUMULATION).backward()
            if (i+1)%ACCUMULATION==0 or i+1==len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach().cpu())
            if (i+1)%500==0:
                print(f'EPOCH {epoch}: {i+1}/{len(tr)} train loss {epoch_loss/(i+1):.4f}',flush=True)
        _,p = evaluate(model,val_loader,device)
        y_hard = (labels[val_idx]>0.5).astype(int)
        val_auc, _ = score(y_hard,p)
        print(f'EPOCH {epoch}: weak holdout AUC={val_auc:.4f}, loss={epoch_loss/len(tr):.4f}',flush=True)
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc,best_epoch=val_auc,epoch
            torch.save(model.state_dict(),best_ckpt)
        if epoch-best_epoch>=2:
            break
    if best_epoch==0:
        raise RuntimeError('Weak holdout model validation failed')
    print(f'BEST WEAK AUC {best_auc:.4f} epoch {best_epoch}',flush=True)
    model.load_state_dict(torch.load(best_ckpt,map_location=device,weights_only=True))
    return best_auc,best_epoch


def predict_in_memory(model,images,masks,device,start,label):
    model.eval()
    outputs=[]
    with torch.inference_mode():
        for i in range(len(images)):
            if time.monotonic()-start > MAX_TOTAL_SECONDS:
                raise TimeoutError('Stopping before 9-hour Kaggle limit; no fallback predictions')
            img = torch.from_numpy(np.array(images[i:i+1],copy=True))
            mask = torch.from_numpy(np.array(masks[i:i+1],copy=True))
            fake_y = torch.zeros((1,len(TARGETS)))
            x,m,_ = to_device((img,mask,fake_y),device)
            with torch.autocast('cuda',dtype=torch.float16):
                out = torch.sigmoid(model(x,m)).float().cpu().numpy()[0]
            outputs.append(out)
            if (i+1)%25==0 or i+1==len(images):
                print(f'{label} {i+1}/{len(images)}',flush=True)
    return np.vstack(outputs)


def report_path_gold(root,chosen,gold,model,device,start):
    ims,masks=[],[]
    for uid in gold[UID].astype(str):
        img,mask = study_images(uid,chosen,root/'train_series')
        ims.append(img)
        masks.append(mask)
    print('GOLD images decoded',len(ims),flush=True)
    gold_probs = predict_in_memory(model,np.stack(ims),np.stack(masks),device,start,'GOLD PRED')
    return score(gold[TARGETS].to_numpy(int),gold_probs)


def main():
    start=time.monotonic()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError('EXP-011 must run with a Kaggle CUDA GPU')
    device=torch.device('cuda')
    root,ckpt,weak_path=paths()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    print('EXP-011: DINOv2 Small final 2 blocks fine-tuning, 140mm crop',flush=True)
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    train=pd.read_csv(root/'train.csv')
    series=pd.read_csv(root/'train_series.csv')
    test=pd.read_csv(root/'test.csv')
    test_series=pd.read_csv(root/'test_series.csv')
    weak=pd.read_csv(weak_path)
    for d in (train,series,test,test_series,weak):
        d[UID] = d[UID].astype(str)
    gold=train.loc[train[TARGETS].notna().all(axis=1),[UID]+TARGETS].copy().reset_index(drop=True)
    weak_train=train[[UID]].merge(weak,on=UID).loc[lambda x:~x[UID].isin(set(gold[UID]))].reset_index(drop=True)
    if len(gold)!=58 or len(weak_train)!=4349:
        raise ValueError(f'Expected gold=58/weak=4349, got {len(gold)}/{len(weak_train)}')
    y=pd.concat([pd.to_numeric(weak_train[t],errors='coerce') for t in TARGETS],axis=1).to_numpy(float)
    if np.any(~np.isfinite(y)):
        medians=np.nanmedian(y,axis=0)
        y=np.where(np.isfinite(y),y,medians[np.newaxis,:])
    if not (np.isfinite(y).all() and y.min()>=0 and y.max()<=1):
        raise ValueError('Invalid weak labels')
    y=y.astype(np.float32)
    selected_train=select_series(series)
    selected_test=select_series(test_series)
    cache,masks=prep_cache(weak_train[UID].tolist(),selected_train,root,start)
    print('WEAK CACHE READY',len(weak_train),'elapsed',round((time.monotonic()-start)/3600,2),'h',flush=True)
    model=KneeMIL(str(ckpt)).to(device)
    weak_auc,best_epoch=train_model(model,cache,masks,y,device,start)
    gold_auc,per_target=report_path_gold(root,selected_train,gold,model,device,start)
    print('EXP011 GOLD MACRO AUC:',round(gold_auc,4),flush=True)
    for k,v in per_target.items():
        print(f'  {k:20s} {v:.4f}',flush=True)
    approved=bool(np.isfinite(gold_auc) and gold_auc>=GOLD_SUBMIT_GATE and weak_auc>=0.70)
    print('EXP011 LOCAL GATE (>=0.820 gold, >=0.70 weak holdout):', 'PASS' if approved else 'FAIL',flush=True)
    info={'weak_auc':weak_auc,'best_epoch':best_epoch,'gold_auc':gold_auc,
          'per_target_auc':per_target,'gate':approved,'elapsed_hours':(time.monotonic()-start)/3600,
          'training_count':len(weak_train),'gold_count':len(gold),
          'notes':'Gold used only for final audit; never for fitting or epoch selection. Public score not guaranteed.'}
    with (OUT_DIR/'exp011_diagnostics.json').open('w',encoding='utf-8') as f:
        json.dump(info,f,indent=2)
    print('VALIDATION COMPLETE - test inference starts',flush=True)
    if time.monotonic()-start > MAX_TRAIN_EVAL_SECONDS:
        raise TimeoutError('Insufficient time for hidden-test inference; submission not created')
    # In scoring rerun, test can be ~1,322 studies; avoid silent partial fallback.
    preds=[]
    model.eval()
    with torch.inference_mode():
        for i,uid in enumerate(test[UID].astype(str)):
            if time.monotonic()-start>MAX_TOTAL_SECONDS:
                raise TimeoutError('Cannot finish all hidden test studies under 9h; no partial submission')
            pixels,mask=study_images(uid,selected_test,root/'test_series')
            x,m,_=to_device((torch.from_numpy(pixels[None]),
                             torch.from_numpy(mask[None]),
                             torch.zeros((1,len(TARGETS)))),device)
            with torch.autocast('cuda',dtype=torch.float16):
                pred=torch.sigmoid(model(x,m)).float().cpu().numpy()[0]
            preds.append(pred)
            if (i+1)%25==0 or i+1==len(test):
                print(f'TEST {i+1}/{len(test)} elapsed={(time.monotonic()-start)/3600:.2f}h',flush=True)
    sub=test[[UID]].copy()
    sub[TARGETS]=np.vstack(preds)
    assert len(sub)==len(test) and sub[UID].is_unique
    assert sub[TARGETS].notna().all().all()
    sub.to_csv(OUT_DIR/'submission.csv',index=False)
    # Weak memmap in temp, not a Kaggle output artifact.
    try:
        cache.unlink()
    except OSError:
        pass
    print('EXP011 COMPLETE','gold',gold_auc,'weak',weak_auc,'LOCAL GATE',approved,
          'hours',round((time.monotonic()-start)/3600,2),flush=True)
    print('DO NOT SUBMIT if gate FAIL; passing is not a 0.800 guarantee.',flush=True)


if __name__=='__main__':
    main()
