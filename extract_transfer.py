"""Frozen ImageNet CNN features; no training-label access or random transforms."""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models import resnet18,ResNet18_Weights,mobilenet_v3_small,MobileNet_V3_Small_Weights
from PIL import Image
from dataset import locate,fingerprint
from ml.preprocessing import normalize_solar_azimuth
from reliability_common import Monitor


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--artifacts',required=True)
    p.add_argument('--download-only',action='store_true'); p.add_argument('--split',choices=['all','test'],default='all'); args=p.parse_args()
    out=Path(args.artifacts); out.mkdir(parents=True,exist_ok=True); torch.hub.set_dir(str(out/'torch_hub'))
    torch.set_num_threads(6); torch.manual_seed(2026); torch.use_deterministic_algorithms(True)
    definitions=[('mobilenet_v3_small',mobilenet_v3_small,MobileNet_V3_Small_Weights.IMAGENET1K_V1),('resnet18',resnet18,ResNet18_Weights.IMAGENET1K_V1)]
    if args.download_only:
        for name,constructor,weights in definitions:
            weights.get_state_dict(progress=True,check_hash=True)
            print('download verified',name,flush=True)
        return
    found=locate(args.data_root); digest=fingerprint(found)
    frame=pd.concat([found['train'][0],found['test'][0]],ignore_index=True); paths=found['train'][1]+found['test'][1]
    if args.split=='test': frame=found['test'][0].reset_index(drop=True); paths=found['test'][1]
    for name,constructor,weights in definitions:
        model=constructor(weights=weights).eval()
        original_parameters=sum(x.numel() for x in model.parameters())
        if name=='resnet18': model.fc=torch.nn.Identity()
        else: model.classifier=torch.nn.Identity()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for view in (['reflect'] if name!='resnet18' else ['reflect','raw']):
            suffix='' if args.split=='all' else '_test'
            target=out/f'{name}_{view}{suffix}_features.npz'
            key=digest+'_'+name+'_'+view+'_imagenet-v1-center224'+suffix
            if target.exists():
                with np.load(target) as data: assert str(data['key'])==key
                print('reuse',target.name,flush=True); continue
            def load(item):
                path,angle=item
                with Image.open(path) as image: image=image.convert('L')
                if view=='reflect': image=normalize_solar_azimuth(image,angle,border_mode='reflect')
                # Official V1 transform: 256 resize then 224 center crop, RGB and ImageNet normalization.
                image=image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR).crop((16,16,240,240))
                return np.asarray(image,dtype=np.float32).transpose(2,0,1)/255
            features=[]
            with Monitor() as monitor,torch.inference_mode(),ThreadPoolExecutor(max_workers=4) as pool:
                for begin in range(0,len(paths),32):
                    batch=torch.from_numpy(np.stack(list(pool.map(load,zip(paths[begin:begin+32],frame.sun_azimuth_angle.iloc[begin:begin+32])))))
                    batch=(batch-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
                    features.append(model(batch).numpy())
                    if begin%512==0: print(name,view,begin+len(batch),'/',len(paths),flush=True)
            x=np.vstack(features).astype(np.float32)
            np.savez_compressed(target,x=x,key=key,image_id=frame.image_id.to_numpy(dtype=str))
            stats={**monitor.measurements,'architecture':name,'view':view,'feature_dimension':x.shape[1],
                   'frozen_backbone_parameters':sum(v.numel() for v in model.parameters()),'original_architecture_parameters':original_parameters,
                   'backbone_trainable_parameters':0,'images':len(x),'images_per_second':len(x)/monitor.measurements['seconds'],
                   'torch':torch.__version__,'torchvision':torchvision.__version__,'weights_url':weights.url,
                   'feature_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'data_fingerprint':digest}
            target.with_suffix('.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
            print('COMPLETE',json.dumps(stats),flush=True)


if __name__=='__main__': main()
