"""Separate deterministic perturbation diagnostics; never change primary OOF inputs."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from dataset import locate
from ml.features import FeatureConfig,extract_feature_vector
from ml.preprocessing import normalize_solar_azimuth
from train import member_predict
from reliability_common import summary,Monitor


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--previous',required=True); p.add_argument('--artifacts',required=True)
    args=p.parse_args(); old=Path(args.previous); out=Path(args.artifacts)
    frame,paths,_=locate(args.data_root)['train']; y=frame.label.to_numpy()
    model=joblib.load(old/'pareidolia_final_model.joblib')
    with np.load(old/'reflect-rbf-balanced_oof.npz') as saved: fold=saved['fold']; baseline=saved['score']>=.5
    report={'baseline':summary(y,baseline),'note':'Only these explicitly requested diagnostic copies are perturbed. Primary validation/test inputs and submitted predictions remain unchanged.'}
    for variant in ['gaussian_noise_sigma_1_graylevel','bilinear_rotation']:
        target=out/f'stress_{variant}.npz'
        if target.exists():
            with np.load(target) as data: x=data['x']
        else:
            def one(item):
                index,path,angle=item
                with Image.open(path) as image: image=image.convert('L')
                if variant.startswith('gaussian'):
                    rng=np.random.default_rng(2026+index)
                    a=np.clip(np.asarray(image,dtype=float)+rng.normal(0,1,image.size[::-1]),0,255).round().astype(np.uint8)
                    image=Image.fromarray(a)
                normalized=normalize_solar_azimuth(image,angle,border_mode='reflect',resample=Image.Resampling.BILINEAR if variant=='bilinear_rotation' else Image.Resampling.BICUBIC)
                return extract_feature_vector(normalized,angle,apply_solar_normalization=False,config=FeatureConfig(border_mode='reflect'))
            rows=[]
            with Monitor() as monitor,ThreadPoolExecutor(max_workers=3) as pool:
                for i,v in enumerate(pool.map(one,zip(range(len(y)),paths,frame.sun_azimuth_angle)),1):
                    rows.append(v)
                    if i%1000==0: print(variant,i,flush=True)
            x=np.asarray(rows,dtype=np.float32); np.savez_compressed(target,x=x)
        p=np.empty(len(y)); score=np.empty(len(y))
        with threadpool_limits(limits=2):
            for f,m in enumerate(model['members'],1):
                ix=fold==f; p[ix],score[ix]=member_predict(m,x[ix])
        report[variant]={**summary(y,score>=.5,p),'prediction_flip_fraction':float(np.mean((score>=.5)!=baseline)),
                         'balanced_accuracy_change':summary(y,score>=.5)['balanced_accuracy']-report['baseline']['balanced_accuracy']}
        np.savez_compressed(out/f'stress_{variant}_oof.npz',probability=p,score=score,fold=fold)
        print('STRESS',variant,json.dumps(report[variant]),flush=True)
    (out/'stress_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()
