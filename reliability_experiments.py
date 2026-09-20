"""Additional prespecified candidates on exactly the previous outer folds."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from dataset import locate,load_config
from ml.features import extract_feature_vector,FeatureConfig
from train import evaluate,blocked,member_predict
from reliability_common import Monitor


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--previous',required=True); p.add_argument('--artifacts',required=True)
    p.add_argument('--family',choices=['classical','transfer'],required=True); p.add_argument('--candidate'); args=p.parse_args()
    old=Path(args.previous); out=Path(args.artifacts); out.mkdir(parents=True,exist_ok=True)
    config=load_config(Path(__file__).with_name('config.yaml')); config['threads']=2
    found=locate(args.data_root); frame,paths,_=found['train']; y=frame.label.to_numpy(); angles=frame.sun_azimuth_angle.to_numpy()
    groupdata=pd.read_csv(old/'duplicate_groups.csv').set_index('image_id'); groups=groupdata.loc[frame.image_id,'group'].to_numpy()
    if args.family=='classical':
        cache=out/'raw_hog_features.npz'
        if not cache.exists():
            def one(item): return extract_feature_vector(item[0],item[1],apply_solar_normalization=False,config=FeatureConfig())
            with Monitor() as monitor,ThreadPoolExecutor(max_workers=4) as pool:
                rows=[]
                for i,v in enumerate(pool.map(one,zip(paths,angles)),1):
                    rows.append(v)
                    if i%1000==0: print('raw HOG',i,flush=True)
            np.savez_compressed(cache,x=np.asarray(rows,dtype=np.float32),image_id=frame.image_id.to_numpy(dtype=str))
            cache.with_suffix('.json').write_text(json.dumps(monitor.measurements,indent=2),encoding='utf-8')
        designs={'raw-hog-logistic':(np.load(cache)['x'],None),
                 'constant-image-logistic':(np.load(old/'constant_features.npz')['x'],None)}
    else:
        designs={}
        for name,feature in [('mobilenet-reflect-logistic','mobilenet_v3_small_reflect'),('resnet-reflect-logistic','resnet18_reflect'),
                             ('resnet-reflect-rbf-balanced','resnet18_reflect'),('resnet-raw-logistic','resnet18_raw')]:
            if args.candidate and name!=args.candidate: continue
            with np.load(out/f'{feature}_features.npz') as data:
                assert data['image_id'][:len(y)].tolist()==frame.image_id.tolist()
                designs[name]=(data['x'][:len(y)],data['x'][len(y):])
    for name,(x,xtest) in designs.items():
        if args.candidate and name!=args.candidate: continue
        target=out/f'{name}_metrics.json'
        existing=None
        if target.exists():
            existing=json.loads(target.read_text())
            if 'resource_measurements' in existing: print('reuse',name,flush=True); continue
        reuse_cv=existing is not None and (out/f'{name}_oof.npz').exists() and (out/f'{name}.joblib').exists()
        with Monitor() as monitor,threadpool_limits(limits=2):
            result=existing if reuse_cv else evaluate(name,x,y,angles,groups,config,out)
            if reuse_cv: print('reuse completed CV; resume sector checks',name,flush=True)
            result['blocked_azimuth']=blocked(name,x,y,angles,groups,config)
            result['blocked_mean_balanced_accuracy']=float(np.mean([v['balanced_accuracy'] for v in result['blocked_azimuth']]))
        result['resource_measurements']=monitor.measurements
        if reuse_cv:
            result['resource_measurements']['resumed_stage_seconds']=monitor.measurements['seconds']
            result['resource_measurements']['seconds']+=result['seconds']
            result['resource_measurements']['coverage_note']='Summed recorded CV time and resumed sector checks. Peak memory covers resumed stage only; interrupted CV peak was not saved.'
        result['checkpoint_bytes']=(out/f'{name}.joblib').stat().st_size
        if xtest is not None:
            model=joblib.load(out/f'{name}.joblib')
            with Monitor() as inference_monitor,threadpool_limits(limits=2): outputs=[member_predict(m,xtest) for m in model['members']]
            np.savez_compressed(out/f'{name}_evaluation.npz',probability=np.vstack([v[0] for v in outputs]),score=np.vstack([v[1] for v in outputs]))
            result['cached_feature_inference_measurements']=inference_monitor.measurements
            result['cached_feature_inference_measurements']['images_per_second']=2000/inference_monitor.measurements['seconds']
            estimator=model['members'][0]['model'][-1]
            result['fitted_parameters_per_member']=int(estimator.coef_.size+estimator.intercept_.size) if hasattr(estimator,'coef_') else None
        target.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('CANDIDATE_COMPLETE',name,result['oof']['balanced_accuracy'],result['blocked_mean_balanced_accuracy'],flush=True)


if __name__=='__main__': main()
