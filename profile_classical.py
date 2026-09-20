"""Measure preprocessing and checkpoint inference stages without changing predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from dataset import locate
from ml.features import FeatureConfig,extract_feature_vector
from train import member_predict
from reliability_common import Monitor


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--previous',required=True); p.add_argument('--artifacts',required=True); args=p.parse_args()
    old=Path(args.previous); out=Path(args.artifacts); frame,paths,_=locate(args.data_root)['test']; matrices={}; profiles={}
    for view in ['raw','constant','reflect']:
        def one(item): return extract_feature_vector(item[0],item[1],apply_solar_normalization=view!='raw',config=FeatureConfig(border_mode='reflect' if view=='reflect' else 'constant'))
        with Monitor() as monitor,ThreadPoolExecutor(max_workers=3) as pool: matrices[view]=np.asarray(list(pool.map(one,zip(paths,frame.sun_azimuth_angle))),dtype=np.float32)
        profiles[view]={**monitor.measurements,'images':2000,'images_per_second':2000/monitor.measurements['seconds']}
        (out/f'{view}_hog_test_features.json').write_text(json.dumps(profiles[view],indent=2),encoding='utf-8')
        print('PROFILE preprocessing',view,monitor.measurements['seconds'],flush=True)
    angle=np.deg2rad(frame.sun_azimuth_angle.to_numpy()); cyc=np.column_stack([np.sin(angle),np.cos(angle),np.sin(2*angle),np.cos(2*angle)]).astype(np.float32)
    designs={'existing-metadata-logistic':(old,np.column_stack([matrices['constant'],cyc]),'constant'),
             'reflect-logistic':(old,matrices['reflect'],'reflect'),'reflect-rbf-balanced':(old,matrices['reflect'],'reflect'),
             'reflect-rbf-unbalanced':(old,matrices['reflect'],'reflect'),'angle-only-diagnostic':(old,cyc,None),
             'raw-hog-logistic':(out,matrices['raw'],'raw'),'constant-image-logistic':(out,matrices['constant'],'constant')}
    results={}
    with threadpool_limits(limits=2):
        for name,(directory,x,view) in designs.items():
            path=directory/f'{name}.joblib'
            with Monitor() as monitor:
                members=joblib.load(path)['members']; predictions=[member_predict(m,x) for m in members]
            stage=monitor.measurements['seconds']; preprocessing=profiles[view]['seconds'] if view else 0
            counts=[]
            for m in members:
                estimator=m['model'][-1]
                learned=estimator.coef_.size+estimator.intercept_.size if hasattr(estimator,'coef_') else estimator.dual_coef_.size+estimator.intercept_.size
                counts.append(int(learned+m['calibrator'].coef_.size+m['calibrator'].intercept_.size))
            results[name]={**monitor.measurements,'checkpoint_bytes':path.stat().st_size,'images':2000,
                           'preprocessing_stage_seconds':preprocessing,'staged_total_seconds':stage+preprocessing,
                           'staged_images_per_second':2000/(stage+preprocessing),'fitted_coefficients_including_calibration':sum(counts),
                           'notes':'Sum of separately measured preprocessing and load/predict stages; excludes CSV export, includes no fitting. Kernel dual counts are not neural parameter counts.'}
            print('PROFILE',name,json.dumps(results[name]),flush=True)
    (out/'classical_inference_profiles.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__': main()
