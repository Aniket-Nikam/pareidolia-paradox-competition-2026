"""CPU reproducible nested-calibration CV and azimuth-block robustness audit."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from dataset import locate, load_config, fingerprint
from ml.features import FeatureConfig, extract_feature_vector


def feature_matrix(frame, paths, border_mode, metadata=False):
    config = FeatureConfig(border_mode=border_mode, include_metadata=metadata)
    def one(item):
        path, angle = item
        return extract_feature_vector(path, angle, apply_solar_normalization=True, config=config)
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,row in enumerate(pool.map(one, zip(paths, frame.sun_azimuth_angle)),1):
            rows.append(row)
            if i%1000==0: print(f'{border_mode} features {i}/{len(frame)}',flush=True)
    return np.asarray(rows,dtype=np.float32)


def new_model(name, config):
    if 'rbf' in name:
        model=SVC(C=config['svm_c'],gamma=config['svm_gamma'],class_weight='balanced' if name.endswith('balanced') and not name.endswith('unbalanced') else None,cache_size=512,random_state=config['seed'])
    else:
        model=LogisticRegression(C=1 if name=='existing-metadata-logistic' else .1,max_iter=1500,solver='liblinear',class_weight='balanced',random_state=config['seed'])
    return make_pipeline(StandardScaler(),model)


def metrics(y,p,threshold=.5):
    pred=(p>=threshold).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1])
    return {'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'recall_depth':float(cm[0,0]/cm[0].sum()) if cm[0].sum() else None,
            'recall_rise':float(cm[1,1]/cm[1].sum()) if cm[1].sum() else None,
            'confusion_matrix':cm.tolist(), 'threshold':float(threshold),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'n':len(y)}


def threshold_search(y,p,config):
    low,high,n=config['threshold_grid']
    choices=np.linspace(low,high,n)
    # Ties prefer 0.5 instead of an extreme cutoff.
    return float(max(choices,key=lambda t:(balanced_accuracy_score(y,p>=t),-abs(t-.5))))


def fit_member(name,x,y,groups,config,seed):
    # Calibration and threshold fitting see only the outer training partition.
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed)
    fit,cal=next(splitter.split(x,y,groups))
    base=new_model(name,config).fit(x[fit],y[fit])
    score=base.decision_function(x[cal]).reshape(-1,1)
    calibrator=LogisticRegression(C=1,random_state=seed).fit(score,y[cal])
    probability=calibrator.predict_proba(score)[:,1]
    threshold=threshold_search(y[cal],probability,config)
    # Keep threshold selection independent of the outer validation labels.
    model=new_model(name,config).fit(x,y)
    return {'model':model,'calibrator':calibrator,'threshold':threshold,'calibration_n':len(cal)}


def member_predict(member,x):
    probability=member['calibrator'].predict_proba(member['model'].decision_function(x).reshape(-1,1))[:,1]
    # This normalized decision score has a common cutoff of .5 across members.
    adjusted=expit(logit(np.clip(probability,1e-7,1-1e-7))-logit(member['threshold']))
    return probability,adjusted


def evaluate(name,x,y,angles,groups,config,out):
    start=time.perf_counter()
    splits=StratifiedGroupKFold(n_splits=config['folds'],shuffle=True,random_state=config['seed'])
    oof=np.zeros(len(y)); raw=np.zeros(len(y)); folds=np.zeros(len(y),dtype=int)
    members=[]; results=[]
    for f,(tr,va) in enumerate(splits.split(x,y,groups),1):
        assert not set(groups[tr]) & set(groups[va])
        member=fit_member(name,x[tr],y[tr],groups[tr],config,config['seed']+f)
        raw[va],oof[va]=member_predict(member,x[va]); folds[va]=f
        m=metrics(y[va],raw[va],member['threshold']); m['fold']=f
        results.append(m); members.append(member)
        print(name,'fold',f,json.dumps(m),flush=True)
    bins=[]
    for lower in range(0,360,45):
        ix=(angles>=lower)&(angles<lower+45)
        if ix.any(): bins.append({'start':lower,**metrics(y[ix],oof[ix])})
    result={'name':name,'oof':metrics(y,oof),'folds':results,'azimuth_bins':bins,'seconds':time.perf_counter()-start}
    joblib.dump({'members':members,'config':config,'name':name},out/f'{name}.joblib',compress=3)
    np.savez_compressed(out/f'{name}_oof.npz',score=oof,probability=raw,fold=folds)
    (out/f'{name}_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def blocked(name,x,y,angles,groups,config):
    results=[]
    for lower in range(0,360,90):
        va=np.flatnonzero((angles>=lower)&(angles<lower+90))
        tr=np.flatnonzero(~np.isin(groups,groups[va]))
        if len(np.unique(y[va]))<2 or len(np.unique(y[tr]))<2:
            results.append({'start':lower,'skipped':'one class missing'}); continue
        member=fit_member(name,x[tr],y[tr],groups[tr],config,config['seed']+lower)
        p,score=member_predict(member,x[va])
        result={'start':lower,**metrics(y[va],p,member['threshold'])}
        results.append(result); print('AZIMUTH_BLOCK',name,json.dumps(result),flush=True)
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--artifacts',default='artifacts')
    p.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); config=load_config(args.config)
    if config['image_size'] != 48 or config['border_mode'] != 'reflect':
        raise ValueError('This feature version requires image_size=48 and border_mode=reflect')
    np.random.seed(config['seed']); out=Path(args.artifacts); out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((out/'data_audit.json').read_text())
    found=locate(args.data_root); frame,paths,_=found['train']; y=frame.label.to_numpy(); angles=frame.sun_azimuth_angle.to_numpy()
    actual_fingerprint=fingerprint(found)
    if actual_fingerprint!=audit['fingerprint']: raise ValueError('Dataset changed; rerun data_checks.py')
    signature=hashlib.sha256((actual_fingerprint+json.dumps(config,sort_keys=True)).encode())
    for source in [Path(__file__), Path(__file__).parent/'ml/features.py', Path(__file__).parent/'ml/preprocessing.py', out/'duplicate_groups.csv']:
        signature.update(source.read_bytes())
    run_key=signature.hexdigest()
    group_df=pd.read_csv(out/'duplicate_groups.csv').set_index('image_id'); groups=group_df.loc[frame.image_id,'group'].to_numpy()
    feature_key=config['feature_version']+'_'+actual_fingerprint
    matrices={}
    for border in ['constant','reflect']:
        cache=out/f'{border}_features.npz'
        if cache.exists():
            with np.load(cache) as saved:
                if str(saved['key'])!=feature_key: raise ValueError('Stale feature cache')
                matrices[border]=saved['x']
        else:
            matrices[border]=feature_matrix(frame,paths,border)
            np.savez_compressed(cache,x=matrices[border],key=feature_key)
    radians=np.deg2rad(angles)
    cyc=np.column_stack([np.sin(radians),np.cos(radians),np.sin(2*radians),np.cos(2*radians)]).astype(np.float32)
    designs={'existing-metadata-logistic':np.column_stack([matrices['constant'],cyc]),
             'reflect-logistic':matrices['reflect'],'reflect-rbf-balanced':matrices['reflect'],
             'reflect-rbf-unbalanced':matrices['reflect'],'angle-only-diagnostic':cyc}
    results=[]
    with threadpool_limits(limits=config['threads']):
        for name,x in designs.items():
            saved=out/f'{name}_metrics.json'
            if args.resume and saved.exists():
                result=json.loads(saved.read_text())
                if result.get('run_key') != run_key: raise ValueError('Resume configuration, source, or data changed; use a new artifacts directory')
            else:
                result=evaluate(name,x,y,angles,groups,config,out)
                result['run_key']=run_key
                saved.write_text(json.dumps(result,indent=2),encoding='utf-8')
            results.append(result)
        image_candidates=[r for r in results if r['name'].startswith('reflect-')]
        # Evaluate every image-only candidate and the original under held-out 90-degree sectors.
        for result in results:
            if 'blocked_azimuth' not in result:
                result['blocked_azimuth']=blocked(result['name'],designs[result['name']],y,angles,groups,config)
                result['blocked_mean_balanced_accuracy']=float(np.mean([r['balanced_accuracy'] for r in result['blocked_azimuth'] if 'balanced_accuracy' in r]))
                (out/f"{result['name']}_metrics.json").write_text(json.dumps(result,indent=2),encoding='utf-8')
        # Balance ordinary CV and distribution-shift robustness; direct angle is diagnostic only.
        for result in image_candidates:
            result['selection_score']=.5*(result['oof']['balanced_accuracy']+result['blocked_mean_balanced_accuracy'])
        winner=max(image_candidates,key=lambda r:r['selection_score'])
    selected=joblib.load(out/f"{winner['name']}.joblib")
    selected.update({'fingerprint':actual_fingerprint,'feature_config':asdict(FeatureConfig(border_mode='reflect')),
                     'final_cutoff':.5,'feature_version':config['feature_version'],'selection':winner})
    joblib.dump(selected,out/'pareidolia_final_model.joblib',compress=3)
    with np.load(out/f"{winner['name']}_oof.npz") as saved:
        pd.DataFrame({'image_id':frame.image_id,'label':y,'fold':saved['fold'],'group':groups,
                      'probability_rise':saved['probability'],'decision_score':saved['score']}).to_csv(out/'oof_predictions.csv',index=False)
    final={'selected':winner['name'],'selection_rule':'Image-only candidates: mean of grouped OOF BA and mean held-out 90-degree sector BA',
           'fingerprint':actual_fingerprint,'config':config,'experiments':results,
           'limitations':['Model selection uses these validation results, so reported winner score is not an untouched test estimate.',
                          'Inner calibration and threshold use the same inner holdout; outer validation remains independent.',
                          'Near-duplicate grouping is conservative; unknown scene families may remain.',
                          'No CUDA available; no pretrained CNN was trained.']}
    (out/'metrics.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print('SELECTED',winner['name'],json.dumps(winner['oof']),flush=True)


if __name__=='__main__': main()
