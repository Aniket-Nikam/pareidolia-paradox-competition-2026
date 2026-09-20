"""Outer-group-held-out evaluation of a five-inner-fold SVM ensemble."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from dataset import locate,load_config
from train import fit_member,member_predict
from reliability_common import Monitor,summary


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--previous',required=True); p.add_argument('--artifacts',required=True); args=p.parse_args()
    old=Path(args.previous); out=Path(args.artifacts)
    frame,_,_=locate(args.data_root)['train']; y=frame.label.to_numpy()
    groupmap=pd.read_csv(old/'duplicate_groups.csv').set_index('image_id'); groups=groupmap.loc[frame.image_id,'group'].to_numpy()
    x=np.load(old/'reflect_features.npz')['x']; config=load_config(Path(__file__).with_name('config.yaml')); config['threads']=2
    fold=np.load(old/'reflect-rbf-balanced_oof.npz')['fold']; probability=np.zeros(len(y)); score=np.zeros(len(y)); results=[]; total_seconds=0; peak=0; system_peak=0
    for f in range(1,6):
        target=out/f'nested_ensemble_outer_{f}.joblib'; saved=out/f'nested_ensemble_outer_{f}.npz'; stats=out/f'nested_ensemble_outer_{f}.json'
        va=np.flatnonzero(fold==f); tr=np.flatnonzero(fold!=f)
        if saved.exists():
            data=np.load(saved); probability[va]=data['probability']; score[va]=data['score']; result=json.loads(stats.read_text())
        else:
            splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026)
            members=[]; predictions=[]
            with Monitor() as monitor,threadpool_limits(limits=2):
                for j,(fit,_) in enumerate(splitter.split(x[tr],y[tr],groups[tr]),1):
                    ix=tr[fit]; assert not set(groups[ix])&set(groups[va])
                    member=fit_member('reflect-rbf-balanced',x[ix],y[ix],groups[ix],config,2026+j)
                    members.append(member); predictions.append(member_predict(member,x[va]))
                    print('nested ensemble outer',f,'member',j,'complete',flush=True)
                probability[va]=np.mean([v[0] for v in predictions],axis=0); score[va]=np.mean([v[1] for v in predictions],axis=0)
            result={'fold':f,'soft_probability_default_05':summary(y[va],probability[va]>=.5),
                    'training_threshold_adjusted_vote':summary(y[va],score[va]>=.5),'resources':monitor.measurements,
                    'member_thresholds':[m['threshold'] for m in members]}
            joblib.dump({'members':members,'config':config},target,compress=3)
            np.savez_compressed(saved,probability=probability[va],score=score[va]); stats.write_text(json.dumps(result,indent=2),encoding='utf-8')
        results.append(result); total_seconds+=result['resources']['seconds']; peak=max(peak,result['resources']['peak_process_rss_bytes_sampled']); system_peak=max(system_peak,result['resources']['peak_system_used_ram_bytes_sampled'])
        print('NESTED_FOLD',f,result['training_threshold_adjusted_vote']['balanced_accuracy'],flush=True)
    np.savez_compressed(out/'nested-fivefold-svm_oof.npz',probability=probability,score=score,fold=fold)
    report={'name':'nested-fivefold-svm','oof':summary(y,score>=.5,probability),'soft_probability_default_05':summary(y,probability>=.5,probability),
            'folds':results,'total_fit_and_validation_seconds':total_seconds,'peak_process_rss_bytes_sampled':peak,'peak_system_used_ram_bytes_sampled':system_peak,
            'all_outer_checkpoint_bytes':sum((out/f'nested_ensemble_outer_{f}.joblib').stat().st_size for f in range(1,6)),
            'caveat':'Each outer validation row is unseen by every member of its ensemble. Members use 80% of each outer training split (64% of the full data); this is nested validation of the fivefold-ensemble procedure, not an exact measurement of the already-fitted final checkpoint.'}
    (out/'nested-fivefold-svm_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('NESTED_COMPLETE',json.dumps(report['oof']),flush=True)


if __name__=='__main__': main()
