"""Audit the saved grouped-CV models without retuning on their held-out labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from threadpoolctl import threadpool_limits
from dataset import locate
from train import feature_matrix,member_predict
from reliability_common import summary,calibration,group_bootstrap,Monitor


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--previous',required=True); p.add_argument('--artifacts',required=True)
    args=p.parse_args(); old=Path(args.previous); out=Path(args.artifacts); out.mkdir(parents=True,exist_ok=True)
    found=locate(args.data_root); frame,paths,_=found['train']; y=frame.label.to_numpy()
    prior=json.loads((old/'metrics.json').read_text()); selected=prior['selected']
    groupdata=pd.read_csv(old/'duplicate_groups.csv').set_index('image_id'); groups=groupdata.loc[frame.image_id,'group'].to_numpy()
    names=[r['name'] for r in prior['experiments']]; predictions=[]; arrays={}; results={}
    for name in names:
        with np.load(old/f'{name}_oof.npz') as data: arrays[name]={k:data[k] for k in data.files}
        a=arrays[name]; pred=(a['score']>=.5).astype(int); predictions.append(pred)
        assert pd.DataFrame({'group':groups,'fold':a['fold']}).groupby('group').fold.nunique().max()==1
        result=summary(y,pred,a['probability']); fs=[summary(y[a['fold']==f],pred[a['fold']==f]) for f in range(1,6)]
        fold_scores=[v['balanced_accuracy'] for v in fs]
        result.update({'per_fold':fs,'fold_mean':float(np.mean(fold_scores)),'fold_std_sample':float(np.std(fold_scores,ddof=1)),
                       'worst_fold':min(fold_scores),'default_probability_05':summary(y,a['probability']>=.5),
                       'recall_gap':abs(result['recall_depth']-result['recall_rise']),
                       'checkpoint_bytes':(old/f'{name}.joblib').stat().st_size,
                       'recorded_cv_fit_seconds':next(r['seconds'] for r in prior['experiments'] if r['name']==name),
                       'historical_peak_ram':None,'historical_total_training_time':None})
        results[name]=result
    with threadpool_limits(limits=2): boot=group_bootstrap(y,groups,predictions)
    np.savez_compressed(out/'bootstrap_existing.npz',names=np.asarray(names),balanced_accuracy=boot)
    for j,name in enumerate(names): results[name]['confidence_interval_95']=np.quantile(boot[:,j],[.025,.975]).tolist()
    s=names.index(selected); b=names.index('existing-metadata-logistic')
    difference={'selected_minus_existing_ba':results[selected]['balanced_accuracy']-results['existing-metadata-logistic']['balanced_accuracy'],
                'paired_group_bootstrap_95':np.quantile(boot[:,s]-boot[:,b],[.025,.975]).tolist()}
    # Compare calibration to each estimator's native probability (sigmoid margin for SVM).
    reflect=np.load(old/'reflect_features.npz')['x']; constant=np.load(old/'constant_features.npz')['x']
    angle=np.deg2rad(frame.sun_azimuth_angle.to_numpy()); cyc=np.column_stack([np.sin(angle),np.cos(angle),np.sin(2*angle),np.cos(2*angle)]).astype(np.float32)
    designs={'existing-metadata-logistic':np.column_stack([constant,cyc]),'reflect-logistic':reflect,
             'reflect-rbf-balanced':reflect,'reflect-rbf-unbalanced':reflect,'angle-only-diagnostic':cyc}
    with threadpool_limits(limits=2):
        for name in names:
            model=joblib.load(old/f'{name}.joblib'); uncal=np.empty(len(y)); a=arrays[name]
            for f,member in enumerate(model['members'],1):
                ix=a['fold']==f; uncal[ix]=expit(member['model'].decision_function(designs[name][ix]))
            results[name]['uncalibrated_sigmoid_margin']=calibration(y,uncal)
            results[name]['uncalibrated_default_05']=summary(y,uncal>=.5)
            results[name]['calibration_improves_brier']=results[name]['calibration']['brier']<results[name]['uncalibrated_sigmoid_margin']['brier']
    traitpath=out/'image_traits.csv'
    if not traitpath.exists():
        def traits(path):
            with Image.open(path) as image: a=np.asarray(image.convert('L'),dtype=np.float32)/255
            return float(a.mean()),float(a.std()),float((a<.03).mean())
        with ThreadPoolExecutor(max_workers=4) as pool: values=list(pool.map(traits,paths))
        traits_df=pd.DataFrame(values,columns=['brightness','contrast','dark_fraction']); traits_df.insert(0,'image_id',frame.image_id)
        traits_df.to_csv(traitpath,index=False)
    traits_df=pd.read_csv(traitpath); assert traits_df.image_id.tolist()==frame.image_id.tolist()
    current=arrays[selected]; pred=current['score']>=.5; conditions={}
    for col in ['brightness','contrast']:
        cut=np.quantile(traits_df[col],[1/3,2/3]); bin_id=np.digitize(traits_df[col],cut)
        conditions[col]={'quantile_cutpoints':cut.tolist(),'bins':{label:summary(y[bin_id==j],pred[bin_id==j]) for j,label in enumerate(['low','medium','high'])}}
    az=[]
    for lower in range(0,360,45):
        ix=frame.sun_azimuth_angle.between(lower,lower+45,inclusive='left').to_numpy()
        az.append({'lower':lower,**summary(y[ix],pred[ix])})
    conditions['azimuth_bins']=az
    # Near 45 mod 90 needs maximal new corner support; near 0 mod 90 needs least.
    mod=frame.sun_azimuth_angle.to_numpy()%90
    for label,ix in [('near_maximum_rotation_padding',np.abs(mod-45)<=10),('near_right_angle',np.minimum(mod,90-mod)<=10)]:
        conditions[label]=summary(y[ix],pred[ix])
    counts=pd.Series(groups).value_counts(); duplicated=np.asarray([counts[g]>1 for g in groups])
    for label,ix in [('duplicate_group_rows',duplicated),('singleton_rows',~duplicated)]: conditions[label]=summary(y[ix],pred[ix])
    conflicting=pd.DataFrame({'g':groups,'y':y}).groupby('g').y.nunique(); ix=np.asarray([conflicting[g]>1 for g in groups])
    conditions['conflicting_label_group_rows']=summary(y[ix],pred[ix])
    results[selected]['worst_azimuth_bin']=min(m['balanced_accuracy'] for m in az)
    # Timed end-to-end inference, including loading/preprocessing; previous CSV is NOT replaced.
    test,testpaths,_=found['test']
    with Monitor() as monitor,threadpool_limits(limits=2):
        model=joblib.load(old/'pareidolia_final_model.joblib')
        x=feature_matrix(test,testpaths,'reflect')
        np.savez_compressed(out/'evaluation_hog_features.npz',x=x,image_id=test.image_id.to_numpy(dtype=str))
        member_output=[member_predict(member,x) for member in model['members']]
        probs=np.vstack([v[0] for v in member_output]); scores=np.vstack([v[1] for v in member_output]); votes=scores>=.5
        meanp=probs.mean(axis=0); means=scores.mean(axis=0); labels=(means>=.5).astype(int)
    original=pd.read_csv(old/'test_predictions.csv'); assert np.array_equal(labels,original.label)
    confidence=np.where(labels==1,meanp,1-meanp); agreement=np.maximum(votes.sum(axis=0),len(votes)-votes.sum(axis=0))
    diagnostics={'mean_selected_label_confidence':float(confidence.mean()),'median_selected_label_confidence':float(np.median(confidence)),
                 'mean_max_class_probability':float(np.maximum(meanp,1-meanp).mean()),
                 'confidence_ge_70':float(np.mean(confidence>=.7)),'confidence_ge_80':float(np.mean(confidence>=.8)),'confidence_ge_90':float(np.mean(confidence>=.9)),
                 'near_threshold_abs_score_within_005':int(np.sum(np.abs(means-.5)<=.05)),
                 'unanimous_agreement':float(np.mean(agreement==5)),'majority_agreement_at_least_3_of_5':float(np.mean(agreement>=3)),
                 'supermajority_agreement_at_least_4_of_5':float(np.mean(agreement>=4)),
                 'average_probability_std':float(probs.std(axis=0).mean()),
                 'raw_probability_soft_vote_disagrees_with_current_labels':int(np.sum((meanp>=.5)!=labels)),
                 'warning':'Confidence describes model probabilities, NOT hidden-label accuracy. For five binary members a 3/5 majority always exists.'}
    evaluation=pd.DataFrame({'image_id':test.image_id,'probability_rise':meanp,'decision_score':means,'label':labels,
                             'selected_label_confidence':confidence,'probability_std':probs.std(axis=0),'agreeing_members':agreement})
    for j in range(5): evaluation[f'member_{j+1}_probability']=probs[j]; evaluation[f'member_{j+1}_label']=votes[j].astype(int)
    evaluation.to_csv(out/'evaluation_probabilities.csv',index=False)
    diagnostics['most_disputed']=evaluation.sort_values(['agreeing_members','probability_std'],ascending=[True,False]).head(20).image_id.tolist()
    resource={**monitor.measurements,'images':2000,'images_per_second':2000/monitor.measurements['seconds'],
              'checkpoint_bytes':(old/'pareidolia_final_model.joblib').stat().st_size,
              'learned_dual_coefficients_and_intercepts':sum(m['model'][-1].dual_coef_.size+m['model'][-1].intercept_.size for m in model['members']),
              'stored_support_vector_values':sum(m['model'][-1].support_vectors_.size for m in model['members']),
              'neural_trainable_parameters':None,'note':'Kernel SVM parameter counts are not directly comparable with neural parameters; no CUDA available.'}
    output={'selected_existing_submission':selected,'experiments':results,'paired_comparison':difference,'conditions':conditions,
            'evaluation_confidence':diagnostics,'inference_efficiency':resource,
            'bootstrap':{'resamples':10000,'seed':2026,'method':'Composition-stratified duplicate-group cluster bootstrap; percentile interval, fixed OOF models.',
                         'limitations':'Conditional sampling uncertainty only; excludes retraining, model selection and deployment distribution shift. Lower bound is not a guaranteed future performance floor.'},
            'ensemble_validation_status':'Existing five-fold ensemble has no unbiased ensemble OOF score; stored OOF evaluates one held-out member per row.'}
    (out/'existing_reliability.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],'k--',label='Ideal')
    for label,key in [('Platt calibrated','calibration'),('Uncalibrated sigmoid margin','uncalibrated_sigmoid_margin')]:
        bins=results[selected][key]['bins']; ax.plot([v['mean_probability'] for v in bins],[v['observed_rise_fraction'] for v in bins],'o-',label=label)
    ax.set(xlabel='Predicted probability of Rise',ylabel='Observed Rise fraction',title='Held-out OOF calibration, 15 equal-width bins',xlim=(0,1),ylim=(0,1)); ax.legend(); fig.tight_layout()
    fig.savefig(out/'calibration_plot.png',dpi=160); plt.close(fig)
    print('EXISTING_RELIABILITY',json.dumps({'oof':results[selected]['balanced_accuracy'],'ci':results[selected]['confidence_interval_95'],'comparison':difference,'diagnostics':diagnostics,'efficiency':resource}),flush=True)


if __name__=='__main__': main()
