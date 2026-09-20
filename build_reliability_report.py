"""Assemble measured reliability, candidate comparisons and an explicit selection gate."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from dataset import locate,fingerprint
from reliability_common import summary,group_bootstrap
from validate_submission import check


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--previous',required=True); p.add_argument('--artifacts',required=True)
    p.add_argument('--reports',required=True); p.add_argument('--submission',required=True); p.add_argument('--legacy-metrics'); args=p.parse_args()
    old=Path(args.previous); out=Path(args.artifacts); reports=Path(args.reports); reports.mkdir(parents=True,exist_ok=True)
    found=locate(args.data_root); frame=found['train'][0]; y=frame.label.to_numpy(); angles=frame.sun_azimuth_angle.to_numpy()
    data_audit=json.loads((old/'data_audit.json').read_text())
    assert fingerprint(found)==data_audit['fingerprint'],'Dataset changed since the audited experiments'
    groups=pd.read_csv(old/'duplicate_groups.csv').set_index('image_id').loc[frame.image_id,'group'].to_numpy()
    existing=json.loads((out/'existing_reliability.json').read_text()); oldmetrics=json.loads((old/'metrics.json').read_text())
    arrays={}; measured={}; candidates={}
    for directory in [old,out]:
        for path in directory.glob('*_metrics.json'):
            name=path.name.removesuffix('_metrics.json'); datafile=directory/f'{name}_oof.npz'
            if not datafile.exists(): continue
            measured[name]=json.loads(path.read_text())
            with np.load(datafile) as data: arrays[name]={k:data[k] for k in data.files}
    primary='nested-fivefold-svm'
    assert primary in arrays and (out/'nested_diagnostics.json').exists()
    reference_fold=arrays['reflect-rbf-balanced']['fold']
    for name,a in arrays.items():
        assert np.array_equal(a['fold'],reference_fold),name
        assert pd.DataFrame({'group':groups,'fold':a['fold']}).groupby('group').fold.nunique().max()==1
    # Equal-weight architecture/view ensembles: no OOF fitting of weights or thresholds.
    pairs=[('hog-resnet', 'reflect-rbf-balanced','resnet-reflect-logistic'),
           ('raw-normalized-resnet','resnet-raw-logistic','resnet-reflect-logistic'),
           ('resnet-mobilenet','resnet-reflect-logistic','mobilenet-reflect-logistic')]
    for title,left,right in pairs:
        probability=(arrays[left]['probability']+arrays[right]['probability'])/2
        adjusted=(arrays[left]['score']+arrays[right]['score'])/2
        for suffix,score in [('soft-probability',probability),('training-threshold-adjusted',adjusted)]:
            name=title+'-'+suffix; arrays[name]={'probability':probability,'score':score,'fold':reference_fold}
            measured[name]={'components':[left,right],'note':'Equal weights fixed before inspecting ensemble scores. Both component predictions are genuinely held out. No ensemble-specific OOF threshold optimization.'}
            np.savez_compressed(out/f'{name}_oof.npz',**arrays[name])
    arrays['nested-fivefold-svm-soft-probability']={**arrays[primary],'score':arrays[primary]['probability']}
    measured['nested-fivefold-svm-soft-probability']={'components':[primary],'note':'Default 0.5 soft vote on calibrated probabilities, same nested ensemble fits.'}
    names=list(arrays)
    with threadpool_limits(limits=2): boot=group_bootstrap(y,groups,[(arrays[n]['score']>=.5) for n in names])
    np.savez_compressed(out/'bootstrap_all_candidates.npz',names=np.asarray(names),balanced_accuracy=boot)
    baseline=names.index('existing-metadata-logistic'); single=names.index('reflect-rbf-balanced')
    for j,name in enumerate(names):
        a=arrays[name]; pred=a['score']>=.5; folds=[summary(y[reference_fold==f],pred[reference_fold==f]) for f in range(1,6)]
        azimuth=[]
        for lower in range(0,360,45):
            ix=(angles>=lower)&(angles<lower+45); azimuth.append({'lower':lower,**summary(y[ix],pred[ix])})
        stats=summary(y,pred,a['probability']); ci=np.quantile(boot[:,j],[.025,.975]).tolist()
        stats.update({'confidence_interval_95':ci,'conservative_reliability_floor':ci[0],
                      'per_fold':folds,'fold_mean':float(np.mean([f['balanced_accuracy'] for f in folds])),
                      'fold_std_sample':float(np.std([f['balanced_accuracy'] for f in folds],ddof=1)),
                      'worst_fold':min(f['balanced_accuracy'] for f in folds),'azimuth_bins':azimuth,
                      'worst_azimuth_bin':min(f['balanced_accuracy'] for f in azimuth),
                      'macro_azimuth_bin_ba':float(np.mean([f['balanced_accuracy'] for f in azimuth])),
                      'default_probability_05':summary(y,a['probability']>=.5),
                      'recall_gap':abs(stats['recall_depth']-stats['recall_rise']),
                      'paired_difference_vs_existing_95':np.quantile(boot[:,j]-boot[:,baseline],[.025,.975]).tolist(),
                      'paired_difference_vs_single_svm_95':np.quantile(boot[:,j]-boot[:,single],[.025,.975]).tolist(),
                      'measured_run':measured[name]})
        if 'components' in measured[name]:
            stats['paired_differences_vs_components_95']={c:np.quantile(boot[:,j]-boot[:,names.index(c)],[.025,.975]).tolist() for c in measured[name]['components']}
        candidates[name]=stats
    main=candidates[primary]; a=arrays[primary]; pred=a['score']>=.5
    traits=pd.read_csv(out/'image_traits.csv'); conditions={}
    assert traits.image_id.tolist()==frame.image_id.tolist(),'Image-trait row order mismatch'
    for col in ['brightness','contrast']:
        cuts=np.quantile(traits[col],[1/3,2/3]); bins=np.digitize(traits[col],cuts)
        conditions[col]={'cuts':cuts.tolist(),'bins':{label:summary(y[bins==i],pred[bins==i]) for i,label in enumerate(['low','medium','high'])}}
    modulo=angles%90
    masks={'near_maximum_rotation_padding':abs(modulo-45)<=10,'near_right_angle':np.minimum(modulo,90-modulo)<=10}
    counts=pd.Series(groups).value_counts(); duplicate=np.asarray([counts[g]>1 for g in groups])
    masks.update({'duplicate_group_rows':duplicate,'singleton_rows':~duplicate})
    conflict=pd.DataFrame({'group':groups,'label':y}).groupby('group').label.nunique()
    masks['conflicting_label_group_rows']=np.asarray([conflict[g]>1 for g in groups])
    for label,mask in masks.items(): conditions[label]=summary(y[mask],pred[mask])
    valid=check(args.submission,found['test'][2]); diagnostics=existing['evaluation_confidence']
    nested_diagnostics=json.loads((out/'nested_diagnostics.json').read_text())
    # No substitute composite score. Improvement must be credible on paired OOF,
    # and acceptable condition performance cannot be inferred from pooled BA.
    eligible=[n for n in names if n not in ['existing-metadata-logistic','angle-only-diagnostic','raw-hog-logistic','resnet-raw-logistic','constant-image-logistic']]
    credible=[n for n in eligible if candidates[n]['paired_difference_vs_existing_95'][0]>0]
    gate={'data_integrity_pass':True,'rotation_sign_verified':True,'cv_complete':True,'detected_group_leakage':False,
          'credible_improvements_over_existing':credible,
          'condition_reliability_demonstrated':False,
          'new_final_submission_promoted':False,
          'status':'NOT PASSED',
          'reason':'No new system is promoted automatically. Pooled CV is strongly azimuth-confounded, within-bin performance is near chance, and ensemble complexity requires a credible improvement. Existing CSV/checkpoint are preserved, not certified as scientifically reliable.'}
    if credible: gate['reason']='A candidate has a positive paired OOF improvement, but its condition performance and full inference must be reviewed before promotion.'
    feature_resources={p.stem:json.loads(p.read_text()) for p in out.glob('*_features.json')}
    classical_profiles=json.loads((out/'classical_inference_profiles.json').read_text())
    backbone_files={p.name:p.stat().st_size for p in (out/'torch_hub'/'checkpoints').glob('*.pth')}
    environment_audit=json.loads((out/'environment_audit.json').read_text()) if (out/'environment_audit.json').exists() else {'status':'not recorded'}
    payload={'primary_procedure':primary,'primary':main,'submission_structure':valid,'candidates':candidates,'data_fingerprint':data_audit['fingerprint'],
             'conditions':conditions,'nested_calibration_and_stress':nested_diagnostics,
             'evaluation_diagnostics':diagnostics,'inference_efficiency':existing['inference_efficiency'],
             'feature_extraction_resources':feature_resources,'bootstrap':existing['bootstrap'],
             'classical_inference_profiles':classical_profiles,'pretrained_checkpoint_bytes':backbone_files,
             'environment_audit':environment_audit,
             'selection_gate':gate,'hidden_test_accuracy':None,
             'historical_pre_grouping_experiments':json.loads(Path(args.legacy_metrics).read_text())['experiments'] if args.legacy_metrics else [],
             'limitations':['Intervals condition on fitted OOF models and exclude model selection, retraining and deployment shift.',
                            'Nested OOF evaluates the same ensemble recipe at smaller training size, not the exact final fitted checkpoint.',
                            'Old total end-to-end training time and training peak RAM were not instrumented and cannot be reconstructed honestly.',
                            'Benchmarks ran with concurrent CPU workloads. Sampled whole-system RAM includes unrelated processes.',
                            'Frozen pretrained feature experiments are not end-to-end fine-tuning. No GPU was available.']}
    (reports/'reliability_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    pd.DataFrame({'image_id':frame.image_id,'label':y,'group':groups,'fold':reference_fold,'probability_rise':a['probability'],'decision_score':a['score'],'prediction':pred.astype(int)}).to_csv(out/'oof_predictions.csv',index=False)
    shutil.copy2(out/'nested_calibration_plot.png',reports/'calibration_plot.png')
    def pc(x): return f'{100*x:.2f}%'
    def table(header,rows): return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
    lines=[f'Submission structure: {"PASS" if valid["valid"] else "FAIL"}',f'Estimated reliability: {pc(main["balanced_accuracy"])} balanced accuracy',
           f'95% confidence interval: {pc(main["confidence_interval_95"][0])}–{pc(main["confidence_interval_95"][1])}',
           f'Conservative reliability floor: {pc(main["confidence_interval_95"][0])}',f'Class 0 recall: {pc(main["recall_depth"])}',f'Class 1 recall: {pc(main["recall_rise"])}',
           f'Worst-fold balanced accuracy: {pc(main["worst_fold"])}',f'Worst azimuth-bin balanced accuracy: {pc(main["worst_azimuth_bin"])}',
           f'Ensemble unanimous agreement: {pc(diagnostics["unanimous_agreement"])}',f'Mean evaluation confidence: {pc(diagnostics["mean_selected_label_confidence"])}',
           'Actual hidden-test accuracy: Unknown until organizer evaluation']
    comparison=table(['Candidate','OOF BA','95% CI','Depth recall','Rise recall','Worst fold','Worst azimuth bin'],
                     [[n,pc(v['balanced_accuracy']),'–'.join(pc(x) for x in v['confidence_interval_95']),pc(v['recall_depth']),pc(v['recall_rise']),pc(v['worst_fold']),pc(v['worst_azimuth_bin'])] for n,v in candidates.items()])
    folding=table(['Fold','BA','Accuracy','Depth recall','Rise recall','Confusion matrix'],[[i,pc(v['balanced_accuracy']),pc(v['accuracy']),pc(v['recall_depth']),pc(v['recall_rise']),v['confusion_matrix']] for i,v in enumerate(main['per_fold'],1)])
    bins=table(['Azimuth','N','BA','Depth recall','Rise recall'],[[f"{v['lower']}-{v['lower']+45}",v['n'],pc(v['balanced_accuracy']),pc(v['recall_depth']),pc(v['recall_rise'])] for v in main['azimuth_bins']])
    condition_rows=[]
    for name,v in conditions.items():
        if 'bins' in v:
            for label,m in v['bins'].items(): condition_rows.append([name+' '+label,m['n'],pc(m['balanced_accuracy']),pc(m['recall_depth']),pc(m['recall_rise'])])
        else: condition_rows.append([name,v['n'],pc(v['balanced_accuracy']),pc(v['recall_depth']),pc(v['recall_rise'])])
    efficiencies=[]
    for name,v in measured.items():
        if 'components' in v: continue
        resource=v.get('resource_measurements',{})
        seconds=resource.get('seconds',v.get('total_fit_and_validation_seconds',v.get('seconds')))
        size=v.get('checkpoint_bytes') or ((old/f'{name}.joblib').stat().st_size if (old/f'{name}.joblib').exists() else v.get('all_outer_checkpoint_bytes'))
        head=v.get('cached_feature_inference_measurements',{})
        efficiencies.append([name,f'{seconds:.2f}' if seconds is not None else 'not measured',f'{size/1048576:.2f}' if size else 'see components',
                             f"{resource.get('peak_process_rss_bytes_sampled',v.get('peak_process_rss_bytes_sampled',0))/1048576:.1f}" if resource or 'peak_process_rss_bytes_sampled' in v else 'not measured',
                             f"{head['seconds']:.4f}" if head else 'not measured'])
    infer=existing['inference_efficiency']; cal=main['calibration']; uncal=nested_diagnostics['uncalibrated_sigmoid_margin']
    report=f'''# Reliability and best-output audit

## Decision

**Final selection gate: {gate['status']}.** {gate['reason']} No new final CSV was generated by this study. The existing 2,000-row submission and exact checkpoint remain unchanged. Do not confuse structural validity with demonstrated scientific reliability. The higher-scoring original metadata model is retained as a comparator, not endorsed as a robust shortcut.

## Required summary

```text
{chr(10).join(lines)}
```

The main estimate now comes from **nested five-fold ensemble OOF**, rather than the earlier single-held-out-member score of 70.93%. Each outer validation row and its detected duplicate group are unseen by all five ensemble members. Each member trains on 80% of the outer training partition; its calibration/threshold uses an additional training-side grouped holdout. Final deployment members use 80% of the full training data, so this nested estimate uses a smaller training size. It remains a model-selection estimate, not a measurement of hidden-test accuracy.

## Uncertainty and bias

10,000 paired cluster-bootstrap resamples, seed 2026, stratify duplicate groups into pure-depth, pure-rise and mixed-label groups. Resampling retains all rows in a group together; group counts per stratum stay fixed while class row counts may vary. The 2.5th and 97.5th percentiles form the interval. The requested conservative floor is only this interval's lower bound, **not a guaranteed deployment floor**. It excludes refitting, model-selection uncertainty, unknown related scenes and lighting-distribution shift. All evaluated candidates, including failed baselines and fixed equal-weight ensembles, are retained below; choosing their maximum is biased.

## Classification metrics

Ordinary OOF accuracy: {pc(main['accuracy'])}. Mean fold BA: {pc(main['fold_mean'])}; sample standard deviation: {100*main['fold_std_sample']:.3f} percentage points. Recall gap: {100*main['recall_gap']:.2f} percentage points. Confusion matrix (true rows / predicted columns, 0 then 1): `{main['confusion_matrix']}`. False positives (Depth called Rise): {main['false_positive_rise']}; false negatives (Rise called Depth): {main['false_negative_rise']}.

{folding}

The primary decision rule averages member probabilities after a log-odds shift by each training-only threshold and uses 0.5. This is a decision score, not a calibrated probability. Raw calibrated-probability soft voting at the default 0.5 yields {pc(main['default_probability_05']['balanced_accuracy'])} BA. Both rules were tested without optimizing any threshold on outer OOF labels. Any advantage from choosing a voting rule on these results must still be treated as model-selection bias. An ensemble is not promoted merely because a voting rule is better than another ensemble rule.

## All matched-protocol comparisons

{comparison}

The JSON includes paired improvement intervals against the original metadata model and the single SVM, plus component comparisons for architecture ensembles. No invented composite reliability score is used. The original historical 75.38% random-stratified result used pooled-OOF threshold selection and is kept separately; its stricter matched-protocol score is shown above. Candidate folds, feature versions and group identities are shared and explicitly checked. Frozen ImageNet models do not train on competition validation labels; they are feature extractors with learned sklearn heads, not fine-tuned CNNs.

## Reliability by condition

{bins}

Macro average of within-azimuth-bin BA: {pc(main['macro_azimuth_bin_ba'])}. The contrast between pooled BA and near-chance within-bin BA is consistent with strong azimuth-related confounding. Solar normalization alone does not eliminate that shortcut, because rotated scene orientation and original black regions can still reveal the angle. This is not evidence of dependable physical morphology recognition.

{table(['Condition','N','BA','Depth recall','Rise recall'],condition_rows)}

Brightness is mean grayscale intensity and contrast is grayscale standard deviation on [0,1]. Low/medium/high use full-training empirical tertiles for descriptive slicing only, not model fitting. Rotation-boundary proxies are within 10 degrees of 45 modulo 90 (maximal new corner support) or of a multiple of 90. These are geometry proxies, not manually annotated object-boundary locations. No confirmed cross-fold leakage remains among detected groups; unrecognized scene relationships cannot be ruled out.

The raw HOG / constant-fill / reflect logistic comparisons isolate normalization and border handling at the same regularization and splits. The raw-versus-normalized ResNet comparison supplies a separate pretrained-image ablation. The rotation sign was already correct; the repair changes border handling, not the sign. Neither normalization nor reflect padding is claimed to improve robustness merely because its pooled score rises.

## Noise and interpolation sensitivity

Only separate requested diagnostic copies were perturbed; original validation/test inputs and submission predictions were never augmented. Deterministic Gaussian noise uses sigma=1 grayscale level, and the alternate rotation uses bilinear instead of bicubic interpolation. Each diagnostic is predicted only by its outer-held-out ensemble. Results and flip fractions:

{table(['Diagnostic','BA','Prediction flip rate'],[[n,pc(v['balanced_accuracy']),pc(v['flip_fraction_vs_original'])] for n,v in nested_diagnostics.items() if 'flip_fraction_vs_original' in v])}

## Probability calibration

Nested OOF Platt-calibrated ensemble: Brier {cal['brier']:.6f}, log loss {cal['log_loss']:.6f}, ECE {cal['ece_15_equal_width']:.6f}. Uncalibrated sigmoid-margin ensemble: Brier {uncal['brier']:.6f}, log loss {uncal['log_loss']:.6f}, ECE {uncal['ece_15_equal_width']:.6f}. ECE uses 15 equal-width bins of positive-class probability, weighted by each bin's sample count. A sigmoid-transformed SVM margin is an explicit uncalibrated comparator, not an inherently probabilistic SVM output. Calibration is fit only on training-side held-out scores. Classification thresholds and calibration must be judged separately; better Brier score does not establish better balanced accuracy or calibration after distribution shift.

Default-0.5 uncalibrated sigmoid-margin soft voting gives {pc(nested_diagnostics['uncalibrated_soft_vote_05']['balanced_accuracy'])} BA, versus {pc(main['default_probability_05']['balanced_accuracy'])} for calibrated soft voting. Thus better calibration does not preserve default-cutoff BA here. The threshold-adjusted deployed rule scores {pc(main['balanced_accuracy'])}, but this does not isolate calibration's causal benefit. The checkpoint is preserved for audit, not newly promoted under the calibration/selection requirement.

![Nested held-out calibration](calibration_plot.png)

## Evaluation confidence and agreement (not accuracy)

Mean confidence in the submitted label: {pc(diagnostics['mean_selected_label_confidence'])}; median {pc(diagnostics['median_selected_label_confidence'])}. At least 70%: {pc(diagnostics['confidence_ge_70'])}; at least 80%: {pc(diagnostics['confidence_ge_80'])}; at least 90%: {pc(diagnostics['confidence_ge_90'])}. There are {diagnostics['near_threshold_abs_score_within_005']} predictions within 0.05 of the ensemble decision cutoff. Confidence means p(Rise) for submitted Rise and 1-p(Rise) for submitted Depth; it is not always max(p,1-p), because balanced-accuracy thresholds need not be 0.5. Mean max-class probability, a different quantity, is {pc(diagnostics['mean_max_class_probability'])}.

All five members agree on {pc(diagnostics['unanimous_agreement'])}; at least four agree on {pc(diagnostics['supermajority_agreement_at_least_4_of_5'])}. At least three agree on {pc(diagnostics['majority_agreement_at_least_3_of_5'])}, which is automatic for five binary voters and is not evidence of reliability. Average probability standard deviation: {diagnostics['average_probability_std']:.6f}. Changing the deployed decision rule to raw probability soft voting at 0.5 would change {diagnostics['raw_probability_soft_vote_disagrees_with_current_labels']} labels; it was not applied to the submission.

Most disputed IDs: {', '.join(diagnostics['most_disputed'])}.

## Efficiency and resource measurements

Existing checkpoint end-to-end inference benchmark for all 2,000 images: {infer['seconds']:.2f} seconds, {infer['images_per_second']:.2f} images/second; model size {infer['checkpoint_bytes']/1048576:.2f} MiB. Timing covers model loading, preprocessing, prediction and a feature-cache write, not process startup or final CSV writing. Sampled peak process RSS: {infer['peak_process_rss_bytes_sampled']/1048576:.2f} MiB; sampled whole-system used RAM: {infer['peak_system_used_ram_bytes_sampled']/1073741824:.2f} GiB. CUDA was unavailable, so GPU memory is 0 and mixed precision is not applicable.

The kernel model stores {infer['stored_support_vector_values']:,} support-vector values and {infer['learned_dual_coefficients_and_intercepts']:,} fitted dual/intercept coefficients, plus ten sigmoid-calibration coefficients across five members. This is not a neural-network trainable-parameter count. Frozen backbone and fitted head parameter counts are recorded in the JSON for transfer candidates. Feature caches and checkpoint reuse avoid repeated image computation. Sklearn estimators converge without epoch scheduling/early stopping; no unperformed neural training, mixed precision or fine-tuning is claimed.

{table(['Candidate','Fit + validation seconds*','Checkpoint MiB**','Sampled process peak MiB','Cached-head inference seconds'],efficiencies)}

* New candidates include five-fold CV plus four held-out-sector checks. Historical entries contain only their recorded five-fold CV timer. Nested ensemble entries include all 25 member fits and their internal calibration fits. These are not directly comparable total pipeline times. Historical full training wall time and training peak RAM were not instrumented; they are explicitly unavailable rather than invented. ** Nested size is all 25 research-validation checkpoints, not the deployed five-member file; CNN entries are heads only and require the shared pretrained weights below. Cached-head latency is not end-to-end image latency.

{table(['Deterministic feature pass','Images','Seconds','Images/second','Process peak MiB'],[[n,v['images'],f"{v['seconds']:.2f}",f"{v['images_per_second']:.2f}",f"{v['peak_process_rss_bytes_sampled']/1048576:.2f}"] for n,v in feature_resources.items() if 'images' in v])}

{table(['Classical model','Staged 2,000-image seconds','Images/second','Fitted coefficients incl. calibration'],[[n,f"{v['staged_total_seconds']:.2f}",f"{v['staged_images_per_second']:.2f}",v['fitted_coefficients_including_calibration']] for n,v in classical_profiles.items()])}

These staged classical timings add separately measured preprocessing and checkpoint load/prediction. They exclude CSV export. Pretrained backbone files required in addition to learned CNN heads: {', '.join(f'{n}: {v/1048576:.2f} MiB' for n,v in backbone_files.items())}. Backbone parameters are frozen (zero trainable backbone parameters); linear heads learn feature-dimension + 1 coefficients per member, plus two calibration coefficients. Kernel heads store support vectors and dual coefficients rather than a neural parameter count. Fixed architecture ensembles reuse these component resources; their complete joint pipeline was not separately benchmarked.

The dedicated 2,000-image passes measure feature-extraction latency without classifier fitting. Add the recorded head inference stage to estimate the corresponding staged pipeline duration; model construction, startup and CSV export are excluded. Timers were collected while other experiments could run, so they are device-session measurements, not controlled hardware benchmarks. Peak system RAM includes other applications and is sampled every 0.1 second rather than a guaranteed instantaneous maximum. No measured value is substituted for an unrecorded historical measurement.

The resumed ResNet/SVM run combines its saved CV duration with newly timed sector checks. Its saved peak memory covers only the resumed stage, not the interrupted earlier CV stage. This narrower coverage is also marked in the JSON. Raw/normalized CNN timing differences under concurrent load should not be interpreted as intrinsic architecture speed differences.

Official architecture/preprocessing references: [ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and [MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html). Runs used torch 2.12.0+cpu and torchvision 0.27.0+cpu with verified ImageNet V1 weights and deterministic 224-pixel center crops after required normalization.

The core competition training/inference environment was isolated and its nine tests pass. The extended research environment reused system-site CPU PyTorch and was not clean-isolated. Its dependency check reported unrelated inherited arxiv/requests, pyiceberg/rich and realtime/websockets conflicts, detailed in the JSON. These packages are not used by the competition scripts, and the research runs completed, but a clean extended-environment reproduction has not been demonstrated. No global dependencies were changed to hide these conflicts.

## Deliverables and remaining qualification

`reliability_metrics.json` contains measured candidate metrics, confidence intervals, conditions, resource details and the selection gate. The primary saved OOF table is nested ensemble OOF; evaluation probabilities come from the unchanged actual inference checkpoint. Structural validity is independently checked against test metadata. Actual hidden-test accuracy remains unknown until organizer evaluation. The existing public weights action remains manual at the user's request. No form or LinkedIn post was submitted.
'''
    (reports/'reliability_report.md').write_text(report,encoding='utf-8')
    print('\n'.join(lines)); print('FINAL_SELECTION_GATE',json.dumps(gate))


if __name__=='__main__': main()
