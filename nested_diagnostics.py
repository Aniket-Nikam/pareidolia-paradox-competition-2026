"""Calibration and perturbation diagnostics for genuinely held-out ensembles."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from dataset import locate
from train import member_predict
from reliability_common import summary,calibration


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--previous',required=True); p.add_argument('--artifacts',required=True); args=p.parse_args()
    old=Path(args.previous); out=Path(args.artifacts); y=locate(args.data_root)['train'][0].label.to_numpy()
    original=np.load(out/'nested-fivefold-svm_oof.npz'); fold=original['fold']
    matrices={'original':np.load(old/'reflect_features.npz')['x'],
              'gaussian_noise_sigma_1_graylevel':np.load(out/'stress_gaussian_noise_sigma_1_graylevel.npz')['x'],
              'bilinear_rotation':np.load(out/'stress_bilinear_rotation.npz')['x']}
    outputs={name:{'p':np.zeros(len(y)),'score':np.zeros(len(y))} for name in matrices}; uncal=np.zeros(len(y))
    with threadpool_limits(limits=2):
        for f in range(1,6):
            members=joblib.load(out/f'nested_ensemble_outer_{f}.joblib')['members']; ix=fold==f
            for name,x in matrices.items():
                predictions=[member_predict(m,x[ix]) for m in members]
                outputs[name]['p'][ix]=np.mean([v[0] for v in predictions],axis=0)
                outputs[name]['score'][ix]=np.mean([v[1] for v in predictions],axis=0)
            uncal[ix]=np.mean([expit(m['model'].decision_function(matrices['original'][ix])) for m in members],axis=0)
            print('nested diagnostics fold',f,'complete',flush=True)
    assert np.allclose(outputs['original']['p'],original['probability'],rtol=0,atol=1e-12)
    report={name:{**summary(y,v['score']>=.5,v['p']),
                  'flip_fraction_vs_original':float(np.mean((v['score']>=.5)!=(original['score']>=.5)))} for name,v in outputs.items()}
    report['uncalibrated_sigmoid_margin']=calibration(y,uncal)
    report['uncalibrated_soft_vote_05']=summary(y,uncal>=.5)
    (out/'nested_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],'k--',label='Ideal')
    for label,metrics in [('Platt-calibrated ensemble',report['original']['calibration']),('Sigmoid-margin ensemble',report['uncalibrated_sigmoid_margin'])]:
        rows=metrics['bins']; ax.plot([r['mean_probability'] for r in rows],[r['observed_rise_fraction'] for r in rows],'o-',label=label)
    ax.set(xlabel='Predicted probability of Rise',ylabel='Observed Rise fraction',xlim=(0,1),ylim=(0,1),title='Nested held-out ensemble calibration')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'nested_calibration_plot.png',dpi=160); plt.close(fig)
    print('NESTED_DIAGNOSTICS',json.dumps({k:{m:v for m,v in val.items() if m!='calibration'} for k,val in report.items() if k in outputs}),flush=True)


if __name__=='__main__': main()
