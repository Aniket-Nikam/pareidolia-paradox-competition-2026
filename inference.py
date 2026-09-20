"""Load the exact fold ensemble and create a metadata-ordered CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from dataset import locate
from train import feature_matrix, member_predict
from validate_submission import check


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True)
    p.add_argument('--checkpoint',default='artifacts/pareidolia_final_model.joblib')
    p.add_argument('--output',default='output/submission.csv')
    p.add_argument('--artifacts',default='artifacts')
    args=p.parse_args(); checkpoint=Path(args.checkpoint); model=joblib.load(checkpoint)
    if model['feature_version']!='reflect-hog-pixels-v2': raise ValueError('Unsupported feature version')
    found=locate(args.data_root); frame,paths,metadata=found['test']
    x=feature_matrix(frame,paths,model['feature_config']['border_mode'])
    with threadpool_limits(limits=model['config']['threads']):
        results=[member_predict(member,x) for member in model['members']]
    probabilities=np.mean([v[0] for v in results],axis=0)
    score=np.mean([v[1] for v in results],axis=0)
    labels=(score>=model['final_cutoff']).astype(np.int64)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'image_id':frame.image_id,'label':labels}).to_csv(out,index=False)
    artifacts=Path(args.artifacts); artifacts.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'image_id':frame.image_id,'probability_rise':probabilities,'decision_score':score,'label':labels}).to_csv(artifacts/'test_predictions.csv',index=False)
    report=check(out,metadata)
    report.update({'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'model':model['name'],'members':len(model['members'])})
    (artifacts/'inference_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
