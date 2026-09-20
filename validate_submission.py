"""Validate the upload artifact, including exact metadata row order."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from ml.submission import validate_submission


def check(path,metadata):
    raw=Path(path).read_text(encoding='utf-8')
    if raw.splitlines()[0]!='image_id,label': raise ValueError('Invalid CSV header')
    frame=pd.read_csv(path); expected=pd.read_csv(metadata)
    if len(expected)!=2000: raise ValueError('Evaluation metadata must have exactly 2000 rows')
    validate_submission(frame,expected_rows=2000,expected_ids=expected.image_id.tolist())
    return {'valid':True,'rows':len(frame),'columns':list(frame.columns),'exact_id_order':True,
            'label_distribution':frame.label.value_counts().sort_index().to_dict(),
            'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--submission',required=True); p.add_argument('--metadata',required=True); p.add_argument('--report')
    args=p.parse_args(); result=check(args.submission,args.metadata)
    print(json.dumps(result,indent=2)); frame=pd.read_csv(args.submission)
    print('FIRST FIVE\n'+frame.head().to_string(index=False)); print('LAST FIVE\n'+frame.tail().to_string(index=False))
    if args.report: Path(args.report).write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__': main()
