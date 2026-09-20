"""Strict metadata matching and portable dataset discovery."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def locate(data_root):
    root = Path(data_root).resolve()
    found = {}
    for split, filename, count, prefix in [('train', 'train_metadata.csv', 7854, 'train_'), ('test', 'test_metadata.csv', 2000, 'eval_')]:
        candidates = sorted(root.rglob(filename))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one {filename} below {root}, found {len(candidates)}')
        frame = pd.read_csv(candidates[0])
        expected = ['image_id', 'sun_azimuth_angle'] + (['label'] if split == 'train' else [])
        if list(frame.columns) != expected or len(frame) != count:
            raise ValueError(f'{filename}: expected columns {expected} and {count} rows')
        if frame.isna().any().any() or frame.image_id.duplicated().any():
            raise ValueError(f'{filename}: null or duplicate metadata')
        if not frame.image_id.str.fullmatch(prefix + r'\d{5}\.png').all():
            raise ValueError('Unsafe or unexpected image ID')
        if not np.isfinite(frame.sun_azimuth_angle).all() or not frame.sun_azimuth_angle.between(0, 360).all():
            raise ValueError('Invalid azimuth')
        if split == 'train' and (not pd.api.types.is_integer_dtype(frame.label) or not frame.label.isin([0, 1]).all()):
            raise ValueError('Invalid labels')
        files = list(root.rglob(prefix + '*.png'))
        by_name = {}
        for p in files:
            if p.name in by_name:
                raise ValueError(f'Multiple image files for {p.name}')
            by_name[p.name] = p
        if set(by_name) != set(frame.image_id):
            raise ValueError(f'{split}: image IDs differ from metadata')
        found[split] = (frame, [by_name[name] for name in frame.image_id], candidates[0])
    return found


def fingerprint(found):
    digest = hashlib.sha256()
    for split in ['train', 'test']:
        frame, paths, csv = found[split]
        digest.update(csv.read_bytes())
        for path in paths:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_config(path='config.yaml'):
    # JSON is a YAML 1.2 subset; no additional YAML parser is required.
    return json.loads(Path(path).read_text(encoding='utf-8'))
