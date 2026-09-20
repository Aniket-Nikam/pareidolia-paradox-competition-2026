"""Decode every image, check source integrity and conservatively group duplicates."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.fft import dctn
from scipy.stats import ks_2samp
from dataset import locate, fingerprint
from ml.preprocessing import normalize_solar_azimuth


def inspect_image(item):
    path, angle = item
    with Image.open(path) as image:
        image.load()
        a = np.asarray(image)
        grayscale = image.convert('L')
        normalized = normalize_solar_azimuth(grayscale, angle, border_mode='reflect')
        thumb = np.asarray(normalized.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        rawthumb = np.asarray(grayscale.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        hashes = []
        for values in [rawthumb, thumb]:
            coeff = dctn(values, norm='ortho')[:8, :8].ravel()[1:]
            hashes.append(int(''.join('1' if v > np.median(coeff) else '0' for v in coeff), 2))
        return {'size': list(image.size), 'mode': image.mode, 'dtype': str(a.dtype),
                'min': int(a.min()), 'max': int(a.max()), 'std': float(a.std()),
                'equal_rgb': bool(a.ndim == 2 or (a.ndim == 3 and np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]))),
                'sha256_pixels': hashlib.sha256(a.tobytes()).hexdigest(),
                'phash': hashes, 'thumbs': [rawthumb, thumb]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--artifacts', default='artifacts')
    args = parser.parse_args()
    out = Path(args.artifacts); out.mkdir(parents=True, exist_ok=True)
    found = locate(args.data_root)
    joined = pd.concat([found['train'][0].assign(split='train'), found['test'][0].assign(split='test')], ignore_index=True)
    paths = found['train'][1] + found['test'][1]
    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, record in enumerate(pool.map(inspect_image, zip(paths, joined.sun_azimuth_angle)), 1):
            records.append(record)
            if index % 1000 == 0: print(f'decoded {index}/{len(paths)}', flush=True)
    assert all(r['size'] == [256, 256] and r['dtype'] == 'uint8' for r in records)
    parent = list(range(len(paths)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        parent[find(j)] = find(i)
    exact = defaultdict(list)
    for i, r in enumerate(records): exact[r['sha256_pixels']].append(i)
    pairs = []
    for members in exact.values():
        for j in members[1:]:
            union(members[0], j); pairs.append((members[0], j, 'exact_pixels', 0.0))
    # Five hash partitions guarantee a shared partition when <=4 bits differ.
    # Both raw and solar-normalized pHashes are checked; RMSE rejects false matches.
    for view in range(2):
        buckets = defaultdict(list)
        for i, r in enumerate(records):
            code = r['phash'][view]
            candidates = set()
            for part in range(5): candidates.update(buckets[(part, (code >> (part * 13)) & 8191)])
            for j in candidates:
                if (code ^ records[j]['phash'][view]).bit_count() <= 4:
                    rmse = float(np.sqrt(np.mean((r['thumbs'][view] - records[j]['thumbs'][view]) ** 2)))
                    if rmse <= .04 and find(i) != find(j):
                        union(i, j); pairs.append((i, j, ['raw_near', 'normalized_near'][view], rmse))
            for part in range(5): buckets[(part, (code >> (part * 13)) & 8191)].append(i)
    joined['group'] = [find(i) for i in range(len(paths))]
    joined[['image_id', 'split', 'group']].to_csv(out/'duplicate_groups.csv', index=False)
    pair_rows = [{'a': joined.image_id.iloc[i], 'b': joined.image_id.iloc[j], 'kind': kind, 'rmse': rmse} for i,j,kind,rmse in pairs]
    pd.DataFrame(pair_rows, columns=['a','b','kind','rmse']).to_csv(out/'duplicate_pairs.csv', index=False)
    bins = []
    for lower in range(0, 360, 45):
        train = joined[(joined.split=='train') & joined.sun_azimuth_angle.between(lower, lower+45, inclusive='left')]
        test = joined[(joined.split=='test') & joined.sun_azimuth_angle.between(lower, lower+45, inclusive='left')]
        bins.append({'start': lower, 'end':lower+45, 'train_n':len(train), 'test_n':len(test),
                     'depth':int((train.label==0).sum()), 'rise':int((train.label==1).sum()), 'rise_fraction':float(train.label.mean()) if len(train) else None})
    conflict = joined[joined.split=='train'].groupby('group').label.nunique()
    mixed = joined.groupby('group').split.nunique()
    report = {'fingerprint':fingerprint(found), 'decoded_count':len(records), 'train_count':7854, 'test_count':2000,
              'class_counts':found['train'][0].label.value_counts().sort_index().to_dict(),
              'modes':dict(Counter(r['mode'] for r in records)), 'dtype':'uint8', 'dimensions':[256,256],
              'global_intensity_range':[min(r['min'] for r in records),max(r['max'] for r in records)],
              'equal_rgb_count':sum(r['equal_rgb'] for r in records),
              'constant_images':[joined.image_id.iloc[i] for i,r in enumerate(records) if r['std']==0],
              'exact_duplicate_extra_images':sum(len(v)-1 for v in exact.values()),
              'near_duplicate_connections':sum(p[2]!='exact_pixels' for p in pairs),
              'conflicting_train_groups':int((conflict>1).sum()), 'train_test_shared_groups':int((mixed>1).sum()),
              'azimuth_bins':bins, 'azimuth_ks':float(ks_2samp(found['train'][0].sun_azimuth_angle, found['test'][0].sun_azimuth_angle).statistic),
              'duplicate_method':'Exact decoded-pixel SHA256; raw and normalized 63-bit DCT pHash Hamming<=4 plus 32x32 RMSE<=0.04, connected groups. Conservative heuristic, not scene identity proof.'}
    (out/'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    canvas=Image.new('RGB',(3*256,6*290),(24,24,24)); draw=ImageDraw.Draw(canvas)
    for row,index in enumerate([0, 500, 1500, 3000, 5000, 7000]):
        with Image.open(paths[index]) as im:
            source=im.convert('L'); angle=float(joined.sun_azimuth_angle.iloc[index])
            for col,(label,image) in enumerate([('Original',source),('Black fill',normalize_solar_azimuth(source,angle)),('Reflect',normalize_solar_azimuth(source,angle,border_mode='reflect'))]):
                canvas.paste(image,(col*256,row*290)); draw.text((col*256+4,row*290+260),f'{label} {joined.image_id.iloc[index]} {angle:.1f}',fill='white')
    canvas.save(out/'normalization_comparison.jpg',quality=92)
    print(json.dumps({k:v for k,v in report.items() if k!='azimuth_bins'},indent=2),flush=True)


if __name__=='__main__': main()
