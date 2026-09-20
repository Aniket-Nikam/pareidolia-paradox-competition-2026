"""Extract only competition CSV/PNG members from official ZIPs, including wrappers."""
import argparse
import io
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sources',nargs='+',required=True,help='Official ZIP archives and/or metadata CSV files')
    p.add_argument('--destination',default='data')
    args=p.parse_args(); out=Path(args.destination).resolve()
    if out.exists(): raise ValueError('Use a new destination directory to avoid overwriting existing data')
    sources=[Path(s).resolve() for s in args.sources]
    if any(not s.is_file() or s.suffix.lower() not in ['.zip','.csv'] for s in sources):
        raise ValueError('Every source must be an existing ZIP or CSV')
    out.mkdir(parents=True)
    seen=set(); byte_count=0
    def write(name,contents):
        nonlocal byte_count
        if name in seen: raise ValueError(f'Duplicate competition filename: {name}')
        seen.add(name); byte_count+=len(contents)
        if byte_count>3_000_000_000 or len(seen)>9856: raise ValueError('Unexpected dataset size')
        if name in ['train_metadata.csv','test_metadata.csv']: target=out/name
        elif re.fullmatch(r'(train|eval)_\d{5}\.png',name):
            target=out/('train_images' if name.startswith('train_') else 'eval_images')/name
        else: return
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f: f.write(contents)
    def extract(source,depth=0):
        if depth>2: raise ValueError('Unexpected archive nesting')
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                member=PurePosixPath(item.filename.replace('\\','/'))
                if member.is_absolute() or '..' in member.parts or stat.S_ISLNK(item.external_attr>>16):
                    raise ValueError('Unsafe archive member')
                if item.is_dir(): continue
                name=member.name
                if name in ['train_images.zip','eval_images.zip','test_images.zip']:
                    if item.file_size>1_500_000_000: raise ValueError('Unexpected nested archive size')
                    extract(io.BytesIO(archive.read(item)),depth+1)
                elif name in ['train_metadata.csv','test_metadata.csv'] or re.fullmatch(r'(train|eval)_\d{5}\.png',name):
                    if item.file_size>5_000_000: raise ValueError('Unexpected image/CSV size')
                    write(name,archive.read(item))
    for source in sources:
        if source.suffix.lower()=='.zip': extract(source)
        elif source.name in ['train_metadata.csv','test_metadata.csv']: write(source.name,source.read_bytes())
        else: raise ValueError('Unexpected CSV filename')
    from dataset import locate
    found=locate(out)
    print(f"Validated {len(found['train'][0])} train and {len(found['test'][0])} evaluation records in {out}")


if __name__=='__main__': main()
