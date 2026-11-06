"""Validate our archived results and optional external root inputs."""
import argparse
import csv
import hashlib
import itertools
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ['MRE','Raw_MRE','Stress','Hierarchy','Physical_Hydraulic','Hydraulic',
           'Hydraulic_Preservation','Embedding_Transport_Preservation',
           'Nutrient_Uptake','Nutrient_Final','Nutrient_Flux']
SCENARIOS = dict(zip(['severe_drought','moderate_drought','normal','wet','very_wet'],
                    [0.2,0.4,0.6,0.8,1.0]))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def validate(root=ROOT, rsml_dir=None):
    rows = read_rows(root/'FINAL_EXP_RESULTS.csv')
    require(bool(rows), 'The results CSV is empty.')
    fields = ['file','environment','dimension','model','seed']
    require(set(fields+METRICS+['theta']).issubset(rows[0]), 'Required columns are missing.')
    keys = [tuple(row[k] for k in fields) for row in rows]
    files = sorted({r['file'] for r in rows})
    expected = set(itertools.product(files, SCENARIOS, ['2','16','32'],
                   ['Poincare','MDS','Spectral'], ['42','123','456','789','1011']))
    require(len(files)==78 and len(rows)==17550, 'Expected 78 filenames and 17,550 model runs.')
    require(len(set(keys))==len(keys), 'Duplicate model-run keys were found.')
    require(set(keys)==expected, 'Missing or unexpected factorial combinations were found.')
    require(all(math.isfinite(float(r[m])) for r in rows for m in METRICS), 'A metric is nonfinite.')
    require(all(math.isclose(float(r['theta']),SCENARIOS[r['environment']],abs_tol=1e-12)
                for r in rows), 'Scenario labels and theta values do not agree.')
    require(len(read_rows(root/'02_file_level_seed_averaged_data.csv'))==3510,
            'Expected 3,510 seed-averaged records.')
    for line in (root/'SHA256SUMS.txt').read_text(encoding='utf8').splitlines():
        digest,name=line.split('  ',1)
        target=(root/name).resolve()
        require(target.is_relative_to(root.resolve()), 'Invalid checksum path.')
        require(target.is_file(), f'Missing archive file: {name}')
        require(hashlib.sha256(target.read_bytes()).hexdigest()==digest, f'Checksum mismatch: {name}')
    if rsml_dir is not None:
        manifest=read_rows(root/'rsml_input_manifest.csv')
        require({r['file'] for r in manifest}==set(files), 'Root manifest and filenames differ.')
        for item in manifest:
            target=(rsml_dir/item['file']).resolve()
            require(target.is_relative_to(rsml_dir.resolve()), 'Invalid root input path.')
            require(target.is_file(), f'Missing root input: {item["file"]}')
            require(hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256'],
                    f'Root input checksum mismatch: {item["file"]}')
    return len(rows),len(files)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rsml-dir',type=Path,help='Optional directory containing the original 78 root files.')
    args=parser.parse_args()
    try:
        runs,files=validate(rsml_dir=args.rsml_dir)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(1,f'Validation failed: {error}\n')
    print(f'PASS: {runs:,} model runs; {files} root files; 3,510 seed-averaged records; archive checksums verified.')
    if args.rsml_dir is not None:
        print('PASS: All 78 external root input checksums match the manifest.')

if __name__=='__main__':
    main()
