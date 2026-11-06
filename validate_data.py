"""Validate the full factorial output and archived checksums without dependencies."""
import csv
import hashlib
import itertools
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    with (ROOT / 'FINAL_EXP_RESULTS.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fields = ['file', 'environment', 'dimension', 'model', 'seed']
    keys = [tuple(row[k] for k in fields) for row in rows]
    files = sorted({r['file'] for r in rows})
    expected = set(itertools.product(files,
        ['severe_drought', 'moderate_drought', 'normal', 'wet', 'very_wet'],
        ['2', '16', '32'], ['Poincare', 'MDS', 'Spectral'],
        ['42', '123', '456', '789', '1011']))
    assert len(files) == 78 and len(rows) == 17550
    assert len(set(keys)) == len(keys) and set(keys) == expected
    metrics = ['MRE','Raw_MRE','Stress','Hierarchy','Physical_Hydraulic',
        'Hydraulic','Hydraulic_Preservation','Embedding_Transport_Preservation',
        'Nutrient_Uptake','Nutrient_Final','Nutrient_Flux']
    assert all(math.isfinite(float(r[m])) for r in rows for m in metrics)
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    print('PASS: 17,550 unique complete factorial rows, 78 filenames, 11 finite metrics, and all archived file checksums.')

if __name__ == '__main__':
    main()
