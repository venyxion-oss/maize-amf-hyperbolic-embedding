"""Compare our regenerated statistical tables with the archived outputs."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

def verify(output_dir):
    tables = sorted(ROOT.glob('[0-9][0-9]_*.csv'))
    if len(tables) != 12:
        raise ValueError('The archive must contain all 12 numbered statistical tables.')
    for source in tables:
        target = output_dir/source.name
        if not target.is_file():
            raise FileNotFoundError(f'Missing regenerated table: {target}. Run run_analysis.py first.')
        pd.testing.assert_frame_equal(pd.read_csv(source), pd.read_csv(target),
                                      check_exact=False, rtol=1e-9, atol=1e-12)
        print(f'PASS: {source.name}')
    return len(tables)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reproduced_analysis')
    args = parser.parse_args()
    try:
        count = verify(args.output_dir.resolve())
    except (AssertionError, ValueError, OSError) as error:
        parser.exit(1, f'Reproduction check failed: {error}\n')
    print(f'All {count} statistical tables match within rtol=1e-9 and atol=1e-12.')

if __name__ == '__main__':
    main()
