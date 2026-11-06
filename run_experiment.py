"""Run EXP.py with explicit input paths and a separate results CSV."""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'rerun_results.csv')
    p.add_argument('--max-files', type=int, default=None)
    args = p.parse_args()
    if args.output.resolve() == (ROOT / 'FINAL_EXP_RESULTS.csv').resolve():
        p.error('The archived results CSV must not be used as a rerun destination.')
    if not args.dataset_dir.is_dir():
        p.error('Dataset directory does not exist.')
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('original_experiment', ROOT / 'EXP.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_full_experiment_suite(
        dataset_dir=str(args.dataset_dir.resolve()),
        amf_nodes_csv=str(ROOT / 'AMF_network_nodes.csv'),
        amf_edges_csv=str(ROOT / 'AMF_network_edges.csv'),
        output_path=str(args.output.resolve()), max_files=args.max_files,
        clear_output=False,
    )

if __name__ == '__main__':
    main()
