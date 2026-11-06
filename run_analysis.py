"""Run the unchanged statistical source with portable, isolated paths."""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load_stats(input_csv, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    tree = ast.parse((ROOT / 'STATS.py').read_text(encoding='utf-8-sig'))
    replacements = {'INPUT_CSV': str(input_csv), 'OUTPUT_DIR': str(output_dir)}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.Constant(replacements[target.id])
    ast.fix_missing_locations(tree)
    namespace = {'__name__': 'portable_stats', '__file__': str(ROOT / 'STATS.py')}
    exec(compile(tree, str(ROOT / 'STATS.py'), 'exec'), namespace)
    return namespace

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'FINAL_EXP_RESULTS.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced_analysis')
    args = parser.parse_args()
    inp, out = args.input.resolve(), args.output_dir.resolve()
    if out == ROOT:
        parser.error('Use a separate output directory to preserve archived results.')
    ns = load_stats(inp, out)
    ns['main']()

if __name__ == '__main__':
    main()
