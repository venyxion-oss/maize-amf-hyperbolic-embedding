"""Exact MDS-output-signature sensitivity analysis described in the manuscript."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from run_analysis import ROOT, load_stats

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'FINAL_EXP_RESULTS.csv')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'reproduced_sensitivity')
    args = p.parse_args()
    out = args.output_dir.resolve()
    if out == ROOT:
        p.error('Use a separate output directory.')
    ns = load_stats(args.input.resolve(), out)
    pd, metrics = ns['pd'], ns['METRICS']
    df = ns['load_data'](str(args.input.resolve()))
    groups = {}
    for filename, group in df[df.model == 'MDS'].groupby('file'):
        values = group.sort_values(['environment','dimension','model','seed'])[metrics].to_numpy()
        signature = hashlib.sha256(values.tobytes()).hexdigest()
        groups.setdefault(signature, []).append(filename)
    cluster_for = {name: names[0] for names in groups.values() for name in names}
    averaged = df.groupby(['file','environment','theta','dimension','model'])[metrics].mean().reset_index()
    averaged['file'] = averaged['file'].map(cluster_for)
    averaged = averaged.groupby(['file','environment','theta','dimension','model'])[metrics].mean().reset_index()
    averaged.to_csv(out / 'cluster_seed_averaged_data.csv', index=False)
    calls = {'models':'run_friedman_model_tests','pairwise':'pairwise_model_tests',
             'dimensions':'dimension_comparison','environments':'environment_comparison'}
    with contextlib.redirect_stdout(io.StringIO()):
        results = {k:ns[v](averaged) for k,v in calls.items()}
    summary = {
        'signature_rule':'Exact MDS arrays across eleven metrics, five scenarios, three dimensions and five seeds.',
        'distinct_signatures':len(groups),
        'duplicate_groups':[g for g in groups.values() if len(g)>1],
        'aggregated_rows':len(averaged),
        'test_counts':{k:{'rows':len(v),'finite_pvalues':int(v.p_value.notna().sum()),
                          'significant':int((v.p_holm_global < .05).sum())} for k,v in results.items()},
        'overall_means':averaged.groupby('model')[metrics].mean().reset_index().to_dict(orient='records'),
        'interpretation':'Robustness check only. Output equality does not establish biological identity or independence.'}
    (out/'sensitivity_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
