# Reproducing our analyses

We provide two workflows: reanalyzing the archived simulation results, and rerunning the embedding experiment from root architecture files. Start with the statistical analysis; it does not require PyTorch or a GPU.

## 1 Set up Python

The statistical workflow has been verified with Python 3.12. Open a terminal in the downloaded repository directory.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis-verified.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis-verified.txt
```

The pinned file records the environment used for the statistical verification, not the original embedding experiment. `requirements-analysis.txt` lists the same direct dependencies without version pins for other compatible environments.

## 2 Validate the archive

```bash
python validate_data.py
```

This checks file integrity, the complete factorial design, unique model-run keys, finite metrics, scenario parameters, and seed-averaged record counts. A validation failure returns a nonzero exit status and an explanation.

## 3 Reproduce the statistical tables and figures

```bash
python run_analysis.py
python verify_reproduction.py
```

The launcher executes `STATS.py` with the input and output paths supplied in memory. The statistical functions and model-run CSV are unchanged. Results are written to `reproduced_analysis/`, leaving the archived tables intact. The verification script compares the 12 regenerated tables with the archive using relative tolerance 1e-9 and absolute tolerance 1e-12.

To analyze a different CSV or use a different output directory:

```bash
python run_analysis.py --input /path/to/results.csv --output-dir /path/to/new-analysis
python verify_reproduction.py --output-dir /path/to/new-analysis
```

Comparison with the archived tables is meaningful only when the original input is used.

## 4 Run the output-signature sensitivity analysis

```bash
python run_sensitivity.py
```

Outputs are written to `reproduced_sensitivity/`. The script groups files with exactly matching MDS output arrays across all 11 metrics, scenarios, dimensions, and seeds. For the supplied CSV, this yields 69 groups and 3,105 group-condition records. It averages file-level results within each group and repeats the model, dimension, and scenario comparisons. This is a robustness check; equal outputs do not prove that the source biological specimens are identical.

## 5 Rerun the full embedding experiment

Obtain [Wu's dataset, version 5](https://doi.org/10.6084/m9.figshare.4836890.v5), and place the 78 files listed in `rsml_input_manifest.csv` in one directory. The external root files are not redistributed here. Optionally verify them before running:

```bash
python validate_data.py --rsml-dir /path/to/rsml_inputs
python -m pip install -r requirements-experiment.txt
python run_experiment.py --dataset-dir /path/to/rsml_inputs
```

The full experiment can require substantial computing time. `run_experiment.py` supplies the root-data directory, AMF input paths, and a separate output path to `EXP.py`. Its default destination is `rerun_results.csv`; the original resume behavior is retained.

A smaller exploratory run can be requested with `--max-files 1`. It does not reproduce the full study.

The archived original software environment was not recorded in the supplied source files. Package defaults, hardware, and numerical differences may affect a fresh embedding run. We therefore distinguish statistical reproduction from full experimental reproduction in [VALIDATION.md](VALIDATION.md).
