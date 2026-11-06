# Maize AMF network embedding research archive

Supporting code and simulation data for **In Silico Optimization of Zea mays–Arbuscular Mycorrhizal Fungi Networks: A Hyperbolic Embedding Framework for Modeling Nutrient Uptake Dynamics**.

## Start here

- [Experiment implementation](EXP.py) — original desktop source, unchanged.
- [Statistical analysis implementation](STATS.py) — original desktop source, unchanged.
- [Complete 17,550-row simulation results](FINAL_EXP_RESULTS.csv).
- [Data dictionary](DATA_DICTIONARY.md).
- [Consolidated statistical report](STATISTICAL_REPORT.txt).
- [Download the statistical figures](statistical_figures.zip).
- [File checksums](SHA256SUMS.txt).

The CSV is the complete model-run output, not raw physiological measurements. It contains 78 RSML filenames × 5 scenarios × 3 dimensions × 3 methods × 5 seeds. Seed averaging gives 3,510 file-condition records. Repeated conditions and seeds do not increase the biological sample size.

## Reproduce the statistical analysis

Use Python in a new virtual environment. The dependencies below are requirements, not a recovered lockfile from the original experiment. The original Python/library versions and hardware were not recorded in these files, so bitwise reproduction of the experiment is not promised.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python validate_data.py
python run_analysis.py
python run_sensitivity.py
```

`run_analysis.py` executes the original STATS.py with repository-relative input and a separate output directory. It changes only the two path assignments in memory. Existing archived results and desktop files are not overwritten. Outputs go to `reproduced_analysis/`. The sensitivity script separately implements the paper's exact-MDS-output-signature grouping analysis and writes `reproduced_sensitivity/`; this is a robustness check, not confirmation that the groups are distinct biological samples.

## Re-run the embedding experiment

Obtain Wu's *Lateral roots dataset*, version 5, from https://doi.org/10.6084/m9.figshare.4836890.v5 and extract the 78 matching XML/RSML files into a directory. [rsml_input_manifest.csv](rsml_input_manifest.csv) lists the exact source filenames and SHA-256 hashes used locally. Source root files are referenced at their original repository rather than redistributed here.

```bash
python -m pip install -r requirements-experiment.txt
python run_experiment.py --dataset-dir /path/to/rsml_inputs
```

This is a computationally intensive run. It writes to a new `rerun_results.csv`, using EXP.py's existing resume behavior. It never replaces the archived `FINAL_EXP_RESULTS.csv`. GPU/CPU differences, package versions, file provenance, and numerical defaults may affect output. The portable launcher preserves the original experiment functions and supplies explicit paths.

## Inputs

| File | Description |
|---|---|
| AMF_network_nodes.csv | Synthetic fungal nodes and supplied attributes |
| AMF_network_edges.csv | Synthetic fungal network edges |
| AMF_literature_parameters.csv | Accompanying parameter metadata; not directly read by EXP.py |
| rsml_input_manifest.csv | Exact root input names and hashes; obtain originals from the dataset DOI |

The AMF graph is a constructed computational input, not a reconstruction of observed microscopy. The modeled resistance, flux, and uptake quantities require physiological calibration.

## Archived statistical outputs

| File | Description |
|---|---|
| 01_seed_level_summary.csv | Seed-level descriptive summary |
| 02_file_level_seed_averaged_data.csv | 3,510 seed-averaged file-condition records |
| 03_descriptive_statistics.csv | Grouped descriptive statistics |
| 04_friedman_model_tests.csv | Model comparisons and Holm correction |
| 05_pairwise_wilcoxon_model_tests.csv | Paired model contrasts and rank-biserial effects |
| 06_dimension_friedman_tests.csv | Dimension comparisons |
| 07_environment_friedman_tests.csv | Scenario comparisons |
| 08_environmental_trends.csv | Within-file Spearman trends |
| 09_poincare_curvature_summary.csv | Stored Poincare parameter summaries |
| 10_metric_spearman_correlations.csv | Exploratory metric correlations |
| 11_best_model_by_condition.csv | Metric-specific method rankings |
| 12_pairwise_percent_improvements.csv | Descriptive percentage differences |
| STATISTICAL_REPORT.txt | Consolidated original analysis report |

All archived sources, inputs, results, and output tables are copied byte-for-byte from the supplied desktop files. Portable launchers, validation, documentation, manifests, and the standalone sensitivity script were added for sharing. No embedding experiment was rerun when preparing this archive. See VALIDATION.md for checks actually completed.

## Citation and versioning

For a paper submission, cite an immutable release or commit rather than only the changing main branch. A DOI-issuing archive such as Zenodo can preserve a repository release. No DOI has been minted for this archive yet. Retain the dataset's own DOI citation independently.

Authors listed on the manuscript: Raven V. Santos, Akhira S. Manlangit, and Erra Nicole F. De Guzman. No publication year, journal acceptance, or original software environment is asserted here. No reuse license has been selected; public visibility alone does not grant an open-source license. Third-party data retains its source terms.

## Research computing reference

Wilson, G., et al. (2017). Good enough practices in scientific computing. *PLOS Computational Biology*, 13(6), e1005510. https://doi.org/10.1371/journal.pcbi.1005510
