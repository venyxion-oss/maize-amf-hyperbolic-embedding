# Results and output catalogue

We provide the complete model-run CSV, the seed-averaged analysis data, all 12 statistical tables, the report, and 24 archived figures. CSV files can be downloaded and opened in Python, R, or a spreadsheet application. GitHub may limit the preview of larger files; use its download control or download the repository ZIP.

## Main data

| File | Contents |
|---|---|
| [FINAL_EXP_RESULTS.csv](FINAL_EXP_RESULTS.csv) | Complete 17,550-row simulation output |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Meaning and interpretation of every results column |
| [02_file_level_seed_averaged_data.csv](02_file_level_seed_averaged_data.csv) | 3,510 seed-averaged records used for file-based analysis |

## Statistical tables

| File | Contents |
|---|---|
| [01_seed_level_summary.csv](01_seed_level_summary.csv) | Seed-level descriptive summary |
| [03_descriptive_statistics.csv](03_descriptive_statistics.csv) | Grouped descriptive statistics |
| [04_friedman_model_tests.csv](04_friedman_model_tests.csv) | Model comparisons and Holm adjustment |
| [05_pairwise_wilcoxon_model_tests.csv](05_pairwise_wilcoxon_model_tests.csv) | Paired model contrasts and rank-biserial effects |
| [06_dimension_friedman_tests.csv](06_dimension_friedman_tests.csv) | Dimension comparisons |
| [07_environment_friedman_tests.csv](07_environment_friedman_tests.csv) | Environmental-scenario comparisons |
| [08_environmental_trends.csv](08_environmental_trends.csv) | Within-file Spearman trends |
| [09_poincare_curvature_summary.csv](09_poincare_curvature_summary.csv) | Stored Poincaré parameter summaries |
| [10_metric_spearman_correlations.csv](10_metric_spearman_correlations.csv) | Exploratory metric correlations |
| [11_best_model_by_condition.csv](11_best_model_by_condition.csv) | Metric-specific method rankings |
| [12_pairwise_percent_improvements.csv](12_pairwise_percent_improvements.csv) | Descriptive percentage differences |

## Report and figures

- [STATISTICAL_REPORT.txt](STATISTICAL_REPORT.txt): consolidated analysis report.
- [statistical_figures.zip](statistical_figures.zip): 24 original PNG figures, organized by metric and dimension.
- [comparison_preview.png](comparison_preview.png): one unchanged figure extracted from the archive for the repository overview.

## Inputs and integrity

- [AMF_network_nodes.csv](AMF_network_nodes.csv) and [AMF_network_edges.csv](AMF_network_edges.csv): synthetic AMF network inputs.
- [AMF_literature_parameters.csv](AMF_literature_parameters.csv): accompanying parameter metadata; not read directly by EXP.py.
- [rsml_input_manifest.csv](rsml_input_manifest.csv): source root filenames and checksums.
- [SHA256SUMS.txt](SHA256SUMS.txt): checksums for the repository's research files and documentation.

New analysis outputs go to `reproduced_analysis/`; new sensitivity outputs go to `reproduced_sensitivity/`. These directories are excluded from version control to distinguish fresh runs from the archived results.
