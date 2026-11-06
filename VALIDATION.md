# Verification record

We distinguish checks of our archived simulation output from a fresh execution of the embedding experiment.

## Archive integrity

- The results CSV contains 17,550 records and 78 root filenames.
- The file × scenario × dimension × method × seed combinations are unique and form the complete factorial design.
- All 11 metric columns contain finite values.
- Seed averaging produces 3,510 file-condition records.
- The original EXP.py, STATS.py, AMF inputs, simulation CSV, statistical tables, and report are retained without content changes.
- The EXP.py checksum matches Appendix A: `2f9228f31471c3c705d206b03ed0aa27ce1c0104091c98395c22ff55c0aa31f7`.

`python validate_data.py` repeats the archive checks. `--rsml-dir` also checks the 78 external root files against our manifest.

## Statistical reproduction

We ran the portable statistical launcher successfully and compared all 12 regenerated tables against the archived outputs with relative tolerance 1e-9 and absolute tolerance 1e-12. These comparisons passed. `verify_reproduction.py` makes the same comparison available to readers.

## Sensitivity analysis

The sensitivity script recovered 69 exact MDS-output-signature groups and 3,105 group-condition records. The significant comparison counts were:

| Comparison family | Significant comparisons |
|---|---:|
| Model Friedman tests | 150 |
| Pairwise Wilcoxon tests | 426 |
| Dimension Friedman tests | 111 |
| Scenario Friedman tests | 57 |

Grouping uses model-output signatures. It does not establish biological specimen identity.

## Verification environment

Python 3.12; NumPy 2.5.4; pandas 3.0.1; SciPy 1.18.1; Matplotlib 3.11.2. `requirements-analysis-verified.txt` records these direct package versions. It is not a complete lockfile and does not describe the original embedding environment.

All Python sources passed syntax parsing. The full embedding experiment has not been rerun for this verification; the experiment launcher has received syntax checks only. Original software versions and hardware were not recovered from the supplied files.
