# Verification performed

- The complete results CSV contains 17,550 rows and 78 filenames.
- All file × scenario × dimension × model × seed keys are unique and match the complete factorial design.
- All eleven metric columns contain finite values.
- EXP.py SHA-256 equals the implementation hash recorded in Appendix A: `2f9228f31471c3c705d206b03ed0aa27ce1c0104091c98395c22ff55c0aa31f7`.
- Original desktop scripts, AMF input files, results CSV, statistical tables, and report are preserved byte-for-byte.
- The portable statistical launcher completed successfully.
- All 12 regenerated statistical tables matched the archived tables using relative tolerance 1e-9 and absolute tolerance 1e-12.
- The sensitivity script completed successfully and recovered 69 exact MDS-output-signature groups and 3,105 group-condition rows. Its significant comparison counts were 150 model tests, 426 pairwise tests, 111 dimension tests, and 57 environment tests.
- All Python files passed syntax parsing. The full embedding experiment was not rerun; its portable launcher received syntax checks only.

Analysis verification environment: Python 3.12, NumPy 2.5.4, pandas 3.0.1, SciPy 1.18.1, and Matplotlib 3.11.2. These are verification versions, not a claim about the original experiment environment.
