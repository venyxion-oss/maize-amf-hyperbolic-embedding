# Methods and interpretation

## Research question

We compare Poincaré, multidimensional scaling (MDS), and spectral embedding as representations of the same coupled maize root–AMF networks. Our objectives are to assess reconstruction error, radial hierarchy, resistance-related preservation, and simulated nutrient-response quantities across dimensions and environmental parameters.

## Inputs and graph construction

Root geometry and branching information are taken from 78 files in Wu's *Lateral roots dataset*, version 5. `rsml_input_manifest.csv` records the filenames and SHA-256 checksums. Our fungal graph is supplied through `AMF_network_nodes.csv` and `AMF_network_edges.csv`. It is a synthetic tree with supplied interface and nutrient-capacity attributes, not observed fungal microscopy.

`EXP.py` transforms root architecture using five scenario parameters (0.2, 0.4, 0.6, 0.8, 1.0), constructs resistance-inspired edge costs, couples fungal nodes to roots, and coarsens graphs toward 400 nodes. Normalized shortest-path costs define the common representation target. These shortest paths are not a solution of a whole-root pressure–flow system.

## Embedding comparison

Each method is evaluated at dimensions 2, 16, and 32 with five seeds. Poincaré coordinates and a positive curvature parameter are optimized with the implemented distance-reconstruction objective. The implementation details, numerical safeguards, alignment rule, and defaults are given in `EXP.py`; readers should use that implementation when reproducing the results.

The design contains 78 × 5 × 3 × 3 × 5 = 17,550 model runs. These are repeated computational conditions applied to 78 root-file identifiers.

## Outcomes

We distinguish distance reconstruction (MRE, Raw MRE, Stress), radial hierarchy, resistance-related measures, edge-based transport preservation, and simulated uptake outcomes. `DATA_DICTIONARY.md` defines the CSV columns. Lower MRE and Stress indicate less reconstruction error under their respective definitions.

Resistance is modeled through graph proxies. The hydraulic-preservation ratio can reach its maximum under resistance underestimation, so a high value alone is not physiological validation. Nutrient flux and uptake are relative simulation outputs; curve area and final uptake are linked through the modeled response function.

## Statistical analysis

We average seeds within each file × scenario × dimension × method condition, yielding 3,510 records. `STATS.py` computes descriptive summaries, Friedman comparisons, paired Wilcoxon contrasts, rank-biserial effects, and Holm-adjusted p-values. Undefined constant-reference tests are retained as undefined rather than interpreted as nonsignificant. The correction families and operational details are defined in the statistical source.

The root-file identifier is the analysis block. Separate filenames do not independently establish separate biological plants or acquisitions. Source provenance remains important when interpreting generalization.

## Sensitivity analysis

Nine pairs of filenames have exactly identical MDS output arrays across all 11 metrics, scenarios, dimensions, and seeds. We assess robustness by averaging within the 69 distinct output-signature groups and repeating the inferential comparisons. The grouping rule and implementation are in `run_sensitivity.py`. This analysis does not replace the original file-based analysis or determine biological sample identity.

## Scope

We evaluate static computational architectures and parameterized scenarios. The study does not measure plant uptake, validate fungal taxonomy, or establish field-treatment responses. Physiological calibration, independent architecture collections, and acquisition provenance are needed before extending these comparisons to biological prediction.

## Source and related research

- Wu, Q. (2017). *Lateral roots dataset* (Version 5). https://doi.org/10.6084/m9.figshare.4836890.v5
- Nickel, M., & Kiela, D. (2017). Poincaré embeddings for learning hierarchical representations. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1705.08039
- Dunbabin, V. M., et al. (2013). Modelling root–soil interactions using three-dimensional models of root growth, architecture and function. *Plant and Soil*, 372, 93–124. https://doi.org/10.1007/s11104-013-1769-y
- Wilson, G., et al. (2017). Good enough practices in scientific computing. *PLOS Computational Biology*, 13(6), e1005510. https://doi.org/10.1371/journal.pcbi.1005510
