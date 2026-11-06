# Results data dictionary

One row represents one file × environment × dimension × method × seed.

| Column | Meaning |
|---|---|
| file | RSML/XML source filename; file-level analysis block, not guaranteed independent plant identity. |
| environment | Parameterized scenario label; not an observed field treatment. |
| theta | Scenario parameter: 0.2, 0.4, 0.6, 0.8, or 1.0. |
| model | Poincare, MDS, or Spectral. |
| dimension | Embedding dimension: 2, 16, or 32. |
| seed | Stochastic seed: 42, 123, 456, 789, or 1011. |
| MRE | Mean relative aligned distance error with first-percentile denominator floor; lower is better. |
| Raw_MRE | Relative error without percentile floor; lower is better. |
| Stress | Normalized aligned distance distortion; lower is better. |
| Hierarchy | Mean valid root-origin Spearman correlation of branch level and radial embedded distance. |
| Physical_Hydraulic | Reference graph mean resistance proxy; not calibrated hydraulic conductance. |
| Hydraulic | Embedding-adjusted mean edge-resistance proxy. |
| Hydraulic_Preservation | Clipped reference-to-adjusted resistance ratio; 1 also occurs under underestimation. |
| Embedding_Transport_Preservation | Mean clipped reciprocal log-distance distortion score over valid edges. |
| Nutrient_Uptake | Trapezoidal uptake-curve area over relative time [0,1]. |
| Nutrient_Final | Simulated uptake response at relative time 1. |
| Nutrient_Flux | Relative modeled interface-mediated flux; not measured nutrient transport. |
| curvature | Stored learned positive Poincare parameter c; interpret using EXP.py distance convention. |
| nodes | Processed graph node count. |
| edges | Processed graph edge count. |
| symbiotic_interfaces | Retained root-fungal interface count. |
| amf_species | Supplied fungal species label; metadata, not taxonomic verification. |

The 17,550 rows are simulation outputs. Seeds are averaged before inference, producing 3,510 file-condition records. There are 78 filenames; independence of biological samples requires acquisition provenance.
