# Archived result notes

The CSV files in `results/` are the result tables used for the manuscript figures and Table I. Their numerical values are preserved as archived; this supplementary copy does not rewrite measured values.

- `power_sweep.csv`: power sweep used for the reported runs.
- `elements_sweep.csv`: RIS element-count sweep.
- `convergence_trace.csv`: averaged alternating-optimisation trace.
- `csi_error_sweep.csv`: CSI-error sweep.
- `phase_resolution_sweep.csv`: phase-resolution table.

## Phase-resolution label

In `phase_resolution_sweep.csv`, the last row is labelled `16` in the archived file. It is the 36-point phase-grid reference used for Table I, whose manuscript value is 12.70. The label is retained so the archived CSV bytes remain unchanged; only the accompanying README explains the correspondence.

## Record

The archived tables were produced from the simulation code in this directory with the fixed seed in `simulation_config.py`. No separate execution logs were present in the original workspace; the CSV tables and vector figures are the authoritative archived record. The code prints result filenames only, and the archived result values are not altered by this packaging step.
