# Supplementary material

## Robust joint beamforming and discrete phase-shift design for secure RIS-assisted wireless communications

This directory contains the simulation code and the archived numerical outputs used for the manuscript. The result tables are frozen snapshots; the archived CSV values are not edited in this package.

## Directory map

```text
.
├── README.md
├── NOTES.md
├── requirements.txt
├── src/
│   ├── simulation_config.py
│   ├── channel_model.py
│   ├── optimization.py
│   ├── experiment_runner.py
│   ├── reproduce_results.py
│   ├── plot_paper_figures.py
│   └── draw_system_diagram.py
├── results/
│   ├── power_sweep.csv
│   ├── elements_sweep.csv
│   ├── convergence_trace.csv
│   ├── csi_error_sweep.csv
│   └── phase_resolution_sweep.csv
├── figures/
│   ├── system_model_and_algorithm_flow.pdf
│   ├── secrecy_rate_vs_transmit_power.pdf
│   ├── secrecy_rate_vs_ris_elements.pdf
│   ├── ao_convergence_trace.pdf
│   └── secrecy_rate_vs_csi_error.pdf
└── style/
    └── ieee.mplstyle
```

## Implementation components

| File | Role |
|---|---|
| `src/simulation_config.py` | Geometry, antenna/RIS sizes, power, noise, path-loss, Rician factor, CSI error radius, optimiser limits, and random seed |
| `src/channel_model.py` | Rician/Rayleigh channel generation, equivalent legitimate/eavesdropper channels, norm-bounded error radius, discrete phase set |
| `src/optimization.py` | Worst-case secrecy objective, projected Wirtinger-gradient beamforming, element-wise phase search, alternating optimisation and restarts |
| `src/experiment_runner.py` | Power, element-count, CSI-error, convergence and phase-resolution sweeps |
| `src/reproduce_results.py` | Runs the five reported sweeps with 500 channel realisations |
| `src/plot_paper_figures.py` | Rebuilds the four data figures from the archived result tables |
| `src/draw_system_diagram.py` | Rebuilds the system-model and algorithm-flow schematic |
| `style/ieee.mplstyle` | Shared Matplotlib style for the manuscript figures |

The optimiser uses NumPy only; it does not call an external convex solver.

## Result-to-claim map

| Archived table or figure | Manuscript location | Relevant claim |
|---|---|---|
| `results/power_sweep.csv`; `figures/secrecy_rate_vs_transmit_power.pdf` | Fig. 1; abstract; conclusion | Proposed rate rises from 10.2 to 15.7 bit/s/Hz over 10–40 dBm; maximum gap to the 36-point reference is 0.53 bit/s/Hz |
| `results/elements_sweep.csv`; `figures/secrecy_rate_vs_ris_elements.pdf` | Fig. 2 | Increasing `N` from 8 to 128 raises the rate from 9.2 to 15.1 bit/s/Hz; no-RIS remains near 3.0 |
| `results/convergence_trace.csv`; `figures/ao_convergence_trace.pdf` | Fig. 3 | Average rate rises from 12.26 to 12.33 and reaches a fixed point within six iterations over 20 realisations |
| `results/csi_error_sweep.csv`; `figures/secrecy_rate_vs_csi_error.pdf` | Fig. 4 | From `epsilon=0` to `0.2`, the proposed rate falls from 12.80 to 11.13; secrecy-unaware remains 1.8–2.7 lower |
| `results/phase_resolution_sweep.csv` | Table I | At 20 dBm, 1–4 bits give 12.28, 12.30, 12.57 and 12.66; the 36-point reference gives 12.70 |
| `src/simulation_config.py` | Parameter table | `N_t=4`, `N=32`, `b=2`, `P=10–40 dBm`, `-90 dBm` noise, 500 realisations, `epsilon=0.05` |
| `figures/system_model_and_algorithm_flow.pdf`; `src/draw_system_diagram.py` | Fig. 1 schematic | System geometry and alternating-optimisation flow |

The manuscript's 36-point phase grid is a numerical reference, not a formal continuous-phase upper bound. The alternating optimisation reports a monotone objective value but does not claim global optimality.

## Parameters used for the reported runs

| Parameter | Value |
|---|---|
| Base-station antennas `N_t` | 4 |
| RIS elements `N` | 32; swept 8, 16, 32, 64, 128 |
| Quantisation bits `b` | 2; swept 1–4 |
| Transmit power `P` | 10–40 dBm |
| Noise power | −90 dBm |
| Direct / RIS path-loss exponent | 3 / 2 |
| Rician factor | 10 dB |
| Blockage on BS–user / BS–Eve | 35 dB / 20 dB |
| CSI error `epsilon` | 0.05; swept 0–0.2 |
| Channel realisations | 500 for sweeps; 20 for the convergence trace |

## Reproduce the figures and tables

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/reproduce_results.py
python src/plot_paper_figures.py
python src/draw_system_diagram.py
```

`src/reproduce_results.py` writes the five result tables into `results/`; `plot_paper_figures.py` rebuilds the four data figures. The archived tables and figures are already included, so the commands are optional for inspection.

For a single sweep, use `python src/experiment_runner.py power`, `elements`, `epsilon`, `convergence`, `bits`, or `all`.

## Notes

See `NOTES.md` for the archived table record and the phase-reference label used in `phase_resolution_sweep.csv`.
