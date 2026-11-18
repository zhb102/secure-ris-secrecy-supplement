"""Reproduce the numerical results reported in the manuscript."""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from experiment_runner import (
    convergence_trace,
    sweep_bits,
    sweep_elements,
    sweep_epsilon,
    sweep_power,
)
from simulation_config import Config


def main() -> None:
    cfg = Config(n_realizations=500)
    schemes = ["no_ris", "random_phase", "unaware", "proposed", "continuous"]
    started = time.time()

    print("power sweep", flush=True)
    print(sweep_power(cfg, [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0], schemes).name)
    print("element sweep", flush=True)
    print(sweep_elements(cfg, [8, 16, 32, 64, 128], schemes).name)
    print("CSI-error sweep", flush=True)
    print(sweep_epsilon(cfg, [0.0, 0.02, 0.05, 0.10, 0.20], ["proposed", "unaware"]).name)
    print("convergence", flush=True)
    print(convergence_trace(cfg, n_realizations=20).name)
    print("phase-resolution sweep", flush=True)
    print(sweep_bits(cfg, n_realizations=cfg.n_realizations).name)
    print(f"elapsed {time.time() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
