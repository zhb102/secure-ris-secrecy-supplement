"""Simulation driver for the RIS-assisted secrecy-rate experiments."""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from optimization import (
    alternating_optimization,
    objective,
    secrecy_unaware_design,
    update_beamforming,
)
from channel_model import (
    equivalent_channels,
    generate_channels,
    worst_case_eps_abs,
)
from simulation_config import Config

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _write_csv(path: pathlib.Path, header: list[str], rows: list[list[float]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def design_value(
    ch: dict[str, np.ndarray], cfg: Config, scheme: str, eps_eval: float
) -> float:
    """Unclamped design value R_b - R_e (used for reporting the raw gap)."""

    if scheme == "no_ris":
        h_b, h_e = ch["h_d"], ch["h_de"]
        w = update_beamforming(h_b, h_e, eps_eval, cfg)
        return objective(h_b, h_e, w, eps_eval, cfg.noise_lin)
    if scheme == "random_phase":
        rng = np.random.default_rng(cfg.seed + 7)
        theta = rng.uniform(0.0, 2.0 * np.pi, cfg.n_ris)
        h_b, h_e = equivalent_channels(ch, theta)
        w = update_beamforming(h_b, h_e, eps_eval, cfg)
        return objective(h_b, h_e, w, eps_eval, cfg.noise_lin)
    if scheme == "proposed":
        out = alternating_optimization(ch, cfg, design_eps_abs=eps_eval)
    elif scheme == "unaware":
        out = secrecy_unaware_design(ch, cfg)
    elif scheme == "continuous":
        grid = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
        out = alternating_optimization(
            ch, cfg, design_eps_abs=eps_eval, candidate_set=grid
        )
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    return objective(out["h_b"], out["h_e"], out["w"], eps_eval, cfg.noise_lin)


def evaluate_scheme(
    ch: dict[str, np.ndarray], cfg: Config, scheme: str, eps_eval: float
) -> float:
    """Return the worst-case secrecy rate of a scheme for one realization."""

    return max(0.0, design_value(ch, cfg, scheme, eps_eval))


def sweep_power(cfg: Config, powers_dbm: list[float], schemes: list[str]) -> pathlib.Path:
    rows: list[list[float]] = []
    for p_dbm in powers_dbm:
        cfg_p = Config(**{**cfg.__dict__, "p_dbm": p_dbm})
        acc = {s: 0.0 for s in schemes}
        for r in range(cfg_p.n_realizations):
            rng = cfg_p.rng(offset=r)
            ch = generate_channels(cfg_p, rng)
            eps_abs = worst_case_eps_abs(ch, cfg_p)
            for s in schemes:
                acc[s] += evaluate_scheme(ch, cfg_p, s, eps_abs)
        rows.append([p_dbm] + [acc[s] / cfg_p.n_realizations for s in schemes])
        print(
            f"  P={p_dbm:5.1f} dBm  "
            + "  ".join(f"{s}={acc[s] / cfg_p.n_realizations:6.3f}" for s in schemes),
            flush=True,
        )
    path = RESULTS / "power_sweep.csv"
    _write_csv(path, ["p_dbm"] + schemes, rows)
    return path


def sweep_elements(cfg: Config, n_list: list[int], schemes: list[str]) -> pathlib.Path:
    rows: list[list[float]] = []
    for n in n_list:
        cfg_n = Config(**{**cfg.__dict__, "n_ris": n})
        acc = {s: 0.0 for s in schemes}
        for r in range(cfg_n.n_realizations):
            rng = cfg_n.rng(offset=r)
            ch = generate_channels(cfg_n, rng)
            eps_abs = worst_case_eps_abs(ch, cfg_n)
            for s in schemes:
                acc[s] += evaluate_scheme(ch, cfg_n, s, eps_abs)
        rows.append([n] + [acc[s] / cfg_n.n_realizations for s in schemes])
        print(
            f"  N={n:4d}  "
            + "  ".join(f"{s}={acc[s] / cfg_n.n_realizations:6.3f}" for s in schemes),
            flush=True,
        )
    path = RESULTS / "elements_sweep.csv"
    _write_csv(path, ["n_ris"] + schemes, rows)
    return path


def sweep_epsilon(cfg: Config, eps_list: list[float], schemes: list[str]) -> pathlib.Path:
    rows: list[list[float]] = []
    for eps in eps_list:
        cfg_e = Config(**{**cfg.__dict__, "eps": eps})
        acc = {s: 0.0 for s in schemes}
        for r in range(cfg_e.n_realizations):
            rng = cfg_e.rng(offset=r)
            ch = generate_channels(cfg_e, rng)
            eps_abs = worst_case_eps_abs(ch, cfg_e)
            for s in schemes:
                acc[s] += evaluate_scheme(ch, cfg_e, s, eps_abs)
        rows.append([eps] + [acc[s] / cfg_e.n_realizations for s in schemes])
        print(
            f"  eps={eps:4.2f}  "
            + "  ".join(f"{s}={acc[s] / cfg_e.n_realizations:6.3f}" for s in schemes),
            flush=True,
        )
    path = RESULTS / "csi_error_sweep.csv"
    _write_csv(path, ["eps"] + schemes, rows)
    return path


def convergence_trace(cfg: Config, n_realizations: int = 20) -> pathlib.Path:
    max_len = 0
    traces = []
    for r in range(n_realizations):
        rng = cfg.rng(offset=r)
        ch = generate_channels(cfg, rng)
        eps_abs = worst_case_eps_abs(ch, cfg)
        out = alternating_optimization(ch, cfg, design_eps_abs=eps_abs, record_history=True)
        hist = np.asarray(out["history"], dtype=float)
        traces.append(hist)
        max_len = max(max_len, len(hist))

    rows = []
    for it in range(max_len):
        vals = [t[it] for t in traces if len(t) > it]
        rows.append([it + 1, float(np.mean(vals))])
    path = RESULTS / "convergence_trace.csv"
    _write_csv(path, ["iteration", "mean_secrecy_rate"], rows)
    return path


def sweep_bits(cfg: Config, n_realizations: int = 500) -> pathlib.Path:
    """Measure secrecy rate for 1--4 quantization bits and a 36-point grid."""

    rows: list[list[float]] = []
    for bits in (1, 2, 3, 4):
        cfg_b = Config(**{**cfg.__dict__, "bits": bits, "n_realizations": n_realizations})
        acc = 0.0
        for r in range(n_realizations):
            rng = cfg_b.rng(offset=r)
            ch = generate_channels(cfg_b, rng)
            eps_abs = worst_case_eps_abs(ch, cfg_b)
            acc += evaluate_scheme(ch, cfg_b, "proposed", eps_abs)
        rows.append([bits, acc / n_realizations])
        print(f"  b={bits}  proposed={acc / n_realizations:6.3f}", flush=True)

    acc = 0.0
    for r in range(n_realizations):
        rng = cfg.rng(offset=r)
        ch = generate_channels(cfg, rng)
        eps_abs = worst_case_eps_abs(ch, cfg)
        acc += evaluate_scheme(ch, cfg, "continuous", eps_abs)
    # The archived result table stores the 36-point reference under label 16.
    # Keep the label so the reproduction writes the same table layout.
    rows.append([16, acc / n_realizations])
    print(f"  continuous={acc / n_realizations:6.3f}", flush=True)

    path = RESULTS / "phase_resolution_sweep.csv"
    _write_csv(path, ["bits", "secrecy_rate"], rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=["power", "elements", "epsilon", "convergence", "bits", "all"],
    )
    args = parser.parse_args()

    cfg = Config()
    schemes = ["no_ris", "random_phase", "proposed", "continuous"]
    if args.task == "power":
        print(sweep_power(cfg, [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0], schemes).name)
    elif args.task == "elements":
        print(sweep_elements(cfg, [8, 16, 32, 64, 128], schemes).name)
    elif args.task == "epsilon":
        print(sweep_epsilon(cfg, [0.0, 0.02, 0.05, 0.10, 0.20], ["proposed", "unaware"]).name)
    elif args.task == "convergence":
        print(convergence_trace(cfg, n_realizations=20).name)
    elif args.task == "bits":
        print(sweep_bits(cfg, n_realizations=cfg.n_realizations).name)
    else:
        print(sweep_power(cfg, [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0], schemes).name)
        print(sweep_elements(cfg, [8, 16, 32, 64, 128], schemes).name)
        print(sweep_epsilon(cfg, [0.0, 0.02, 0.05, 0.10, 0.20], ["proposed", "unaware"]).name)
        print(convergence_trace(cfg, n_realizations=20).name)
        print(sweep_bits(cfg, n_realizations=cfg.n_realizations).name)


if __name__ == "__main__":
    main()
