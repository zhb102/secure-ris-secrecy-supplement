"""Simulation parameters for the RIS-assisted secrecy-rate experiments.

The values in this module correspond to the parameter table in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """Two-dimensional positions in metres."""

    # The RIS is deployed close to the legitimate user; the direct BS--user
    # link is heavily blocked, whereas the eavesdropper is far from the RIS and
    # is served mainly by the (partially blocked) direct link. This is the
    # regime in which the RIS genuinely contributes to secrecy.
    bs: tuple[float, float] = (0.0, 0.0)
    ris: tuple[float, float] = (15.0, 0.0)
    user: tuple[float, float] = (20.0, -8.0)
    eve: tuple[float, float] = (60.0, -20.0)


@dataclass(frozen=True)
class Config:
    # Antenna configuration
    nt: int = 4
    n_ris: int = 32
    bits: int = 2

    # Power and noise (dBm)
    p_dbm: float = 20.0
    noise_dbm: float = -90.0

    # Channel model
    pathloss_exp_direct: float = 3.0
    pathloss_exp_ris: float = 2.0
    pathloss_ref_db: float = -30.0
    kappa_db: float = 10.0
    blockage_user_db: float = 35.0
    blockage_eve_db: float = 20.0

    # CSI uncertainty: ||delta|| <= eps * ||h_e||
    eps: float = 0.05

    # Algorithm
    ao_iters: int = 12
    ao_restarts: int = 3
    bf_steps: int = 25
    bf_lr: float = 0.5
    tol: float = 1e-4
    n_restarts: int = 2

    # Monte Carlo
    n_realizations: int = 500
    seed: int = 20261019
    geometry: Geometry = field(default_factory=Geometry)

    @property
    def p_lin(self) -> float:
        return 10.0 ** (self.p_dbm / 10.0)

    @property
    def noise_lin(self) -> float:
        return 10.0 ** (self.noise_dbm / 10.0)

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset)


def db2lin(x_db: float) -> float:
    return 10.0 ** (x_db / 10.0)


def path_loss(distance_m: float, cfg: Config, alpha: float) -> float:
    """Linear power gain of the large-scale path loss."""

    distance_m = max(distance_m, 1.0)
    pl_db = cfg.pathloss_ref_db - 10.0 * alpha * np.log10(distance_m)
    return db2lin(pl_db)
