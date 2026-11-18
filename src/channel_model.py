"""Channel generation for the RIS-assisted secure transmission model."""

from __future__ import annotations

import numpy as np

from simulation_config import Config, db2lin, path_loss


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _rician(
    rng: np.random.Generator, shape: tuple[int, ...], kappa_db: float, gain: float
) -> np.ndarray:
    """Rician fading with a deterministic line-of-sight component."""

    kappa = db2lin(kappa_db)
    los = np.sqrt(kappa / (1.0 + kappa))
    nlos = np.sqrt(1.0 / (1.0 + kappa)) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)
    return np.sqrt(gain) * (los + nlos)


def _rayleigh(rng: np.random.Generator, shape: tuple[int, ...], gain: float) -> np.ndarray:
    return np.sqrt(gain / 2.0) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )


def generate_channels(cfg: Config, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw one channel realization.

    Returns
    -------
    dict with keys:
        G    : BS -> RIS channel, shape (N, Nt)
        g_b  : RIS -> Bob channel, shape (N,)
        g_e  : RIS -> Eve channel, shape (N,)
        h_d  : BS -> Bob direct channel, shape (Nt,)
        h_de : BS -> Eve direct channel, shape (Nt,)
    """

    geo = cfg.geometry
    n, nt = cfg.n_ris, cfg.nt

    d_g = _distance(geo.bs, geo.ris)
    d_gb = _distance(geo.ris, geo.user)
    d_ge = _distance(geo.ris, geo.eve)
    d_hd = _distance(geo.bs, geo.user)
    d_he = _distance(geo.bs, geo.eve)

    a_ris = cfg.pathloss_exp_ris
    a_dir = cfg.pathloss_exp_direct

    g = _rician(rng, (n, nt), cfg.kappa_db, path_loss(d_g, cfg, a_ris))
    g_b = _rician(rng, (n,), cfg.kappa_db, path_loss(d_gb, cfg, a_ris))
    g_e = _rayleigh(rng, (n,), path_loss(d_ge, cfg, a_ris))
    h_d = _rayleigh(
        rng, (nt,), path_loss(d_hd, cfg, a_dir) * db2lin(-cfg.blockage_user_db)
    )
    h_de = _rayleigh(
        rng, (nt,), path_loss(d_he, cfg, a_dir) * db2lin(-cfg.blockage_eve_db)
    )

    return {"G": g, "g_b": g_b, "g_e": g_e, "h_d": h_d, "h_de": h_de}


def equivalent_channels(
    ch: dict[str, np.ndarray], theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (h_b, h_e) for a given RIS configuration theta."""

    phi = np.exp(1j * theta)
    g = ch["G"]
    h_b = ch["h_d"] + g.conj().T @ (phi * ch["g_b"])
    h_e = ch["h_de"] + g.conj().T @ (phi * ch["g_e"])
    return h_b, h_e


def worst_case_eps_abs(ch: dict[str, np.ndarray], cfg: Config, eps_rel: float | None = None) -> float:
    """Absolute CSI error radius, scaled by the eavesdropper channel norm."""

    eps_rel = cfg.eps if eps_rel is None else eps_rel
    h_e = ch["h_de"] + ch["G"].conj().T @ ch["g_e"]
    return eps_rel * float(np.linalg.norm(h_e))

def quantized_phases(bits: int) -> np.ndarray:
    """Discrete phase set with 2**bits uniformly spaced values."""

    levels = 2 ** bits
    return 2.0 * np.pi * np.arange(levels) / levels
