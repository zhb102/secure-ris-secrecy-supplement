"""Alternating optimization for robust secure RIS design."""

from __future__ import annotations

import numpy as np

from channel_model import equivalent_channels, quantized_phases
from simulation_config import Config

LN2 = np.log(2.0)


def secrecy_rate(
    h_b: np.ndarray,
    h_e: np.ndarray,
    w: np.ndarray,
    eps_abs: float,
    noise: float,
) -> float:
    """Worst-case secrecy rate for a fixed beamformer (dB-free, bit/s/Hz)."""

    return max(0.0, objective(h_b, h_e, w, eps_abs, noise))


def objective(
    h_b: np.ndarray,
    h_e: np.ndarray,
    w: np.ndarray,
    eps_abs: float,
    noise: float,
) -> float:
    """Unclamped difference of rates, used as the ascent objective.

    Optimizing the unclamped difference (instead of max(0, .)) keeps the
    gradient informative even when the current design happens to give a
    negative instantaneous difference.
    """

    g_b = abs(np.vdot(h_b, w)) ** 2 / noise
    g_e = (abs(np.vdot(h_e, w)) + eps_abs * np.linalg.norm(w)) ** 2 / noise
    return np.log2(1.0 + g_b) - np.log2(1.0 + g_e)


def beamforming_gradient(
    h_b: np.ndarray, h_e: np.ndarray, w: np.ndarray, eps_abs: float, noise: float
) -> np.ndarray:
    """Wirtinger gradient of the worst-case secrecy rate with respect to w*."""

    a = abs(np.vdot(h_b, w)) ** 2
    term_b = (h_b * np.vdot(h_b, w)) / (LN2 * (1.0 + a / noise) * noise)

    v = np.vdot(h_e, w)
    b = abs(v) + eps_abs * np.linalg.norm(w)
    if abs(v) < 1e-12:
        term_e = np.zeros_like(w)
    else:
        # d|v|/dw* = v * h_e / (2|v|) for v = h_e^H w
        term_e = (
            (b / (LN2 * (1.0 + b ** 2 / noise) * noise))
            * (h_e * v)
            / abs(v)
        )
    return term_b - term_e


def update_beamforming(
    h_b: np.ndarray, h_e: np.ndarray, eps_abs: float, cfg: Config
) -> np.ndarray:
    """Gradient ascent on the sphere ||w||^2 = P with backtracking line search.

    The first restart is maximum-ratio transmission, which is a strong
    initialization for the legitimate link; the remaining restarts are random
    directions that help escape poor local optima.
    """

    rng = np.random.default_rng(cfg.seed)
    radius = np.sqrt(cfg.p_lin)
    inits = [h_b / np.linalg.norm(h_b)]
    for _ in range(max(cfg.n_restarts - 1, 0)):
        v = rng.standard_normal(cfg.nt) + 1j * rng.standard_normal(cfg.nt)
        inits.append(v / np.linalg.norm(v))

    best_w, best_val = None, -np.inf
    for init in inits:
        w = radius * init
        val = objective(h_b, h_e, w, eps_abs, cfg.noise_lin)
        step = 0.5 * radius
        for _ in range(cfg.bf_steps):
            grad = beamforming_gradient(h_b, h_e, w, eps_abs, cfg.noise_lin)
            norm = np.linalg.norm(grad)
            if norm < 1e-15:
                break
            improved = False
            for _ in range(12):
                w_new = w + step * grad / norm
                w_new = radius * w_new / np.linalg.norm(w_new)
                val_new = objective(h_b, h_e, w_new, eps_abs, cfg.noise_lin)
                if val_new > val:
                    w, val = w_new, val_new
                    step *= 1.5
                    improved = True
                    break
                step *= 0.5
            if not improved or step < 1e-9 * radius:
                break
        if val > best_val:
            best_w, best_val = w, val
    return best_w


def update_phases(
    ch: dict[str, np.ndarray],
    w: np.ndarray,
    theta: np.ndarray,
    candidates: np.ndarray,
    eps_abs: float,
    cfg: Config,
) -> np.ndarray:
    """Element-wise coordinate ascent over a discrete (or gridded) phase set."""

    g_mat, g_b, g_e = ch["G"], ch["g_b"], ch["g_e"]
    h_d, h_de = ch["h_d"], ch["h_de"]
    theta = theta.copy()
    phi = np.exp(1j * theta)
    h_b = h_d + g_mat.conj().T @ (phi * g_b)
    h_e = h_de + g_mat.conj().T @ (phi * g_e)

    for n in range(cfg.n_ris):
        base_b = h_b - g_mat[n].conj() * (phi[n] * g_b[n])
        base_e = h_e - g_mat[n].conj() * (phi[n] * g_e[n])
        cand = np.exp(1j * candidates)[:, None] * (
            g_b[n] * g_mat[n].conj()
        )[None, :]
        h_b_all = base_b[None, :] + cand
        h_e_all = base_e[None, :] + cand * (g_e[n] / g_b[n])
        rates = [
            secrecy_rate(h_b_all[i], h_e_all[i], w, eps_abs, cfg.noise_lin)
            for i in range(len(candidates))
        ]
        best = int(np.argmax(rates))
        theta[n] = candidates[best]
        phi[n] = np.exp(1j * theta[n])
        h_b = h_b_all[best]
        h_e = h_e_all[best]
    return theta


def aligned_phases(
    ch: dict[str, np.ndarray], w: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """Phase initialization that coherently aligns the reflected paths."""

    c = ch["G"].conj() * ch["g_b"][:, None]
    proj = np.conj(c) @ w
    theta = np.angle(proj)
    idx = np.argmin(np.abs(np.mod(theta[:, None] - candidates[None, :] + np.pi, 2 * np.pi) - np.pi), axis=1)
    return candidates[idx]


def alternating_optimization(
    ch: dict[str, np.ndarray],
    cfg: Config,
    design_eps_abs: float,
    candidate_set: np.ndarray | None = None,
    record_history: bool = False,
) -> dict[str, object]:
    """Alternating optimization of the beamformer and the RIS phases.

    ``design_eps_abs`` is the uncertainty radius assumed during design.
    """

    rng = np.random.default_rng(cfg.seed + 1)
    candidates = quantized_phases(cfg.bits) if candidate_set is None else candidate_set
    inits: list[np.ndarray] = []

    # Restart 0: align the reflected paths with the direct channel.
    w_mrt = ch["h_d"] / np.linalg.norm(ch["h_d"]) * np.sqrt(cfg.p_lin)
    inits.append(aligned_phases(ch, w_mrt, candidates))
    for _ in range(max(cfg.ao_restarts - 1, 0)):
        inits.append(rng.uniform(0.0, 2.0 * np.pi, cfg.n_ris))

    best: dict[str, object] | None = None
    best_val = -np.inf
    for theta0 in inits:
        theta = theta0.copy()
        history: list[float] = []
        h_b, h_e = equivalent_channels(ch, theta)
        w = update_beamforming(h_b, h_e, design_eps_abs, cfg)
        prev = -np.inf
        for _ in range(cfg.ao_iters):
            theta = update_phases(ch, w, theta, candidates, design_eps_abs, cfg)
            h_b, h_e = equivalent_channels(ch, theta)
            w = update_beamforming(h_b, h_e, design_eps_abs, cfg)
            cur = objective(h_b, h_e, w, design_eps_abs, cfg.noise_lin)
            if record_history:
                history.append(cur)
            if abs(cur - prev) < cfg.tol and not record_history:
                break
            prev = cur
        h_b, h_e = equivalent_channels(ch, theta)
        val = objective(h_b, h_e, w, design_eps_abs, cfg.noise_lin)
        if val > best_val:
            best_val = val
            best = {
                "w": w,
                "theta": theta,
                "h_b": h_b,
                "h_e": h_e,
                "history": np.asarray(history if history else [cur]),
            }
    assert best is not None
    return best


def secrecy_unaware_design(
    ch: dict[str, np.ndarray], cfg: Config, candidates: np.ndarray | None = None
) -> dict[str, object]:
    """Benchmark that maximises the legitimate rate and ignores the eavesdropper.

    The beamformer is maximum-ratio transmission towards the direct channel and
    the RIS phases are aligned to the legitimate link only.
    """

    candidates = quantized_phases(cfg.bits) if candidates is None else candidates
    w = ch["h_d"] / np.linalg.norm(ch["h_d"]) * np.sqrt(cfg.p_lin)
    theta = aligned_phases(ch, w, candidates)
    for _ in range(4):
        h_b, _ = equivalent_channels(ch, theta)
        w = h_b / np.linalg.norm(h_b) * np.sqrt(cfg.p_lin)
        theta = aligned_phases(ch, w, candidates)
    h_b, h_e = equivalent_channels(ch, theta)
    return {"w": w, "theta": theta, "h_b": h_b, "h_e": h_e, "history": np.asarray([])}
