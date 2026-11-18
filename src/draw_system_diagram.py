"""Render the system-model and algorithm-flow schematic as vector artwork."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
STYLE = ROOT / "style" / "ieee.mplstyle"

FS = 6.4


def box(ax, cx, cy, w, h, text, fc="#eef3fb", ec="#22303f", fs=FS, lw=0.8):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, p0, p1, style="-|>", color="#22303f", lw=0.9, ls="-", rad=0.0,
          zorder=1):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=7,
            linewidth=lw,
            linestyle=ls,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.0,
            shrinkB=1.0,
            zorder=zorder,
        )
    )


def draw_system(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.05, 6.15, "(a) System model", fontsize=FS + 0.6, fontweight="bold")

    # Base station
    ax.add_patch(Rectangle((0.5, 2.4), 0.85, 0.95, facecolor="#dfe7f3",
                           edgecolor="#22303f", linewidth=0.8, zorder=2))
    for dx in (0.18, 0.42, 0.66):
        ax.plot([0.5 + dx, 0.5 + dx], [3.35, 3.75], color="#22303f", lw=0.8, zorder=3)
        ax.plot([0.42 + dx, 0.58 + dx], [3.75, 3.75], color="#22303f", lw=0.8, zorder=3)
    ax.text(0.93, 2.05, "BS, $N_t$ antennas", ha="center", va="top", fontsize=FS)

    # RIS
    for i in range(7):
        ax.add_patch(Rectangle((4.55, 3.55 + i * 0.38), 0.30, 0.28,
                               facecolor="#fdf3d8", edgecolor="#22303f",
                               linewidth=0.6, zorder=2))
    ax.text(4.70, 6.15, "RIS, $N$ elements", ha="center", va="center", fontsize=FS)
    ax.text(6.45, 5.90, r"$e^{j\theta_n}$, $\theta_n \in \mathcal{F}$, $|\mathcal{F}| = 2^b$",
            ha="center", va="center", fontsize=FS - 0.4)
    ax.text(5.55, 4.65, "controller", ha="left", va="center", fontsize=FS - 0.6,
            color="#5a6675")
    arrow(ax, (4.95, 4.65), (5.45, 4.65), lw=0.6, color="#5a6675", style="-")

    # Bob and Eve
    ax.add_patch(Circle((8.75, 3.95), 0.28, facecolor="#e4f3e6",
                        edgecolor="#22303f", linewidth=0.8, zorder=2))
    ax.text(8.75, 3.45, "Bob\n(legitimate)", ha="center", va="top", fontsize=FS)
    ax.add_patch(Circle((8.75, 1.35), 0.28, facecolor="#f7e2e2",
                        edgecolor="#22303f", linewidth=0.8, zorder=2))
    ax.text(8.75, 0.85, "Eve\n(passive)", ha="center", va="top", fontsize=FS)

    # Links
    arrow(ax, (1.35, 3.30), (4.50, 4.25), ls="-", lw=1.0)
    ax.text(2.85, 4.05, "BS--RIS", fontsize=FS - 0.4, rotation=14)
    arrow(ax, (4.95, 4.55), (8.45, 4.05), ls="-", lw=1.0)
    ax.text(6.70, 4.62, "RIS--Bob", fontsize=FS - 0.4, rotation=-7)
    arrow(ax, (4.95, 3.70), (8.50, 1.40), ls=":", lw=0.8, color="#8a94a2")
    ax.text(6.60, 2.30, "RIS--Eve (weak)", fontsize=FS - 0.4, rotation=-25,
            color="#5a6675")
    arrow(ax, (1.30, 2.55), (8.50, 3.85), ls="--", lw=0.9)
    ax.add_patch(Rectangle((4.30, 2.86), 0.12, 0.50, facecolor="#9aa4b1",
                           edgecolor="#22303f", linewidth=0.5, zorder=4))
    ax.text(2.35, 3.32, "blocked", fontsize=FS - 0.6, rotation=8,
            color="#5a6675")
    ax.text(1.95, 2.32, "direct link (BS--Bob)", fontsize=FS - 0.4, rotation=8)
    arrow(ax, (1.30, 2.45), (8.50, 1.45), ls="--", lw=0.9)
    ax.text(3.20, 1.55, "Eve direct link", fontsize=FS - 0.4, rotation=-8)

    ax.text(0.05, 0.45,
            r"Eavesdropper CSI: $\mathbf{h}_e = \hat{\mathbf{h}}_e(\Theta) + \Delta$,"
            r"  $\|\Delta\|_2 \leq \delta_e$",
            fontsize=FS, va="center")


def draw_algorithm(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.05, 6.95, "(b) Alternating optimisation", fontsize=FS + 0.6,
            fontweight="bold")

    box(ax, 3.3, 6.25, 6.0, 0.72,
        "Inputs: channel estimates, $P$, $\\epsilon$, $b$, noise powers")
    box(ax, 3.3, 5.15, 6.0, 0.72,
        "Initialise $\\Theta^{(0)}$: coherent alignment (restart 1)\n"
        "or random phases (restarts $2..R$)")
    box(ax, 3.3, 3.95, 6.0, 0.78,
        "Update $\\mathbf{w}$: projected gradient ascent with\n"
        "backtracking on $\\|\\mathbf{w}\\|^2 = P$")
    box(ax, 3.3, 2.75, 6.0, 0.72,
        "Update $\\theta_n$, $n = 1..N$: exhaustive search over $\\mathcal{F}$")
    box(ax, 3.3, 1.55, 6.0, 0.62, "objective improved by more than $\\delta$?",
        fc="#fdf3d8")
    box(ax, 3.3, 0.45, 6.0, 0.66,
        "Output: $\\mathbf{w}^{\\star}$, $\\{\\theta_n^{\\star}\\}$ "
        "(best over $R$ restarts)", fc="#e4f3e6")

    arrow(ax, (3.3, 5.89), (3.3, 5.51))
    arrow(ax, (3.3, 4.79), (3.3, 4.34))
    arrow(ax, (3.3, 3.56), (3.3, 3.11))
    arrow(ax, (3.3, 2.39), (3.3, 1.86))
    arrow(ax, (3.3, 1.24), (3.3, 0.78))
    ax.text(3.42, 1.02, "yes", fontsize=FS - 0.6, va="center")

    # loop back from the convergence test to the beamforming update
    arrow(ax, (6.30, 1.55), (7.60, 1.55), style="-", lw=0.9)
    arrow(ax, (7.60, 1.55), (7.60, 3.95), style="-", lw=0.9)
    arrow(ax, (7.60, 3.95), (6.32, 3.95), style="-|>", lw=0.9)
    ax.text(7.72, 2.75, "no", fontsize=FS - 0.6, va="center", rotation=90)


def main() -> None:
    plt.style.use(str(STYLE))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    draw_system(axes[0])
    draw_algorithm(axes[1])
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005,
                        wspace=0.02)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = "system_model_and_algorithm_flow"
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"figures/{stem}.pdf")


if __name__ == "__main__":
    main()
