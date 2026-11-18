"""Generate the manuscript figures from the archived result tables.

Figures are written to ../figures as vector PDF plus a 300 dpi PNG preview.
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
STYLE = ROOT / "style" / "ieee.mplstyle"

LABELS = {
    "no_ris": "No RIS",
    "random_phase": "Random phase",
    "unaware": "Secrecy-unaware",
    "proposed": "Proposed (2-bit)",
    "continuous": "36-point phase grid",
}

STYLES = {
    "no_ris": dict(linestyle="--", marker="o"),
    "random_phase": dict(linestyle="-.", marker="s"),
    "unaware": dict(linestyle=":", marker="^"),
    "proposed": dict(linestyle="-", marker="D"),
    "continuous": dict(linestyle="-", marker="x"),
}


def _read(name: str) -> tuple[list[str], list[list[float]]]:
    with (RESULTS / name).open() as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], [[float(v) for v in r] for r in rows[1:]]
    return header, data


def _save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"figures/{stem}.pdf")


def _legend_above(ax, ncol: int = 3, fontsize: float = 6.8) -> None:
    """Place the legend in a single strip above the axes.

    Both the power sweep and the element sweep have their curves crossing the
    upper-left corner, so an in-axes legend always hides data. A horizontal
    strip above the axes keeps the plot area clear.
    """

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.12),
        mode="expand",
        ncol=ncol,
        borderaxespad=0.0,
        handlelength=1.7,
        handletextpad=0.5,
        columnspacing=1.0,
        fontsize=fontsize,
    )


def plot_power() -> None:
    header, data = _read("power_sweep.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    for i, scheme in enumerate(header[1:], start=1):
        ax.plot(
            [r[0] for r in data],
            [r[i] for r in data],
            label=LABELS.get(scheme, scheme),
            linewidth=1.2,
            **STYLES.get(scheme, {}),
        )
    ax.set_xlabel("Transmit power $P$ (dBm)")
    ax.set_ylabel("Worst-case secrecy rate (bit/s/Hz)")
    ax.set_ylim(bottom=0)
    _legend_above(ax)
    ax.grid(True)
    _save(fig, "secrecy_rate_vs_transmit_power")


def plot_elements() -> None:
    header, data = _read("elements_sweep.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    for i, scheme in enumerate(header[1:], start=1):
        ax.plot(
            [r[0] for r in data],
            [r[i] for r in data],
            label=LABELS.get(scheme, scheme),
            linewidth=1.2,
            **STYLES.get(scheme, {}),
        )
    ax.set_xlabel("Number of RIS elements $N$")
    ax.set_ylabel("Worst-case secrecy rate (bit/s/Hz)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([r[0] for r in data])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(bottom=0)
    _legend_above(ax)
    ax.grid(True)
    _save(fig, "secrecy_rate_vs_ris_elements")


def plot_epsilon() -> None:
    header, data = _read("csi_error_sweep.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, scheme in enumerate(header[1:], start=1):
        ax.plot(
            [r[0] for r in data],
            [r[i] for r in data],
            label=LABELS.get(scheme, scheme),
            linewidth=1.2,
            **STYLES.get(scheme, {}),
        )
    ax.set_xlabel(r"Reference-normalised CSI error $\epsilon$")
    ax.set_ylabel("Worst-case secrecy rate (bit/s/Hz)")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.6)
    ax.legend(loc="upper right")
    ax.grid(True)
    _save(fig, "secrecy_rate_vs_csi_error")


def plot_convergence() -> None:
    header, data = _read("convergence_trace.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot([r[0] for r in data], [r[1] for r in data], marker="o", linewidth=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Worst-case secrecy rate (bit/s/Hz)")
    ax.grid(True)
    _save(fig, "ao_convergence_trace")


def main() -> None:
    plt.style.use(str(STYLE))
    for fn in (plot_power, plot_elements, plot_epsilon, plot_convergence):
        path = RESULTS / {
            plot_power: "power_sweep.csv",
            plot_elements: "elements_sweep.csv",
            plot_epsilon: "csi_error_sweep.csv",
            plot_convergence: "convergence_trace.csv",
        }[fn]
        if path.exists():
            fn()
        else:
            print(f"skip {fn.__name__}: {path.name} not found")


if __name__ == "__main__":
    main()
