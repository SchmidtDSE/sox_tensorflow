#!/usr/bin/env python3
"""
License: BSD 3-Clause

Plot a before/after parity figure from two compare-spectrograms + compare-model runs.

Each run directory must contain ``results/*.csv``, ``spectrograms/*.tf.png`` and a
``versions.json`` (written by ``parity_check.sh``). The figure mirrors the README
brightness-decile plot, with the two runs side by side, plus per-class model diffs.
The subtitle reports the direct run-vs-run check: differing tf pixels and whether
the result CSVs are byte-identical.

Usage::

    python plot_parity.py tmp/parity/base tmp/parity/head --figure parity.png
"""

import json
import platform
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RESULT_CSVS = (
    "overall_comparison.csv",
    "percentile_comparison.csv",
    "model_comparison.csv",
    "model_class_diffs.csv",
)
COLORS = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dc", "#fcfcfb"
BAR_WIDTH, BAR_GAP = 0.38, 0.02


def _label(run: Path) -> str:
    v = json.loads((run / "versions.json").read_text())
    return f"{v['label']}  (numpy {v['numpy']} · TF {v['tensorflow']})"


def _pixel_mismatches(base: Path, head: Path) -> int:
    """Count differing pixels across all matching .tf.png files."""
    total = 0
    base_pngs = sorted((base / "spectrograms").glob("*.tf.png"))
    if not base_pngs:
        raise click.ClickException(f"no .tf.png files in {base / 'spectrograms'}")
    for png in base_pngs:
        other = head / "spectrograms" / png.name
        if not other.exists():
            raise click.ClickException(f"{other} missing; runs used different segments")
        a = np.array(Image.open(png).convert("L"))
        b = np.array(Image.open(other).convert("L"))
        total += int((a != b).sum())
    return total


def _paired_bars(ax, labels, series, fmt) -> None:
    x = np.arange(len(labels))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - 0.5) * (BAR_WIDTH + BAR_GAP), vals, BAR_WIDTH, color=COLORS[i], label=name)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("base", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("head", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--figure", "-f", default="parity.png", type=click.Path(path_type=Path),
              help="Output PNG path.")
@click.option("--top-classes", default=6, show_default=True,
              help="Number of model classes (largest mean |Δprob|) to plot.")
def main(base: Path, head: Path, figure: Path, top_classes: int) -> None:
    """Plot parity between two comparison runs, BASE (before) and HEAD (after)."""
    runs = {_label(base): base, _label(head): head}
    load = lambda run, name: pd.read_csv(run / "results" / name)
    pct = {k: load(v, "percentile_comparison.csv") for k, v in runs.items()}
    cls = {k: load(v, "model_class_diffs.csv") for k, v in runs.items()}
    overall = load(base, "overall_comparison.csv")

    mismatches = _pixel_mismatches(base, head)
    csvs_identical = all(
        (base / "results" / f).read_bytes() == (head / "results" / f).read_bytes() for f in RESULT_CSVS
    )
    deciles = list(dict.fromkeys(pct[_label(base)]["percentile_range"]))

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})

    by_decile = lambda col: {
        k: df.groupby("percentile_range", sort=False)[col].mean().reindex(deciles) for k, df in pct.items()
    }
    _paired_bars(axes[0], deciles, by_decile("accuracy"), lambda v, _: f"{v:.1f}%")
    axes[0].set_ylim(99.0, 100.0)
    axes[0].set_title("Exact pixel match vs sox, by brightness decile", color=INK)
    axes[0].set_ylabel("% of pixels in decile")

    _paired_bars(axes[1], deciles, by_decile("mean_abs_diff"), lambda v, _: f"{v:.3f}")
    axes[1].set_title("Mean |Δ| vs sox, by brightness decile", color=INK)
    axes[1].set_ylabel("Mean |Δ| (pixel values 0–255)")

    per_class = {k: df.groupby("class_code")["abs_diff"].mean() for k, df in cls.items()}
    top = per_class[_label(base)].sort_values(ascending=False).index[:top_classes]
    _paired_bars(axes[2], list(top), {k: v.reindex(top) for k, v in per_class.items()},
                 lambda v, _: f"{v:.0e}" if v else "0")
    axes[2].set_title(f"Model: mean |Δprob| sox vs tf (top {top_classes} classes)", color=INK)
    axes[2].set_ylabel("Mean |Δ probability|")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.005, 0.89))
    fig.suptitle(f"sox_tensorflow parity — {platform.system().lower()}-{platform.machine()}",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=0.99)
    fig.text(
        0.01, 0.905,
        f"{len(overall)} segments via compare-spectrograms + compare-model.  "
        f"before vs after: {mismatches:,} of {int(overall['total_pixels'].sum()):,} tf pixels differ · "
        + ("all result CSVs byte-identical" if csvs_identical else "result CSVs DIFFER"),
        color=MUTED, fontsize=10.5, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=150)

    click.echo(f"differing tf pixels : {mismatches:,}")
    click.echo(f"result CSVs         : {'identical' if csvs_identical else 'DIFFER'}")
    click.echo(f"figure              : {figure}")


if __name__ == "__main__":
    main()
