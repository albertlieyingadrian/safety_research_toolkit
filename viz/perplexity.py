"""
Perplexity log parsing and visualisation.
"""

import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

_PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
_MAX_COLS = 3
_HIST_BINS = 40
_PLOT_DPI = 150


def parse_perplexity_log(path: str) -> Dict:
    """Parse a perplexity log file into a structured dict.

    Returns::

        {
            dataset_name: {
                "examples": [(orig_idx, tokens, ppl), ...],
                "stats":    {key: value, ...},
                "agg_ppl":  float | None,
            }
        }
    """
    text = Path(path).read_text()
    datasets: Dict = {}

    blocks = re.split(r"=== Dataset: (.+?) ===", text)
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i + 1]

        stats: Dict = {}
        for m in re.finditer(r"^\s{2}(\w[\w ]+):\s*(.+)$", body, re.MULTILINE):
            k, v = m.group(1).strip(), m.group(2).strip()
            try:
                stats[k] = int(v) if "." not in v else float(v)
            except ValueError:
                stats[k] = v

        examples = [
            (int(m.group(1)), int(m.group(2)), float(m.group(3)))
            for m in re.finditer(r"orig_idx=(\d+)\s+tokens=(\d+)\s+ppl=([\d.]+)", body)
        ]

        agg_m = re.search(rf"{re.escape(name)}\s+PPL=([\d.]+)", text)
        agg_ppl = float(agg_m.group(1)) if agg_m else None

        datasets[name] = {"examples": examples, "stats": stats, "agg_ppl": agg_ppl}

    return datasets


def plot_perplexity_log(
    log_path: str,
    output_dir: str = ".",
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """Plot PPL histograms from a perplexity log file.

    Saves ``<output_dir>/perplexity_summary.png``.

    Args:
        log_path: Path to a log file parseable by :func:`parse_perplexity_log`.
        output_dir: Directory to save the figure.
        figsize: Figure size in inches.
    """
    datasets = parse_perplexity_log(log_path)
    if not datasets:
        return

    n = len(datasets)
    cols = min(n, _MAX_COLS)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for ax_idx, (name, data) in enumerate(datasets.items()):
        ax = axes[ax_idx // cols][ax_idx % cols]
        ppls = [e[2] for e in data["examples"]]
        if not ppls:
            ax.set_visible(False)
            continue
        color = _PALETTE[ax_idx % len(_PALETTE)]
        ax.hist(ppls, bins=_HIST_BINS, color=color, alpha=0.7, edgecolor="white")
        agg = data["agg_ppl"]
        ax.set_title(name if agg is None else f"{name}\nPPL={agg:.3f}", fontsize=10)
        ax.set_xlabel("PPL")
        ax.set_ylabel("Count")

    for ax_idx in range(n, rows * cols):
        axes[ax_idx // cols][ax_idx % cols].set_visible(False)

    plt.tight_layout()
    out = Path(output_dir) / "perplexity_summary.png"
    plt.savefig(out, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
