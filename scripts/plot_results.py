"""Draw summary figures from run.py reproduce output; never launches models."""

from pathlib import Path


def plot_results(report: dict, output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires: python -m pip install -r requirements-reproduce.txt") from exc

    output.mkdir(parents=True, exist_ok=True)
    colors = ["#4c78a8", "#72a4ce", "#8c8c8c", "#59a14f", "#e17c30"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for color, row in zip(colors, report["robustness"].values()):
            ax.plot(range(5), [100 * row["conditions"][c]["top1"]["mean"]
                              for c in ("C0", "C1", "C2", "C3", "C4")],
                    marker="o", color=color, label=row["name"])
        ax.set(xticks=range(5), xticklabels=["C0\nClean", "C1\nImports", "C2\nOther names",
                                             "C3\nPartial", "C4\nFull"],
               ylabel="Top-1 accuracy (%)", ylim=(0, 105), title="Identifier robustness")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"robustness.{suffix}", dpi=180)
        plt.close(fig)

        rows = report["geometry_selected"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
        for ax, group, metric, label, scale in (
            (axes[0], "zero_shot", "zs_top1", "Zero-shot Top-1 (%)", 100),
            (axes[1], "geometry", "sep_ratio", "Class separation ratio", 1),
        ):
            pairs = [row[group][metric] for row in rows.values()]
            ax.bar(list(rows), [p["mean"] * scale for p in pairs],
                   yerr=[p["std"] * scale for p in pairs], capsize=4, color=colors[:4])
            ax.set(ylabel=label)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Geometry ablation: mean ± population SD across five seeds")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"geometry.{suffix}", dpi=180)
        plt.close(fig)

        rows = {v: report["symbolic"][v] for v in ("S0", "S1", "S2", "S3", "S4")}
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
        for ax, metric, label in ((axes[0], "top1", "Top-1 accuracy (%)"),
                                  (axes[1], "total_viol_rate", "Logic-violation rate (%)")):
            pairs = [row["metrics"][metric] for row in rows.values()]
            ax.bar(list(rows), [100 * p["mean"] for p in pairs],
                   yerr=[100 * p["std"] for p in pairs], capsize=4, color=colors)
            ax.set(ylabel=label)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Symbolic controls: mean ± population SD across three seeds")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"symbolic.{suffix}", dpi=180)
        plt.close(fig)
