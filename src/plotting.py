"""Publication-style figures for scheduler comparisons.

Plots describe completed runs that are already on disk. They do not synthesize
missing seeds or invent a time-to-target for a threshold that was not reached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import SCHEDULER_LABELS, SCHEDULER_ORDER, load_json

# Okabe-Ito, colorblind-friendly and consistent across every figure.
PALETTE = {
    "constant": "#000000",
    "step": "#0072B2",
    "cosine": "#009E73",
    "linear": "#56B4E9",
    "plateau": "#E69F00",
    "ablr": "#D55E00",
    "ablr_recovery": "#CC79A7",
    "ablr_hybrid": "#332288",
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_campaign(results_root: Path, figures_root: Path, config_name: str) -> list[Path]:
    """Write the comparison figures for one named campaign."""
    apply_style()
    campaign = Path(results_root) / "runs" / config_name
    histories = _load_histories(campaign)
    if not histories:
        raise FileNotFoundError(f"no history.csv files under {campaign}")
    output = Path(figures_root) / config_name
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(histories, ignore_index=True)
    summaries = _load_summaries(campaign)
    written: list[Path] = []
    written.append(_plot_metric_grid(frame, output / "loss_curves", "loss", "Loss", config_name))
    written.append(_plot_metric_grid(frame, output / "accuracy_curves", "accuracy", "Accuracy", config_name))
    written.append(_plot_learning_rates(frame, output / "learning_rates", config_name))
    written.extend(_plot_overlays(frame, output, config_name))
    if not summaries.empty:
        written.append(_plot_bar(summaries, output / "best_val_accuracy", "best_val_accuracy", "Best validation accuracy", config_name))
        written.append(_plot_bar(summaries, output / "test_accuracy", "test_accuracy", "Test accuracy", config_name))
        written.append(_plot_per_seed(summaries, output / "per_seed_test_accuracy", config_name))
        written.append(_plot_time_to_target(summaries, output / "time_to_target", config_name))
        written.append(_plot_ablation(summaries, output / "ablation_recovery", config_name))
    trace_path = _representative_trace(campaign, "ablr")
    if trace_path is not None:
        trace = pd.read_csv(trace_path)
        written.append(
            plot_decision_timeline(
                trace,
                output / "ablr_decision_timeline",
                title=f"ABLR decisions ({config_name}, {_seed_label(trace_path)})",
            )
        )
    recovery_trace = _representative_trace(campaign, "ablr_recovery")
    if recovery_trace is not None:
        trace = pd.read_csv(recovery_trace)
        written.append(
            plot_decision_timeline(
                trace,
                output / "ablr_recovery_decision_timeline",
                title=f"ABLR + recovery decisions ({config_name}, {_seed_label(recovery_trace)})",
            )
        )
    return written


def plot_decision_timeline(trace: pd.DataFrame, output: Path, *, title: str) -> Path:
    """Velocity, regime, and learning rate for one ABLR trace.

    ``trace`` can come from a real run or from a synthetic metric sequence.
    Callers that use synthetic data must say so in ``title``.
    """
    apply_style()
    frame = trace.copy()
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True, constrained_layout=True)
    epochs = frame["epoch"]
    axes[0].plot(epochs, frame["normalized_progress"], color="#0072B2", label="Normalized progress")
    if "reference_velocity" in frame.columns:
        axes[0].plot(epochs, frame["velocity"], color="#009E73", alpha=0.8, label="Relative improvement")
    axes[0].axhline(0.0, color="#666666", linewidth=0.8)
    axes[0].set_ylabel("Progress")
    axes[0].legend(loc="best")

    regimes = frame["regime"].astype(str)
    codes, labels = pd.factorize(regimes, sort=True)
    axes[1].step(epochs, codes, where="mid", color="#D55E00")
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(list(labels))
    axes[1].set_ylabel("Regime")

    axes[2].plot(epochs, frame["lr"], color="#000000", label="Learning rate after the epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")

    if "lr_changed" in frame.columns:
        changed = frame["lr_changed"].map(lambda value: str(value).strip().lower() in {"true", "1"})
        changes = frame.loc[changed]
        for _, change in changes.iterrows():
            for axis in axes:
                axis.axvline(change["epoch"], color="#999999", linewidth=0.8, linestyle="--")
        if len(changes) <= 12:
            for _, change in changes.iterrows():
                axes[2].annotate(
                    str(change.get("reason", "")),
                    (change["epoch"], change["lr"]),
                    textcoords="offset points",
                    xytext=(4, 6),
                    fontsize=7,
                    rotation=30,
                )
    fig.suptitle(title)
    return _save(fig, output, frame)


def plot_class_examples(
    examples: list[tuple[Any, int]],
    class_names: tuple[str, ...],
    output: Path,
) -> Path:
    """Save a one-image-per-class grid. Images are unnormalized float tensors."""
    apply_style()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4), constrained_layout=True)
    for axis, (image, label) in zip(axes.ravel(), examples):
        array = np.transpose(np.asarray(image), (1, 2, 0))
        axis.imshow(np.clip(array, 0.0, 1.0))
        axis.set_title(class_names[label])
        axis.axis("off")
    fig.suptitle("CIFAR-10 examples (one per class, official training split)")
    return _save(fig, output, pd.DataFrame({"class": [class_names[label] for _, label in examples]}))


def _plot_metric_grid(frame: pd.DataFrame, output: Path, suffix: str, ylabel: str, config_name: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, constrained_layout=True)
    for axis, split, title in (
        (axes[0], "train", f"Training {ylabel.lower()}"),
        (axes[1], "val", f"Validation {ylabel.lower()}"),
    ):
        column = f"{split}_{'loss' if suffix == 'loss' else 'accuracy'}"
        for scheduler_name, group in _scheduler_groups(frame):
            mean, std = _mean_std_by_epoch(group, column)
            color = PALETTE.get(scheduler_name, "#333333")
            axis.plot(mean.index, mean.values, color=color, label=SCHEDULER_LABELS.get(scheduler_name, scheduler_name))
            if std is not None:
                axis.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle(f"{config_name}: mean across completed seeds")
    return _save(fig, output, frame)


def _plot_learning_rates(frame: pd.DataFrame, output: Path, config_name: str) -> Path:
    fig, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    for scheduler_name, group in _scheduler_groups(frame):
        mean, std = _mean_std_by_epoch(group, "lr")
        color = PALETTE.get(scheduler_name, "#333333")
        axis.plot(mean.index, mean.values, color=color, label=SCHEDULER_LABELS.get(scheduler_name, scheduler_name))
        if std is not None:
            lower = np.clip(mean - std, a_min=1e-8, a_max=None)
            axis.fill_between(mean.index, lower, mean + std, color=color, alpha=0.12)
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Learning rate used during the epoch")
    axis.set_title(f"{config_name}: learning-rate trajectories")
    axis.legend(loc="best", frameon=True)
    return _save(fig, output, frame)


def _plot_overlays(frame: pd.DataFrame, output: Path, config_name: str) -> list[Path]:
    paths = []
    for scheduler_name, group in _scheduler_groups(frame):
        seed = int(group["seed"].min())
        one = group.loc[group["seed"] == seed].sort_values("epoch")
        fig, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
        axis.plot(one["epoch"], one["val_loss"], color="#0072B2", label="Validation loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation loss")
        twin = axis.twinx()
        twin.plot(one["epoch"], one["lr"], color="#D55E00", label="Learning rate used")
        twin.set_ylabel("Learning rate")
        twin.set_yscale("log")
        changes = one.loc[(one["lr_after"] - one["lr"]).abs() > 1e-8]
        for _, change in changes.iterrows():
            axis.axvline(change["epoch"], color="#999999", linewidth=0.8, linestyle="--")
        axis.set_title(
            f"{config_name}: {SCHEDULER_LABELS.get(scheduler_name, scheduler_name)} "
            f"validation loss and LR (seed {seed})"
        )
        handles, labels = axis.get_legend_handles_labels()
        handles2, labels2 = twin.get_legend_handles_labels()
        axis.legend(handles + handles2, labels + labels2, loc="best")
        paths.append(_save(fig, output / f"overlay_{scheduler_name}", one))
    return paths


def _plot_bar(summaries: pd.DataFrame, output: Path, column: str, ylabel: str, config_name: str) -> Path:
    stats = _seed_stats(summaries, column)
    fig, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    colors = [PALETTE.get(name, "#333333") for name in stats["scheduler"]]
    yerr = stats["std"].fillna(0.0).to_numpy()
    axis.bar(stats["label"], stats["mean"], color=colors, yerr=yerr, capsize=4)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Scheduler")
    axis.set_title(f"{config_name}: {ylabel}")
    axis.tick_params(axis="x", rotation=20)
    return _save(fig, output, stats)


def _plot_per_seed(summaries: pd.DataFrame, output: Path, config_name: str) -> Path:
    fig, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    plot_frame = summaries.copy()
    plot_frame["label"] = plot_frame["scheduler"].map(lambda name: SCHEDULER_LABELS.get(name, name))
    order = [SCHEDULER_LABELS.get(name, name) for name in _present(plot_frame["scheduler"])]
    sns.stripplot(
        data=plot_frame,
        x="label",
        y="test_accuracy",
        hue="label",
        order=order,
        hue_order=order,
        palette={SCHEDULER_LABELS.get(name, name): PALETTE.get(name, "#333333") for name in _present(summaries["scheduler"])},
        ax=axis,
        size=8,
        jitter=0.15,
        legend=False,
    )
    axis.set_xlabel("Scheduler")
    axis.set_ylabel("Test accuracy")
    axis.set_title(f"{config_name}: test accuracy of each seed")
    axis.tick_params(axis="x", rotation=20)
    return _save(fig, output, plot_frame[["scheduler", "seed", "test_accuracy"]])


def _plot_time_to_target(summaries: pd.DataFrame, output: Path, config_name: str) -> Path:
    records = []
    for _, summary in summaries.iterrows():
        mapping = summary["thresholds"]
        if not isinstance(mapping, dict):
            continue
        for key, entry in mapping.items():
            records.append(
                {
                    "scheduler": summary["scheduler"],
                    "label": SCHEDULER_LABELS.get(summary["scheduler"], summary["scheduler"]),
                    "seed": summary["seed"],
                    "threshold": key,
                    "reached": bool(entry.get("reached")),
                    "elapsed_sec": entry.get("elapsed_sec") if entry.get("reached") else np.nan,
                    "epoch": entry.get("epoch") if entry.get("reached") else np.nan,
                }
            )
    frame = pd.DataFrame(records)
    fig, axis = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    if frame.empty:
        axis.text(0.5, 0.5, "No threshold records", ha="center")
    else:
        reached = frame.loc[frame["reached"]].copy()
        if reached.empty:
            axis.text(0.5, 0.5, "No seed reached a configured accuracy threshold", ha="center", va="center")
            axis.set_axis_off()
        else:
            sns.barplot(
                data=reached,
                x="threshold",
                y="elapsed_sec",
                hue="label",
                palette={SCHEDULER_LABELS.get(name, name): PALETTE.get(name, "#333333") for name in _present(summaries["scheduler"])},
                errorbar="sd",
                ax=axis,
            )
            axis.set_xlabel("Validation accuracy threshold")
            axis.set_ylabel("Wall time to first reach (s)")
            axis.legend(title="Scheduler", frameon=True)
            counts = frame.groupby(["threshold", "label"])["reached"].mean()
            axis.set_title(f"{config_name}: time to target, seeds that reached it only")
            del counts
    return _save(fig, output, frame)


def _plot_ablation(summaries: pd.DataFrame, output: Path, config_name: str) -> Path:
    pair = summaries.loc[summaries["scheduler"].isin(["ablr", "ablr_recovery"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    if pair.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "ABLR ablation runs are not on disk", ha="center")
            axis.set_axis_off()
        return _save(fig, output, pair)
    for axis, column, title in (
        (axes[0], "best_val_accuracy", "Best validation accuracy"),
        (axes[1], "test_accuracy", "Test accuracy"),
    ):
        stats = _seed_stats(pair, column)
        colors = [PALETTE.get(name, "#333333") for name in stats["scheduler"]]
        axis.bar(stats["label"], stats["mean"], color=colors, yerr=stats["std"].fillna(0.0), capsize=4)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle(f"{config_name}: recovery ablation")
    return _save(fig, output, pair[["scheduler", "seed", "best_val_accuracy", "test_accuracy"]])


def _load_histories(campaign: Path) -> list[pd.DataFrame]:
    frames = []
    for path in sorted(campaign.glob("*/seed_*/history.csv")):
        frame = pd.read_csv(path)
        frame["scheduler"] = path.parents[1].name
        frame["seed"] = int(path.parent.name.split("_")[-1])
        frames.append(frame)
    return frames


def _load_summaries(campaign: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(campaign.glob("*/seed_*/summary.json")):
        payload = load_json(path)
        if payload.get("completed") and not payload.get("synthetic", False):
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _representative_trace(campaign: Path, scheduler_name: str) -> Path | None:
    paths = sorted((campaign / scheduler_name).glob("seed_*/scheduler_trace.csv"))
    return paths[0] if paths else None


def _seed_label(path: Path) -> str:
    return path.parent.name.replace("_", " ")


def _scheduler_groups(frame: pd.DataFrame):
    for name in _present(frame["scheduler"]):
        yield name, frame.loc[frame["scheduler"] == name]


def _present(series: pd.Series) -> list[str]:
    found = set(series.astype(str))
    ordered = [name for name in SCHEDULER_ORDER if name in found]
    return ordered + sorted(found - set(ordered))


def _mean_std_by_epoch(group: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series | None]:
    pivot = group.groupby("epoch")[column]
    mean = pivot.mean()
    if group["seed"].nunique() < 2:
        return mean, None
    return mean, pivot.std(ddof=1)


def _seed_stats(summaries: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for name in _present(summaries["scheduler"]):
        values = pd.to_numeric(summaries.loc[summaries["scheduler"] == name, column], errors="coerce").dropna()
        std = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        rows.append(
            {
                "scheduler": name,
                "label": SCHEDULER_LABELS.get(name, name),
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": std,
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, output: Path, data: pd.DataFrame) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    data.to_csv(output.with_suffix(".csv"), index=False)
    return output.with_suffix(".png")
