"""Validation and test evaluation, plus campaign aggregation.

Aggregation never fills in a threshold time for a seed that did not reach the
target. Means of time-to-target are computed only over the seeds that reached
it, and the count of those seeds is reported beside the mean.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import SCHEDULER_ORDER


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Return mean loss and accuracy.

    Metrics are computed in float32. Training may use automatic mixed
    precision, but scheduler decisions and model selection use this full
    precision evaluation so autocast rounding is not part of the signal.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_size = int(targets.shape[0])
            total_loss += float(loss) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_seen += batch_size
    if total_seen == 0:
        return float("nan"), float("nan")
    return total_loss / total_seen, total_correct / total_seen


def thresholds_from_history(
    history: list[dict[str, Any]],
    thresholds: list[float],
) -> dict[str, dict[str, Any]]:
    """First validation epoch at which each accuracy threshold is reached.

    A threshold that never happens is ``reached: false`` with null epoch and
    time. It is not replaced with the final epoch.
    """
    report: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        hit = None
        for row in history:
            accuracy = row.get("val_accuracy")
            if accuracy is not None and float(accuracy) >= float(threshold):
                hit = row
                break
        if hit is None:
            report[_threshold_key(threshold)] = {
                "reached": False,
                "epoch": None,
                "elapsed_sec": None,
            }
        else:
            report[_threshold_key(threshold)] = {
                "reached": True,
                "epoch": int(hit["epoch"]),
                "elapsed_sec": float(hit["elapsed_sec"]),
            }
    return report


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.4f}".rstrip("0").rstrip(".")


def mean_std(values: pd.Series) -> tuple[float | None, float | None, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    count = int(clean.shape[0])
    if count == 0:
        return None, None, 0
    mean = float(clean.mean())
    if count < 2:
        return mean, None, count
    return mean, float(clean.std(ddof=1)), count


def aggregate_runs(summaries: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation across seeds for one campaign."""
    if summaries.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    threshold_keys = sorted(
        {
            key
            for mapping in summaries["thresholds"]
            if isinstance(mapping, dict)
            for key in mapping
        }
    )
    names = [name for name in SCHEDULER_ORDER if name in set(summaries["scheduler"])]
    extra = sorted(set(summaries["scheduler"]) - set(names))
    for scheduler_name in names + extra:
        group = summaries.loc[summaries["scheduler"] == scheduler_name]
        test_mean, test_std, n_seeds = mean_std(group["test_accuracy"])
        test_loss_mean, test_loss_std, _ = mean_std(group["test_loss"])
        val_acc_mean, val_acc_std, _ = mean_std(group["best_val_accuracy"])
        val_loss_mean, val_loss_std, _ = mean_std(group["best_val_loss"])
        time_mean, time_std, _ = mean_std(group["total_time_sec"])
        changes_mean, changes_std, _ = mean_std(group["num_lr_changes"])
        row: dict[str, Any] = {
            "scheduler": scheduler_name,
            "n_seeds": n_seeds,
            "epochs": int(group["epochs_completed"].max()),
            "test_accuracy_mean": test_mean,
            "test_accuracy_std": test_std,
            "test_loss_mean": test_loss_mean,
            "test_loss_std": test_loss_std,
            "best_val_accuracy_mean": val_acc_mean,
            "best_val_accuracy_std": val_acc_std,
            "best_val_loss_mean": val_loss_mean,
            "best_val_loss_std": val_loss_std,
            "total_time_sec_mean": time_mean,
            "total_time_sec_std": time_std,
            "num_lr_changes_mean": changes_mean,
            "num_lr_changes_std": changes_std,
        }
        for key in threshold_keys:
            reached_epochs = []
            reached_times = []
            for mapping in group["thresholds"]:
                entry = mapping.get(key) if isinstance(mapping, dict) else None
                if entry and entry.get("reached"):
                    reached_epochs.append(entry["epoch"])
                    reached_times.append(entry["elapsed_sec"])
            row[f"threshold_{key}_n_reached"] = len(reached_epochs)
            row[f"threshold_{key}_n_seeds"] = int(len(group))
            row[f"threshold_{key}_epoch_mean"] = (
                float(sum(reached_epochs) / len(reached_epochs)) if reached_epochs else None
            )
            row[f"threshold_{key}_time_sec_mean"] = (
                float(sum(reached_times) / len(reached_times)) if reached_times else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def format_results_markdown(aggregate: pd.DataFrame, *, config_name: str) -> str:
    """Markdown table of completed runs. Empty input does not invent numbers."""
    if aggregate.empty:
        return (
            f"No completed `{config_name}` runs were found. "
            "This section is left blank rather than filled with example numbers."
        )
    header = (
        "| Scheduler | Seeds | Best val accuracy | Test accuracy | Test loss | "
        "Time (s) | LR changes |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [
        f"Results below are computed from completed `{config_name}` runs only.",
        "",
        header,
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            "| {scheduler} | {n_seeds} | {val} | {test} | {loss} | {time} | {changes} |".format(
                scheduler=row["scheduler"],
                n_seeds=int(row["n_seeds"]),
                val=_mean_std(row["best_val_accuracy_mean"], row["best_val_accuracy_std"]),
                test=_mean_std(row["test_accuracy_mean"], row["test_accuracy_std"]),
                loss=_mean_std(row["test_loss_mean"], row["test_loss_std"]),
                time=_mean_std(row["total_time_sec_mean"], row["total_time_sec_std"], digits=1),
                changes=_mean_std(row["num_lr_changes_mean"], row["num_lr_changes_std"], digits=2),
            )
        )
    threshold_columns = [column for column in aggregate.columns if column.endswith("_n_reached")]
    if threshold_columns:
        lines.extend(["", "Validation accuracy thresholds. Time is the mean only over seeds that reached the target.", ""])
        lines.append("| Scheduler | Threshold | Reached | Epochs (reached seeds) | Wall time s (reached seeds) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for _, row in aggregate.iterrows():
            for column in threshold_columns:
                key = column[len("threshold_") : -len("_n_reached")]
                n_reached = int(row[column])
                n_seeds = int(row[f"threshold_{key}_n_seeds"])
                epoch_mean = row[f"threshold_{key}_epoch_mean"]
                time_mean = row[f"threshold_{key}_time_sec_mean"]
                lines.append(
                    f"| {row['scheduler']} | {key} | {n_reached}/{n_seeds} | "
                    f"{_number(epoch_mean, 2)} | {_number(time_mean, 1)} |"
                )
    lines.extend(
        [
            "",
            "A standard deviation is reported only when at least two seeds finished. "
            "A single seed is a point estimate, not a distribution.",
            "Cosine annealing changes the learning rate every epoch by design, so its "
            "LR-change count is not a stability comparison against event-driven schedules.",
        ]
    )
    return "\n".join(lines)


def _mean_std(mean: Any, std: Any, digits: int = 4) -> str:
    if mean is None or (isinstance(mean, float) and mean != mean):
        return "n/a"
    if std is None or (isinstance(std, float) and std != std):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def _number(value: Any, digits: int) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "not reached"
    return f"{float(value):.{digits}f}"
