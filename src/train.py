"""Single-epoch training and the epoch loop.

The loop reads only the training and validation loaders. The official test
loader is intentionally absent so a scheduler, early decision, or checkpoint
selection cannot see it.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.evaluate import evaluate_model, thresholds_from_history
from src.scheduler import step_scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None,
    max_batches: int | None,
) -> tuple[float, float]:
    """Train for one pass over the loader.

    Returns mean cross-entropy and accuracy over the examples actually seen.
    A non-finite loss skips that update so one bad batch does not end the run.
    """
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            continue
        if scaler is None or not scaler.is_enabled():
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach()) * batch_size
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == targets).sum().item())
        total_seen += batch_size
    if total_seen == 0:
        return float("nan"), float("nan")
    return total_loss / total_seen, total_correct / total_seen


def improved_selection(
    val_accuracy: float,
    val_loss: float,
    best_accuracy: float | None,
    best_loss: float | None,
) -> bool:
    """Checkpoint rule: higher validation accuracy, then lower validation loss.

    The test set is not an input. Accuracy is the selection metric because the
    reported task is classification accuracy. Validation loss breaks ties.
    Scheduler updates are a separate decision and use validation loss only.
    """
    if best_accuracy is None or best_loss is None:
        return True
    if val_accuracy > best_accuracy + 1e-12:
        return True
    if abs(val_accuracy - best_accuracy) <= 1e-12 and val_loss < best_loss - 1e-12:
        return True
    return False


def run_training(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    scheduler_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    epochs: int,
    start_epoch: int,
    history: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    best: dict[str, Any],
    accuracy_thresholds: list[float],
    use_amp: bool,
    max_train_batches: int | None,
    max_eval_batches: int | None,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run epochs ``start_epoch`` through ``epochs`` inclusive, 1-based.

    ``best`` tracks the selected checkpoint and the best validation loss,
    which may fall on different epochs. ``on_epoch_end`` is how the experiment
    runner persists a resumable checkpoint without this function knowing paths.
    """
    scaler = None
    if use_amp and device.type == "cuda" and torch.cuda.is_available():
        scaler = torch.amp.GradScaler("cuda")
        if best.get("scaler"):
            scaler.load_state_dict(best["scaler"])

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        lr_used = float(optimizer.param_groups[0]["lr"])
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            max_batches=max_train_batches,
        )
        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            max_batches=max_eval_batches,
        )
        decision = step_scheduler(scheduler, scheduler_name, val_loss, val_accuracy)
        lr_after = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_start
        best["elapsed_sec"] = float(best.get("elapsed_sec", 0.0)) + epoch_seconds

        if improved_selection(
            val_accuracy,
            val_loss,
            best.get("val_accuracy"),
            best.get("val_loss_at_selection"),
        ):
            best["val_accuracy"] = float(val_accuracy)
            best["val_loss_at_selection"] = float(val_loss)
            best["selected_epoch"] = epoch
            best["model_state"] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if best.get("best_val_loss") is None or val_loss < best["best_val_loss"]:
            best["best_val_loss"] = float(val_loss)
            best["best_val_loss_epoch"] = epoch

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "lr": lr_used,
            "lr_after": lr_after,
            "epoch_seconds": float(epoch_seconds),
            "elapsed_sec": float(best["elapsed_sec"]),
        }
        history.append(row)
        if decision is not None:
            trace.append(decision)
        best["thresholds"] = thresholds_from_history(history, accuracy_thresholds)
        best["scaler"] = scaler.state_dict() if scaler is not None else None

        if log is not None:
            log(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
                f"lr_used={lr_used:.6g} lr_next={lr_after:.6g} "
                f"epoch_s={epoch_seconds:.1f}"
            )
        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "epoch": epoch,
                    "history": history,
                    "trace": trace,
                    "best": best,
                    "scaler": scaler,
                }
            )
    return best
