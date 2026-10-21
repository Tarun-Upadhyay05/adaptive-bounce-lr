"""Reproducible multi-seed campaigns.

A finished run is skipped when its configuration fingerprint matches and the
requested epoch budget is already complete. An interrupted run resumes from
``final.pt``. The official test loader is evaluated only after the epoch budget
is finished, using the checkpoint selected on validation accuracy.
"""

from __future__ import annotations

import random
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import build_cifar10_loaders
from src.evaluate import (
    aggregate_runs,
    evaluate_model,
    format_results_markdown,
    thresholds_from_history,
)
from src.models import resnet18_cifar
from src.scheduler import ALGORITHM_VERSION, build_scheduler
from src.train import run_training
from src.utils import capture_environment, config_fingerprint, load_json, save_json, select_device, set_seed


LogFn = Callable[[str], None]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} did not contain a mapping")
    config["config_path"] = str(path)
    _require(config, ("name", "seeds", "epochs", "batch_size", "schedulers", "optimizer", "model"))
    return config


def apply_overrides(
    config: dict[str, Any],
    *,
    epochs: int | None = None,
    seeds: list[int] | None = None,
    schedulers: list[str] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    if epochs is not None:
        updated["epochs"] = int(epochs)
    if seeds is not None:
        updated["seeds"] = [int(seed) for seed in seeds]
    if schedulers is not None:
        updated["schedulers"] = list(schedulers)
    if device is not None:
        updated["device"] = device
    return updated


def run_campaign(
    config: dict[str, Any],
    *,
    force: bool = False,
    plot: bool = True,
    log: LogFn | None = None,
) -> pd.DataFrame:
    """Run every scheduler and seed in ``config`` and write aggregate tables."""
    logger = log or print
    root = _project_root(config)
    results_root = root / config.get("results_dir", "results")
    campaign_dir = results_root / "runs" / config["name"]
    campaign_dir.mkdir(parents=True, exist_ok=True)
    save_json(campaign_dir / "environment.json", capture_environment())
    save_json(campaign_dir / "config.json", config)

    for scheduler_name in config["schedulers"]:
        for seed in config["seeds"]:
            logger(f"=== {config['name']} | {scheduler_name} | seed {seed} ===")
            run_single(config, scheduler_name, int(seed), force=force, log=logger)

    summaries = load_completed_summaries(results_root, config["name"])
    aggregate = write_aggregate(results_root, config["name"], summaries)
    markdown = format_results_markdown(aggregate, config_name=config["name"])
    (campaign_dir / "RESULTS.md").write_text(markdown + "\n", encoding="utf-8")
    logger(markdown)
    if plot:
        from src.plotting import plot_campaign

        figures_root = root / config.get("figures_dir", "figures")
        plot_campaign(results_root, figures_root, config["name"])
        logger(f"figures written to {figures_root / config['name']}")
    return aggregate


def run_single(
    config: dict[str, Any],
    scheduler_name: str,
    seed: int,
    *,
    force: bool = False,
    loaders: tuple[DataLoader, DataLoader, DataLoader, torch.Generator] | None = None,
    device: torch.device | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    """Train one scheduler and seed. Skip, resume, or start as needed."""
    logger = log or print
    root = _project_root(config)
    run_dir = root / config.get("results_dir", "results") / "runs" / config["name"] / scheduler_name / f"seed_{seed}"
    fingerprint = _fingerprint(config, scheduler_name, seed)
    summary_path = run_dir / "summary.json"
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    existing = _read_summary(summary_path)
    if existing and existing.get("completed") and existing.get("fingerprint") == fingerprint:
        if int(existing.get("epochs_completed", 0)) >= int(config["epochs"]):
            refreshed = _refresh_thresholds(existing, run_dir, config)
            save_json(summary_path, refreshed)
            logger(f"skipping completed run {run_dir}")
            return refreshed

    checkpoint_path = run_dir / "final.pt"
    resume_state = None
    if checkpoint_path.exists():
        resume_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if resume_state.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"{checkpoint_path} was produced by a different configuration. "
                "Pass force=True to overwrite it."
            )

    device = device or select_device(config.get("device", "auto"))
    set_seed(seed, deterministic=bool(config.get("deterministic", True)))
    if device.type == "cuda":
        torch.cuda.empty_cache()

    generator = torch.Generator()
    generator.manual_seed(seed)
    split_meta = None
    if loaders is None:
        train_loader, val_loader, test_loader, split_meta = build_cifar10_loaders(
            str(root / config.get("data_dir", "data")),
            batch_size=int(config["batch_size"]),
            val_fraction=float(config.get("val_fraction", 0.1)),
            split_seed=int(config.get("split_seed", 42)),
            loader_seed=seed,
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        split_path = run_dir.parents[1] / "split.json"
        if not split_path.exists():
            save_json(split_path, split_meta)
    else:
        train_loader, val_loader, test_loader, generator = loaders

    model = resnet18_cifar(num_classes=int(config.get("model", {}).get("num_classes", 10))).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        momentum=float(config["optimizer"]["momentum"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = build_scheduler(scheduler_name, optimizer, config, int(config["epochs"]))
    criterion = torch.nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    best: dict[str, Any] = {"elapsed_sec": 0.0}
    start_epoch = 1
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(_tensors_to_device(resume_state["optimizer"], device))
        if resume_state.get("scheduler") is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        history = list(resume_state.get("history", []))
        trace = list(resume_state.get("trace", []))
        best = {
            "elapsed_sec": float(resume_state.get("elapsed_sec", 0.0)),
            "val_accuracy": resume_state.get("best_val_accuracy"),
            "val_loss_at_selection": resume_state.get("best_val_loss_at_selection"),
            "selected_epoch": resume_state.get("selected_epoch"),
            "model_state": resume_state.get("best_model"),
            "best_val_loss": resume_state.get("best_val_loss"),
            "best_val_loss_epoch": resume_state.get("best_val_loss_epoch"),
            "scaler": resume_state.get("scaler"),
        }
        start_epoch = int(resume_state["epoch"]) + 1
        _restore_rng(resume_state.get("rng"))
        if resume_state.get("loader_generator") is not None:
            generator.set_state(resume_state["loader_generator"])
        logger(f"resuming at epoch {start_epoch}")

    save_json(
        run_dir / "run_config.json",
        {
            "fingerprint": fingerprint,
            "scheduler": scheduler_name,
            "seed": seed,
            "epochs": int(config["epochs"]),
            "algorithm": ALGORITHM_VERSION,
            "device": str(device),
            "started_at": _now(),
        },
    )

    def on_epoch_end(payload: dict[str, Any]) -> None:
        _write_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=payload["epoch"],
            history=payload["history"],
            trace=payload["trace"],
            best=payload["best"],
            fingerprint=fingerprint,
            generator=generator,
            scheduler_name=scheduler_name,
            seed=seed,
        )
        _write_history(run_dir / "history.csv", payload["history"])
        if payload["trace"]:
            _write_history(run_dir / "scheduler_trace.csv", payload["trace"])
        best_model = payload["best"].get("model_state")
        if best_model is not None:
            torch.save(
                {
                    "epoch": payload["best"].get("selected_epoch"),
                    "model": best_model,
                    "val_accuracy": payload["best"].get("val_accuracy"),
                    "val_loss": payload["best"].get("val_loss_at_selection"),
                    "scheduler": scheduler_name,
                    "seed": seed,
                    "synthetic": False,
                },
                run_dir / "best.pt",
            )

    if start_epoch <= int(config["epochs"]):
        run_training(
            model,
            optimizer,
            scheduler,
            scheduler_name,
            train_loader,
            val_loader,
            criterion,
            device,
            epochs=int(config["epochs"]),
            start_epoch=start_epoch,
            history=history,
            trace=trace,
            best=best,
            accuracy_thresholds=list(config.get("accuracy_thresholds", [])),
            use_amp=bool(config.get("amp", False)),
            max_train_batches=config.get("max_train_batches"),
            max_eval_batches=config.get("max_eval_batches"),
            on_epoch_end=on_epoch_end,
            log=lambda message: logger(f"[{scheduler_name} seed={seed}] {message}"),
        )

    best_path = run_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError(f"training finished without a selected checkpoint in {run_dir}")
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    test_loss, test_accuracy = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        max_batches=config.get("max_eval_batches"),
    )
    lr_changes = sum(1 for row in history if abs(row["lr_after"] - row["lr"]) > 1e-8)
    summary = {
        "completed": True,
        "synthetic": False,
        "algorithm": ALGORITHM_VERSION,
        "config_name": config["name"],
        "scheduler": scheduler_name,
        "seed": int(seed),
        "fingerprint": fingerprint,
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "epochs_requested": int(config["epochs"]),
        "best_val_accuracy": best.get("val_accuracy"),
        "best_val_accuracy_epoch": best.get("selected_epoch"),
        "best_val_loss": best.get("best_val_loss"),
        "best_val_loss_epoch": best.get("best_val_loss_epoch"),
        "selected_epoch": best.get("selected_epoch"),
        "selection_rule": "highest validation accuracy; ties broken by lower validation loss",
        "scheduler_signal": "validation loss only",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "total_time_sec": float(best.get("elapsed_sec", 0.0)),
        "num_lr_changes": int(lr_changes),
        "thresholds": thresholds_from_history(history, list(config.get("accuracy_thresholds", []))),
        "device": str(device),
        "finished_at": _now(),
        "split": split_meta,
    }
    if trace:
        summary["num_decay_events"] = sum(
            1 for row in trace if row.get("action") in {"decay_slow", "decay_stagnant"}
        )
        summary["num_recovery_events"] = sum(1 for row in trace if row.get("action") == "recover")
    save_json(summary_path, summary)
    logger(
        f"finished {scheduler_name} seed={seed} "
        f"test_acc={test_accuracy:.4f} best_val_acc={summary['best_val_accuracy']:.4f} "
        f"selected_epoch={summary['selected_epoch']}"
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def load_completed_summaries(results_root: Path, config_name: str) -> pd.DataFrame:
    paths = sorted((results_root / "runs" / config_name).glob("*/seed_*/summary.json"))
    rows = []
    for path in paths:
        payload = load_json(path)
        if payload.get("completed") and not payload.get("synthetic", False):
            payload["run_dir"] = str(path.parent)
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def write_aggregate(results_root: Path, config_name: str, summaries: pd.DataFrame | None = None) -> pd.DataFrame:
    if summaries is None:
        summaries = load_completed_summaries(results_root, config_name)
    campaign_dir = results_root / "runs" / config_name
    campaign_dir.mkdir(parents=True, exist_ok=True)
    if summaries.empty:
        return summaries
    export = summaries.drop(columns=["run_dir"], errors="ignore").copy()
    export["thresholds"] = export["thresholds"].apply(lambda value: str(value))
    export.to_csv(campaign_dir / "per_seed.csv", index=False)
    aggregate = aggregate_runs(summaries)
    aggregate.to_csv(campaign_dir / "aggregate.csv", index=False)
    return aggregate


def _write_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    history: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    best: dict[str, Any],
    fingerprint: str,
    generator: torch.Generator,
    scheduler_name: str,
    seed: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduler_name": scheduler_name,
            "seed": seed,
            "history": history,
            "trace": trace,
            "best_model": best.get("model_state"),
            "best_val_accuracy": best.get("val_accuracy"),
            "best_val_loss_at_selection": best.get("val_loss_at_selection"),
            "selected_epoch": best.get("selected_epoch"),
            "best_val_loss": best.get("best_val_loss"),
            "best_val_loss_epoch": best.get("best_val_loss_epoch"),
            "elapsed_sec": best.get("elapsed_sec", 0.0),
            "scaler": best.get("scaler"),
            "fingerprint": fingerprint,
            "loader_generator": generator.get_state(),
            "rng": _capture_rng(),
            "synthetic": False,
        },
        path,
    )


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _refresh_thresholds(summary: dict[str, Any], run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return summary
    history = pd.read_csv(history_path).to_dict(orient="records")
    summary = dict(summary)
    summary["thresholds"] = thresholds_from_history(history, list(config.get("accuracy_thresholds", [])))
    return summary


def _fingerprint(config: dict[str, Any], scheduler_name: str, seed: int) -> str:
    if scheduler_name in {"ablr", "ablr_recovery", "ablr_hybrid"}:
        settings = dict(config["ablr"])
        settings["enable_recovery"] = scheduler_name == "ablr_recovery"
        if scheduler_name == "ablr_hybrid":
            settings["envelope"] = "linear"
            settings["linear"] = dict(config["linear"])
    elif scheduler_name == "constant":
        settings = {}
    else:
        settings = dict(config[scheduler_name])
    payload = {
        "algorithm": ALGORITHM_VERSION,
        "scheduler": scheduler_name,
        "scheduler_settings": settings,
        "seed": int(seed),
        "epochs": int(config["epochs"]),
        "split_seed": int(config.get("split_seed", 42)),
        "batch_size": int(config["batch_size"]),
        "val_fraction": float(config.get("val_fraction", 0.1)),
        "optimizer": config["optimizer"],
        "model": config.get("model", {}),
        "amp": bool(config.get("amp", False)),
        "deterministic": bool(config.get("deterministic", True)),
        "num_workers": int(config.get("num_workers", 0)),
        "max_train_batches": config.get("max_train_batches"),
        "max_eval_batches": config.get("max_eval_batches"),
    }
    return config_fingerprint(payload)


def _project_root(config: dict[str, Any]) -> Path:
    configured = config.get("project_root")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1]


def _read_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def _require(config: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"config is missing {missing}")


def _tensors_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensors_to_device(item, device) for item in value)
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
