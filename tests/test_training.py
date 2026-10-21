"""Training-pipeline tests on synthetic batches, not CIFAR-10 results."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import stratified_indices
from src.evaluate import aggregate_runs, thresholds_from_history
from src.experiment import run_single
from src.models import count_trainable_parameters, resnet18_cifar
from src.scheduler import SCHEDULER_NAMES
from src.train import run_training


class CountingLoader:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield from self.loader

    def __len__(self) -> int:
        return len(self.loader)


def test_resnet18_stem_is_adapted_for_cifar() -> None:
    model = resnet18_cifar()
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)
    assert count_trainable_parameters(model) > 1_000_000


def test_stratified_split_is_balanced_and_disjoint() -> None:
    targets = np_targets()
    train_idx, val_idx = stratified_indices(targets, val_fraction=0.1, seed=7)
    assert len(np_intersect(train_idx, val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(targets)
    train_counts = torch.bincount(torch.tensor(targets[train_idx]), minlength=10)
    val_counts = torch.bincount(torch.tensor(targets[val_idx]), minlength=10)
    assert torch.all(val_counts == 10)
    assert torch.all(train_counts == 90)


def test_unreached_thresholds_stay_unreached() -> None:
    history = [
        {"epoch": 1, "val_accuracy": 0.2, "elapsed_sec": 3.0},
        {"epoch": 2, "val_accuracy": 0.4, "elapsed_sec": 6.0},
    ]
    report = thresholds_from_history(history, [0.4, 0.9])
    assert report["0.4"]["reached"] is True
    assert report["0.4"]["epoch"] == 2
    assert report["0.9"]["reached"] is False
    assert report["0.9"]["epoch"] is None
    assert report["0.9"]["elapsed_sec"] is None


def test_aggregate_does_not_impute_unreached_targets() -> None:
    summaries = pd.DataFrame(
        [
            {
                "scheduler": "constant",
                "epochs_completed": 3,
                "test_accuracy": 0.4,
                "test_loss": 1.5,
                "best_val_accuracy": 0.45,
                "best_val_loss": 1.4,
                "total_time_sec": 12.0,
                "num_lr_changes": 0,
                "thresholds": {"0.9": {"reached": False, "epoch": None, "elapsed_sec": None}},
            },
            {
                "scheduler": "constant",
                "epochs_completed": 3,
                "test_accuracy": 0.5,
                "test_loss": 1.2,
                "best_val_accuracy": 0.55,
                "best_val_loss": 1.1,
                "total_time_sec": 13.0,
                "num_lr_changes": 0,
                "thresholds": {"0.9": {"reached": True, "epoch": 3, "elapsed_sec": 13.0}},
            },
        ]
    )
    aggregate = aggregate_runs(summaries)
    row = aggregate.iloc[0]
    assert int(row["threshold_0.9_n_reached"]) == 1
    assert row["threshold_0.9_epoch_mean"] == pytest.approx(3.0)
    assert row["test_accuracy_std"] == pytest.approx(0.070710678, rel=1e-4)


def test_training_loop_does_not_accept_a_test_loader() -> None:
    parameters = inspect.signature(run_training).parameters
    assert "test_loader" not in parameters


@pytest.mark.parametrize("scheduler_name", SCHEDULER_NAMES)
def test_each_scheduler_trains_and_writes_checkpoints(tmp_path, scheduler_name: str) -> None:
    test_loader = CountingLoader(_synthetic_loader())
    summary = run_single(
        _config(tmp_path, epochs=1, schedulers=[scheduler_name]),
        scheduler_name,
        seed=0,
        loaders=(_synthetic_loader(), _synthetic_loader(), test_loader, torch.Generator()),
        device=torch.device("cpu"),
    )
    run_dir = tmp_path / "results" / "runs" / "unit" / scheduler_name / "seed_0"
    assert summary["completed"] is True
    assert summary["synthetic"] is False
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "final.pt").exists()
    history = pd.read_csv(run_dir / "history.csv")
    assert list(history["epoch"]) == [1]
    assert {"train_loss", "val_loss", "train_accuracy", "val_accuracy", "lr", "elapsed_sec"} <= set(history.columns)
    assert test_loader.iterations == 1
    assert history["val_accuracy"].iloc[0] <= 1.0
    repeated = run_single(
        _config(tmp_path, epochs=1, schedulers=[scheduler_name]),
        scheduler_name,
        seed=0,
        loaders=(_synthetic_loader(), _synthetic_loader(), test_loader, torch.Generator()),
        device=torch.device("cpu"),
    )
    assert repeated["finished_at"] == summary["finished_at"]
    assert test_loader.iterations == 1


def test_interrupted_run_resumes_and_evaluates_test_once(tmp_path, monkeypatch) -> None:
    import src.experiment as experiment

    test_loader = CountingLoader(_synthetic_loader())
    original = experiment.run_training

    def interrupt_once(*args, **kwargs):
        callback = kwargs["on_epoch_end"]

        def stop(payload):
            callback(payload)
            raise RuntimeError("planned interrupt")

        kwargs["on_epoch_end"] = stop
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, "run_training", interrupt_once)
    config = _config(tmp_path, epochs=2, schedulers=["constant"])
    with pytest.raises(RuntimeError, match="planned interrupt"):
        run_single(
            config,
            "constant",
            seed=0,
            loaders=(_synthetic_loader(), _synthetic_loader(), test_loader, torch.Generator()),
            device=torch.device("cpu"),
        )
    assert test_loader.iterations == 0
    monkeypatch.setattr(experiment, "run_training", original)
    summary = run_single(
        config,
        "constant",
        seed=0,
        loaders=(_synthetic_loader(), _synthetic_loader(), test_loader, torch.Generator()),
        device=torch.device("cpu"),
    )
    history = pd.read_csv(tmp_path / "results" / "runs" / "unit" / "constant" / "seed_0" / "history.csv")
    assert list(history["epoch"]) == [1, 2]
    assert summary["epochs_completed"] == 2
    assert test_loader.iterations == 1
    assert summary["thresholds"]["0.99"]["reached"] is False


def np_targets():
    import numpy as np

    return np.repeat(np.arange(10), 100)


def np_intersect(left, right):
    import numpy as np

    return np.intersect1d(left, right)


def _synthetic_loader() -> DataLoader:
    images = torch.randn(8, 3, 32, 32)
    labels = torch.arange(8) % 10
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)


def _config(tmp_path, epochs: int, schedulers: list[str]) -> dict:
    return {
        "name": "unit",
        "project_root": str(tmp_path),
        "results_dir": "results",
        "seeds": [0],
        "epochs": epochs,
        "batch_size": 4,
        "num_workers": 0,
        "val_fraction": 0.1,
        "split_seed": 0,
        "schedulers": schedulers,
        "deterministic": True,
        "amp": False,
        "device": "cpu",
        "max_train_batches": 1,
        "max_eval_batches": 1,
        "accuracy_thresholds": [0.99],
        "model": {"num_classes": 10},
        "optimizer": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
        "step": {"milestone_fractions": [0.5], "gamma": 0.1},
        "cosine": {"eta_min": 1e-5},
        "plateau": {
            "factor": 0.1,
            "patience": 1,
            "threshold": 0.01,
            "threshold_mode": "rel",
            "min_lr": 1e-6,
        },
        "ablr": {
            "ema_alpha": 0.5,
            "velocity_window": 1,
            "reference_beta": 0.1,
            "healthy_ratio": 0.4,
            "slow_ratio": 0.1,
            "slow_patience": 2,
            "stagnant_patience": 2,
            "healthy_patience": 2,
            "slow_factor": 0.5,
            "stagnant_factor": 0.2,
            "recover_factor": 1.1,
            "recover_margin": 0.02,
            "min_lr": 1e-5,
            "max_lr": 0.1,
            "warmup_epochs": 0,
            "cooldown_epochs": 1,
            "max_recoveries": 1,
            "min_reference_velocity": 1e-4,
        },
        "linear": {"eta_min": 1e-5},
    }
