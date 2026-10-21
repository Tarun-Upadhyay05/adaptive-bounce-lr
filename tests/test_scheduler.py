"""Unit tests for the Adaptive Bounce learning-rate rule.

These tests drive the scheduler with synthetic losses. They do not train a
network and their traces are not experimental results.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.scheduler import (
    Action,
    AdaptiveBounceLR,
    Reason,
    build_scheduler,
    relative_improvement,
    simulate_ablr,
)


def _make(lr: float = 0.1, **kwargs):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=lr)
    settings = dict(
        ema_alpha=1.0,
        velocity_window=1,
        reference_beta=0.0,
        healthy_ratio=0.7,
        slow_ratio=0.2,
        slow_patience=2,
        stagnant_patience=2,
        healthy_patience=2,
        slow_factor=0.5,
        stagnant_factor=0.2,
        recover_factor=1.2,
        recover_margin=0.05,
        min_lr=1e-6,
        max_lr=max(lr, 0.1),
        warmup_epochs=0,
        cooldown_epochs=0,
        max_recoveries=1,
        min_reference_velocity=1e-4,
        enable_recovery=False,
    )
    settings.update(kwargs)
    return AdaptiveBounceLR(optimizer, **settings), optimizer


def test_relative_improvement_sign() -> None:
    assert relative_improvement(4.0, 1.0) == pytest.approx(0.75)
    assert relative_improvement(1.0, 1.0) == pytest.approx(0.0)
    assert relative_improvement(1.0, 1.5) == pytest.approx(-0.5)


def test_initial_learning_rate_is_assigned() -> None:
    scheduler, optimizer = _make(0.1)
    assert scheduler.get_lr() == pytest.approx(0.1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_initial_learning_rate_is_clipped_to_bounds() -> None:
    high, optimizer = _make(0.5, max_lr=0.1, min_lr=1e-5)
    assert high.get_lr() == pytest.approx(0.1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    low, low_optimizer = _make(1e-8, min_lr=1e-4, max_lr=0.1)
    assert low.get_lr() == pytest.approx(1e-4)
    assert low_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_default_stagnation_cut_is_stronger_than_slow_cut() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    scheduler = AdaptiveBounceLR(torch.optim.SGD([parameter], lr=0.1))
    assert scheduler.stagnant_factor < scheduler.slow_factor < 1.0
    assert scheduler.enable_recovery is False


def test_invalid_hyperparameters_raise() -> None:
    with pytest.raises(ValueError):
        _make(healthy_ratio=0.1, slow_ratio=0.2)
    with pytest.raises(ValueError):
        _make(recover_factor=1.0)
    with pytest.raises(ValueError):
        _make(min_lr=0.2, max_lr=0.1)


def test_healthy_geometric_progress_holds_learning_rate() -> None:
    scheduler, _optimizer = _make(velocity_window=2, healthy_ratio=0.5, slow_ratio=0.1)
    losses = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
    for loss in losses:
        decision = scheduler.step(loss, 0.5)
    assert scheduler.get_lr() == pytest.approx(0.1)
    assert all(row["lr_changed"] is False for row in scheduler.history)
    assert decision["reason"] == Reason.HEALTHY
    assert decision["normalized_progress"] == pytest.approx(1.0)


def test_slowing_progress_applies_the_gradual_factor() -> None:
    scheduler, _optimizer = _make()
    losses = [8.0, 2.0, 1.25, 0.78125]
    for loss in losses:
        scheduler.step(loss, 0.4)
    assert scheduler.history[-1]["reason"] == Reason.SLOWING
    assert scheduler.history[-1]["action"] == Action.DECAY_SLOW
    assert scheduler.get_lr() == pytest.approx(0.05)
    assert scheduler.history[-2]["lr_changed"] is False


def test_sustained_stagnation_applies_the_strong_factor() -> None:
    scheduler, _optimizer = _make()
    losses = [4.0, 2.0, 2.0, 2.0]
    for loss in losses:
        scheduler.step(loss, 0.4)
    assert scheduler.history[-2]["reason"] == Reason.PATIENCE
    assert scheduler.history[-1]["reason"] == Reason.STAGNATION
    assert scheduler.history[-1]["action"] == Action.DECAY_STAGNANT
    assert scheduler.get_lr() == pytest.approx(0.02)


def test_warmup_blocks_reductions() -> None:
    scheduler, _optimizer = _make(warmup_epochs=5, stagnant_patience=1)
    for _ in range(5):
        decision = scheduler.step(1.0, 0.2)
        assert decision["reason"] == Reason.WARMUP
        assert decision["lr_changed"] is False
    assert scheduler.get_lr() == pytest.approx(0.1)


def test_single_epoch_fluctuations_do_not_reduce() -> None:
    scheduler, _optimizer = _make(stagnant_patience=2, slow_patience=2)
    losses = [1.0, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125]
    for loss in losses:
        scheduler.step(loss, 0.3)
    assert scheduler.get_lr() == pytest.approx(0.1)
    assert all(row["lr_changed"] is False for row in scheduler.history)


def test_minimum_learning_rate_saturation() -> None:
    scheduler, _optimizer = _make(lr=1e-4, min_lr=1e-4, max_lr=1e-4)
    for loss in (4.0, 2.0, 2.0, 2.0, 2.0):
        scheduler.step(loss, 0.1)
    assert scheduler.get_lr() == pytest.approx(1e-4)
    assert scheduler.history[-1]["reason"] == Reason.MIN_LR
    assert scheduler.history[-1]["lr_changed"] is False


def test_recovery_requires_sustained_loss_improvement_and_respects_caps() -> None:
    losses = [4.0, 2.0, 2.0, 2.0, 1.0, 0.8]
    settings = dict(
        ema_alpha=1.0,
        velocity_window=1,
        reference_beta=0.0,
        healthy_ratio=0.35,
        slow_ratio=0.1,
        slow_patience=2,
        stagnant_patience=2,
        healthy_patience=2,
        slow_factor=0.5,
        stagnant_factor=0.5,
        recover_factor=1.2,
        recover_margin=0.05,
        min_lr=1e-6,
        max_lr=0.1,
        warmup_epochs=0,
        cooldown_epochs=0,
        max_recoveries=1,
        min_reference_velocity=1e-4,
        enable_recovery=True,
    )
    rising = [index / 10 for index in range(len(losses))]
    flat_accuracy = simulate_ablr(losses, [0.0] * len(losses), **settings)
    rising_accuracy = simulate_ablr(losses, rising, **settings)
    assert [row["lr"] for row in flat_accuracy] == pytest.approx([row["lr"] for row in rising_accuracy])
    assert [row["reason"] for row in flat_accuracy] == [row["reason"] for row in rising_accuracy]
    assert flat_accuracy[-1]["reason"] == Reason.RECOVERY
    assert flat_accuracy[-1]["lr"] == pytest.approx(0.06)
    assert flat_accuracy[-1]["lr"] <= 0.1

    capped = simulate_ablr(
        losses,
        lr=0.1,
        enable_recovery=True,
        healthy_ratio=0.35,
        slow_ratio=0.1,
        ema_alpha=1.0,
        velocity_window=1,
        reference_beta=0.0,
        stagnant_patience=2,
        healthy_patience=2,
        stagnant_factor=0.5,
        recover_factor=5.0,
        recover_margin=0.05,
        min_lr=1e-6,
        max_lr=0.2,
        warmup_epochs=0,
        cooldown_epochs=0,
        max_recoveries=1,
    )
    assert capped[-1]["reason"] == Reason.RECOVERY
    assert capped[-1]["lr"] == pytest.approx(0.1)


def test_recovery_cooldown_suppresses_an_immediate_bounce() -> None:
    scheduler, _optimizer = _make(
        enable_recovery=True,
        healthy_ratio=0.35,
        slow_ratio=0.1,
        stagnant_factor=0.5,
        cooldown_epochs=2,
    )
    for loss in (4.0, 2.0, 2.0, 2.0, 1.0, 0.8):
        scheduler.step(loss, 0.9)
    assert scheduler.get_lr() == pytest.approx(0.05)
    assert scheduler.history[-1]["reason"] == Reason.COOLDOWN
    assert scheduler.n_lr_changes == 1


def test_accuracy_improvement_alone_does_not_restore_learning_rate() -> None:
    scheduler, _optimizer = _make(enable_recovery=True, healthy_ratio=0.35, slow_ratio=0.1, stagnant_factor=0.5)
    losses = [4.0, 2.0, 2.0, 2.0]
    accuracies = [0.1, 0.2, 0.4, 0.9]
    for loss, accuracy in zip(losses, accuracies):
        scheduler.step(loss, accuracy)
    reduced = scheduler.get_lr()
    for accuracy in (0.91, 0.95, 0.99):
        scheduler.step(2.0, accuracy)
    assert scheduler.get_lr() <= reduced + 1e-12
    assert all(row["action"] != Action.RECOVER for row in scheduler.history)


def test_invalid_and_missing_metrics_are_ignored() -> None:
    scheduler, _optimizer = _make()
    scheduler.step(2.0, 0.4)
    ema = scheduler.ema_loss
    for value in (float("nan"), None, float("inf"), np.nan):
        decision = scheduler.step(value, 0.9)
        assert decision["reason"] == Reason.INVALID
        assert decision["lr_changed"] is False
    assert scheduler.ema_loss == pytest.approx(ema)
    assert scheduler.get_lr() == pytest.approx(0.1)
    with pytest.raises(TypeError):
        scheduler.step("bad-loss")


def test_checkpoint_round_trip_continues_the_same_decision() -> None:
    scheduler, _optimizer = _make()
    for loss in (8.0, 4.0, 2.0):
        scheduler.step(loss, 0.2)
    restored, _other = _make()
    restored.load_state_dict(scheduler.state_dict())
    original_next = scheduler.step(1.0, 0.3)
    restored_next = restored.step(1.0, 0.3)
    assert restored_next["lr"] == pytest.approx(original_next["lr"])
    assert restored_next["reason"] == original_next["reason"]
    assert restored_next["action"] == original_next["action"]
    assert restored_next["normalized_progress"] == pytest.approx(original_next["normalized_progress"])


def test_checkpoint_rejects_incomplete_or_unknown_state() -> None:
    scheduler, _optimizer = _make()
    scheduler.step(1.0, 0.2)
    state = scheduler.state_dict()
    broken = dict(state)
    del broken["ema_loss"]
    other, _optimizer = _make()
    with pytest.raises(KeyError):
        other.load_state_dict(broken)
    state["version"] = "not-a-version"
    with pytest.raises(ValueError):
        other.load_state_dict(state)


def _baseline_config() -> dict:
    return {
        "step": {"milestone_fractions": [0.5], "gamma": 0.1},
        "cosine": {"eta_min": 0.0},
        "plateau": {
            "factor": 0.1,
            "patience": 1,
            "threshold": 0.0,
            "threshold_mode": "abs",
            "min_lr": 1e-6,
        },
        "ablr": {
            "ema_alpha": 1.0,
            "velocity_window": 1,
            "reference_beta": 0.0,
            "healthy_ratio": 0.7,
            "slow_ratio": 0.2,
            "slow_patience": 2,
            "stagnant_patience": 2,
            "healthy_patience": 2,
            "slow_factor": 0.5,
            "stagnant_factor": 0.2,
            "recover_factor": 1.2,
            "recover_margin": 0.05,
            "min_lr": 1e-6,
            "max_lr": 0.1,
            "warmup_epochs": 0,
            "cooldown_epochs": 0,
            "max_recoveries": 1,
            "min_reference_velocity": 1e-4,
        },
        "linear": {"eta_min": 1e-5},
    }


def _tracked_lrs(name: str, steps: int, losses: list[float] | None = None) -> list[float]:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = build_scheduler(name, optimizer, _baseline_config(), total_epochs=steps)
    seen = [float(optimizer.param_groups[0]["lr"])]
    for index in range(steps):
        if name in {"plateau", "ablr", "ablr_recovery"}:
            loss = 1.0 if losses is None else losses[index]
            if name == "plateau":
                scheduler.step(loss)
            else:
                scheduler.step(loss, 0.1)
        else:
            scheduler.step()
        seen.append(float(optimizer.param_groups[0]["lr"]))
    return seen


def test_baseline_schedulers_can_hold_or_reduce() -> None:
    constant = _tracked_lrs("constant", 4)
    assert constant == pytest.approx([0.1] * len(constant))
    step = _tracked_lrs("step", 4)
    cosine = _tracked_lrs("cosine", 4)
    plateau = _tracked_lrs("plateau", 6)
    assert min(step) < 0.1
    assert min(cosine) < 0.1
    assert min(plateau) < 0.1
    recovery = build_scheduler(
        "ablr_recovery",
        torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        _baseline_config(),
        total_epochs=4,
    )
    plain = build_scheduler(
        "ablr",
        torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        _baseline_config(),
        total_epochs=4,
    )
    assert recovery.enable_recovery is True
    assert plain.enable_recovery is False


def test_linear_decay_ends_at_eta_min_and_resumes() -> None:
    from src.scheduler import LinearDecayLR, linear_decay_lr

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.15)
    scheduler = build_scheduler("linear", optimizer, _baseline_config(), total_epochs=20)
    assert isinstance(scheduler, LinearDecayLR)
    assert scheduler.get_lr() == pytest.approx(0.15)
    for _ in range(20):
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-5)
    assert scheduler.get_lr() == pytest.approx(linear_decay_lr(20, 20, 0.15, 1e-5))

    restored = LinearDecayLR(torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.15), 20, 1e-5)
    restored.load_state_dict(scheduler.state_dict())
    assert restored.get_lr() == pytest.approx(1e-5)
    assert restored.steps_taken == 20


def test_hybrid_tracks_the_envelope_while_the_rule_holds() -> None:
    from src.scheduler import LinearDecayLR

    losses = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    hybrid, _optimizer = _make(
        0.15,
        max_lr=0.15,
        min_lr=1e-6,
        velocity_window=1,
        healthy_ratio=0.5,
        slow_ratio=0.1,
        warmup_epochs=2,
        cooldown_epochs=0,
        envelope="linear",
        total_epochs=len(losses),
        envelope_eta_min=1e-5,
    )
    linear = LinearDecayLR(torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.15), len(losses), 1e-5)
    assert hybrid.get_lr() == pytest.approx(0.15)
    for loss in losses:
        decision = hybrid.step(loss, 0.5)
        linear.step()
        assert hybrid.get_lr() == pytest.approx(linear.get_lr())
        assert hybrid.reactive_lr == pytest.approx(0.15)
        assert decision["lr_changed"] is True
        assert decision["reason"] == Reason.ENVELOPE
    assert hybrid.get_lr() < 0.02


def test_hybrid_cut_undercuts_the_envelope_until_the_envelope_passes_it() -> None:
    scheduler, _optimizer = _make(
        0.1,
        max_lr=0.1,
        min_lr=1e-5,
        stagnant_patience=2,
        stagnant_factor=0.2,
        warmup_epochs=0,
        cooldown_epochs=30,
        envelope="linear",
        total_epochs=20,
        envelope_eta_min=1e-5,
    )
    for loss in (4.0, 2.0, 2.0, 2.0):
        scheduler.step(loss, 0.4)
    assert scheduler.reactive_lr == pytest.approx(0.02)
    assert scheduler.get_lr() == pytest.approx(0.02)
    assert scheduler.envelope_lr is not None and scheduler.envelope_lr > 0.02
    for _ in range(16):
        scheduler.step(2.0, 0.4)
    assert scheduler.reactive_lr == pytest.approx(0.02)
    assert scheduler.get_lr() == pytest.approx(1e-5)
    assert scheduler.get_lr() < scheduler.reactive_lr


def test_hybrid_peak_follows_the_optimizer_when_max_lr_is_lower() -> None:
    config = _baseline_config()
    config["ablr"]["max_lr"] = 0.1
    parameter = torch.nn.Parameter(torch.zeros(1))
    scheduler = build_scheduler(
        "ablr_hybrid",
        torch.optim.SGD([parameter], lr=0.15),
        config,
        total_epochs=20,
    )
    assert scheduler.enable_recovery is False
    assert scheduler.envelope == "linear"
    assert scheduler.get_lr() == pytest.approx(0.15)
    assert scheduler.max_lr == pytest.approx(0.15)
