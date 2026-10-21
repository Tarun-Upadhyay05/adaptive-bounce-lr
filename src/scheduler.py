"""Adaptive Bounce learning-rate scheduler (ABLR).

ABLR is an experimental, validation-driven learning-rate rule. It is not a
claim of a new state-of-the-art optimizer. Validation loss is the only signal
that can change the learning rate. Validation accuracy is smoothed and recorded
so it can be inspected, and it is intentionally excluded from the decision rule:
accuracy can sit still while the loss keeps falling, and it can tick up after a
learning-rate cut even when that cut should stay in place.

Mathematical formulation
------------------------

Let ``ell_t`` be the validation loss at epoch ``t`` and ``alpha`` the smoothing
coefficient. The monitored loss is the exponential moving average

    ema_t = alpha * ell_t + (1 - alpha) * ema_{t-1},

with ``ema_1 = ell_1``. Over a window of ``W`` epochs the relative improvement is

    v_t = (ema_{t-W} - ema_t) / max(|ema_{t-W}|, epsilon).

Positive ``v_t`` means the smoothed loss fell. Dividing by the loss level keeps
late-training improvements, which are small in absolute terms, from being
classified as stagnation merely because the loss itself is small.

The reference scale ``r_t`` is a slow exponential average of *previous* positive
improvements. The current velocity is compared with the reference from before
the current update:

    rho_t = v_t / max(r_prev, epsilon_v).

If no positive reference exists yet, a positive velocity is treated as
``rho_t = 1`` (it defines the baseline) and a non-positive velocity is treated
as no progress. The reference is then updated only when ``v_t > 0``:

    r_t = beta * v_t + (1 - beta) * r_prev.

``beta`` is intentionally small. A large beta lets the baseline chase a
slowdown until every regime looks healthy again.

Regimes, using two thresholds with ``rho_slow < rho_healthy``:

    healthy   if rho_t >= rho_healthy
    slowing   if rho_slow <= rho_t < rho_healthy
    stagnant  if rho_t < rho_slow

A regime must persist for its patience count before it changes the learning
rate. A single noisy epoch therefore cannot decay or restore the rate. While a
cooldown is active, patience counters do not accumulate.

Actions, each followed by clipping to ``[eta_min, eta_max]``:

    slowing, patience met:    eta <- eta * gamma_slow
    stagnant, patience met:   eta <- eta * gamma_stag
    optional recovery:        eta <- min(eta_max, eta_before, eta * gamma_recover)

``gamma_stag <= gamma_slow < 1`` and ``gamma_recover > 1``. Stagnation
therefore cuts at least as hard as slowing, and recovery can only move upward.

Recovery is off in variant A and on in variant B. It fires only when all of
the following are true:

1. Recovery is enabled.
2. A reduction has already happened.
3. The cooldown after the last learning-rate change has expired.
4. The healthy regime has held for ``healthy_patience`` consecutive decision
   epochs since that cooldown.
5. Smoothed loss has improved since the reduction by a relative margin::

       (ema_cut - ema_t) / max(|ema_cut|, epsilon) >= delta

6. Fewer than ``max_recoveries`` restorations have been used since that
   reduction.
7. The proposed rate is strictly above the current rate after the cap
   ``min(eta_max, eta_before_reduction)`` is applied.

Accuracy is not one of these conditions. A rise in accuracy after a cut is not
evidence that the cut should be undone. The required evidence is a sustained
return of *loss* improvement, large enough to clear ``delta``, after the
cooldown. The restored rate cannot pass the rate from before the cut, so a
bounce is a partial, capped step rather than a restart.

Warm-up epochs update the moving averages and the reference but never change
the reactive learning rate and never advance patience counters. Non-finite or
missing losses are ignored: they do not move the averages, the counters, or
the reactive rate.

Budget envelope
---------------

``linear`` decays from the optimizer's initial rate to ``eta_min`` across the
epoch budget. ``ablr_hybrid`` publishes

    eta_applied = min(eta_reactive, eta_envelope)

after every epoch, including warm-up and cooldown. The reactive rule can only
undercut the envelope. It cannot keep the rate at the peak through the end of
a fixed budget. The first epoch uses the peak. Each epoch-end step sets the
next epoch's rate, and ``eta_min`` is reached on the step after the final
epoch, matching cosine annealing's timing.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
from torch.optim import Optimizer


ALGORITHM_VERSION = "ablr-v1"

SCHEDULER_NAMES = (
    "constant",
    "step",
    "cosine",
    "linear",
    "plateau",
    "ablr",
    "ablr_recovery",
    "ablr_hybrid",
)


class Reason:
    WARMUP = "warmup"
    INVALID = "invalid_metric"
    INSUFFICIENT = "insufficient_history"
    HEALTHY = "healthy_progress"
    SLOWING = "slowing_progress"
    STAGNATION = "sustained_stagnation"
    RECOVERY = "bounded_recovery"
    COOLDOWN = "cooldown_hold"
    MIN_LR = "min_lr_saturated"
    RECOVERY_CAP = "recovery_at_cap"
    RECOVERY_MARGIN = "recovery_margin_not_met"
    RECOVERY_LIMIT = "recovery_limit"
    PATIENCE = "patience_hold"
    ENVELOPE = "budget_envelope"


class Action:
    HOLD = "hold"
    DECAY_SLOW = "decay_slow"
    DECAY_STAGNANT = "decay_stagnant"
    RECOVER = "recover"
    ENVELOPE = "budget_envelope"


def relative_improvement(past: float, current: float, eps: float = 1e-12) -> float:
    """Relative drop from ``past`` to ``current``. Positive means improvement."""
    return (past - current) / max(abs(past), eps)


def clip_lr(lr: float, min_lr: float, max_lr: float) -> float:
    return float(min(max_lr, max(min_lr, lr)))


def linear_decay_lr(steps_taken: int, total_epochs: int, base_lr: float, eta_min: float) -> float:
    """Learning rate after ``steps_taken`` epoch-end steps of a linear decay.

    Zero steps leaves the rate at ``base_lr``. ``total_epochs`` steps reach
    ``eta_min``. Intermediate steps are linear in the step index.
    """
    if total_epochs < 1:
        raise ValueError("total_epochs must be >= 1")
    fraction = min(1.0, max(0.0, steps_taken / float(total_epochs)))
    return float(base_lr + (eta_min - base_lr) * fraction)


def classify_regime(ratio: float, healthy_ratio: float, slow_ratio: float) -> str:
    if ratio >= healthy_ratio:
        return "healthy"
    if ratio >= slow_ratio:
        return "slowing"
    return "stagnant"


def _as_optional_float(value: Any, name: str, *, allow_none: bool) -> float | None:
    if value is None:
        if allow_none:
            return None
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # NumPy scalars implement the float interface via ``item`` but are not
        # always subclasses of ``float`` on every version. Accept them by
        # numeric protocol below.
        if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
            value = value.item()
        else:
            raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


class AdaptiveBounceLR:
    """Validation-loss learning-rate scheduler with optional bounded recovery.

    Call :meth:`step` once per epoch, after validation. The learning rate stored
    on the optimizer is the rate that the *next* epoch will use. The rate used
    during the epoch that just finished is the value from before ``step``.

    Parameters
    ----------
    optimizer:
        Optimizer whose param groups share one learning rate.
    ema_alpha:
        EMA coefficient for validation loss and accuracy. Larger values track
        the latest epoch more closely and let more noise through.
    velocity_window:
        Number of epochs ``W`` in the relative-improvement estimate. Larger
        windows react more slowly and ignore brief spikes.
    reference_beta:
        EMA coefficient for the positive-velocity baseline. Smaller values
        remember earlier fast progress for longer, so the scheduler decays
        sooner when progress slows. Larger values accept a new slower pace as
        normal.
    healthy_ratio, slow_ratio:
        Normalized-velocity thresholds. Progress at or above ``healthy_ratio``
        holds the learning rate. Progress between the two thresholds is
        slowing. Progress below ``slow_ratio`` is stagnation.
    slow_patience, stagnant_patience, healthy_patience:
        Consecutive decision epochs required before a slow decay, a strong
        decay, or a recovery. These are the hysteresis controls.
    slow_factor, stagnant_factor:
        Multipliers applied to the learning rate. Both are in ``(0, 1]`` and
        the stagnation factor is no larger than the slow factor.
    recover_factor:
        Multiplier used for a bounce. Must be greater than 1. The result is
        capped by both ``max_lr`` and the learning rate from before the
        reduction that opened the recovery window.
    recover_margin:
        Minimum relative smoothed-loss improvement since the last reduction
        required before a bounce.
    min_lr, max_lr:
        Inclusive bounds. ``max_lr`` defaults to the optimizer's initial rate.
    warmup_epochs:
        First epochs update statistics only. They do not change the rate and
        do not count toward patience.
    cooldown_epochs:
        Epochs after any learning-rate change during which another change is
        suppressed. Patience counters reset on a change and do not move during
        the cooldown, so noise cannot pre-arm the next cut.
    max_recoveries:
        Bounces allowed after each reduction. A new reduction resets the count.
    min_reference_velocity:
        Floor on the reference scale so a tiny baseline cannot explode ``rho``.
    enable_recovery:
        Variant switch. ``False`` is ABLR without bounce. ``True`` is ABLR with
        bounded recovery. The hybrid variant forces this off.
    envelope:
        ``"none"`` publishes the reactive rate. ``"linear"`` publishes the
        minimum of the reactive rate and a linear decay from the peak rate to
        ``envelope_eta_min`` over ``total_epochs``.
    total_epochs:
        Budget used by the linear envelope. Required when ``envelope`` is
        ``"linear"``.
    envelope_eta_min:
        Final rate of the linear envelope. Defaults to ``min_lr``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        ema_alpha: float = 0.2,
        velocity_window: int = 5,
        reference_beta: float = 0.1,
        healthy_ratio: float = 0.4,
        slow_ratio: float = 0.1,
        slow_patience: int = 2,
        stagnant_patience: int = 3,
        healthy_patience: int = 3,
        slow_factor: float = 0.5,
        stagnant_factor: float = 0.1,
        recover_factor: float = 1.1,
        recover_margin: float = 0.02,
        min_lr: float = 1e-5,
        max_lr: float | None = None,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 3,
        max_recoveries: int = 1,
        min_reference_velocity: float = 1e-4,
        enable_recovery: bool = False,
        envelope: str = "none",
        total_epochs: int = 0,
        envelope_eta_min: float | None = None,
    ) -> None:
        if not optimizer.param_groups:
            raise ValueError("optimizer has no parameter groups")
        self.optimizer = optimizer
        self._validate_hyperparameters(
            ema_alpha=ema_alpha,
            velocity_window=velocity_window,
            reference_beta=reference_beta,
            healthy_ratio=healthy_ratio,
            slow_ratio=slow_ratio,
            slow_patience=slow_patience,
            stagnant_patience=stagnant_patience,
            healthy_patience=healthy_patience,
            slow_factor=slow_factor,
            stagnant_factor=stagnant_factor,
            recover_factor=recover_factor,
            recover_margin=recover_margin,
            min_lr=min_lr,
            max_lr=max_lr if max_lr is not None else float(optimizer.param_groups[0]["lr"]),
            warmup_epochs=warmup_epochs,
            cooldown_epochs=cooldown_epochs,
            max_recoveries=max_recoveries,
            min_reference_velocity=min_reference_velocity,
            envelope=envelope,
            total_epochs=total_epochs,
            envelope_eta_min=float(min_lr if envelope_eta_min is None else envelope_eta_min),
        )
        self.ema_alpha = float(ema_alpha)
        self.velocity_window = int(velocity_window)
        self.reference_beta = float(reference_beta)
        self.healthy_ratio = float(healthy_ratio)
        self.slow_ratio = float(slow_ratio)
        self.slow_patience = int(slow_patience)
        self.stagnant_patience = int(stagnant_patience)
        self.healthy_patience = int(healthy_patience)
        self.slow_factor = float(slow_factor)
        self.stagnant_factor = float(stagnant_factor)
        self.recover_factor = float(recover_factor)
        self.recover_margin = float(recover_margin)
        self.min_lr = float(min_lr)
        initial = float(optimizer.param_groups[0]["lr"])
        self.max_lr = float(initial if max_lr is None else max_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.cooldown_epochs = int(cooldown_epochs)
        self.max_recoveries = int(max_recoveries)
        self.min_reference_velocity = float(min_reference_velocity)
        self.enable_recovery = bool(enable_recovery)
        self.envelope = str(envelope)
        self.total_epochs = int(total_epochs)
        self.envelope_eta_min = float(self.min_lr if envelope_eta_min is None else envelope_eta_min)

        self.lr = clip_lr(initial, self.min_lr, self.max_lr)
        self.base_lr = self.lr
        self.reactive_lr = self.lr
        self.envelope_lr: float | None = None
        self.initial_lr_clipped = self.lr != initial
        self._write_lr(self.lr)

        self.n_steps = 0
        self.n_lr_changes = 0
        self.epochs_since_lr_change = 0
        self.ema_loss: float | None = None
        self.ema_accuracy: float | None = None
        self.ema_loss_history: list[float] = []
        self.reference_velocity: float | None = None
        self.healthy_count = 0
        self.slow_count = 0
        self.stagnant_count = 0
        self.last_reduction_ema: float | None = None
        self.lr_before_reduction: float | None = None
        self.recoveries_since_reduction = 0
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _validate_hyperparameters(**kwargs: Any) -> None:
        if not 0.0 < kwargs["ema_alpha"] <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if kwargs["velocity_window"] < 1:
            raise ValueError("velocity_window must be >= 1")
        if not 0.0 <= kwargs["reference_beta"] <= 1.0:
            raise ValueError("reference_beta must be in [0, 1]")
        if kwargs["slow_ratio"] < 0:
            raise ValueError("slow_ratio must be >= 0")
        if kwargs["healthy_ratio"] <= kwargs["slow_ratio"]:
            raise ValueError("healthy_ratio must be greater than slow_ratio")
        for name in ("slow_patience", "stagnant_patience", "healthy_patience"):
            if kwargs[name] < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 < kwargs["stagnant_factor"] <= kwargs["slow_factor"] <= 1.0:
            raise ValueError("require 0 < stagnant_factor <= slow_factor <= 1")
        if kwargs["recover_factor"] <= 1.0:
            raise ValueError("recover_factor must be > 1")
        if kwargs["recover_margin"] < 0:
            raise ValueError("recover_margin must be >= 0")
        if kwargs["min_lr"] <= 0:
            raise ValueError("min_lr must be > 0")
        if kwargs["max_lr"] < kwargs["min_lr"]:
            raise ValueError("max_lr must be >= min_lr")
        if kwargs["warmup_epochs"] < 0 or kwargs["cooldown_epochs"] < 0:
            raise ValueError("warmup_epochs and cooldown_epochs must be >= 0")
        if kwargs["max_recoveries"] < 0:
            raise ValueError("max_recoveries must be >= 0")
        if kwargs["min_reference_velocity"] <= 0:
            raise ValueError("min_reference_velocity must be > 0")
        if kwargs["envelope"] not in {"none", "linear"}:
            raise ValueError("envelope must be 'none' or 'linear'")
        if kwargs["total_epochs"] < 0:
            raise ValueError("total_epochs must be >= 0")
        if kwargs["envelope"] == "linear" and kwargs["total_epochs"] < 1:
            raise ValueError("total_epochs must be >= 1 when envelope is 'linear'")
        if kwargs["envelope_eta_min"] < 0:
            raise ValueError("envelope_eta_min must be >= 0")

    def get_lr(self) -> float:
        return float(self.lr)

    def step(self, val_loss: Any, val_accuracy: Any = None) -> dict[str, Any]:
        """Ingest one epoch of validation metrics and maybe adapt the rate.

        The reactive rule runs first. The published rate is then the minimum of
        that rule and the budget envelope, so a hold, warm-up, or cooldown
        cannot leave the rate above the envelope.
        """
        self.n_steps += 1
        loss = _as_optional_float(val_loss, "val_loss", allow_none=True)
        if val_accuracy is None:
            accuracy = None
        else:
            accuracy = _as_optional_float(val_accuracy, "val_accuracy", allow_none=True)

        velocity = None
        ratio = None
        rule_changed = False
        if loss is None:
            regime, action, reason = "invalid", Action.HOLD, Reason.INVALID
        else:
            self._update_ema(loss, accuracy)
            velocity = self._velocity()
            if velocity is not None:
                ratio = self._ratio_against_reference(velocity)
                self._update_reference(velocity)

            if self.n_steps <= self.warmup_epochs:
                regime, action, reason = "warmup", Action.HOLD, Reason.WARMUP
            elif velocity is None or ratio is None:
                regime, action, reason = "insufficient_history", Action.HOLD, Reason.INSUFFICIENT
            else:
                regime = classify_regime(ratio, self.healthy_ratio, self.slow_ratio)
                if self.n_lr_changes > 0:
                    self.epochs_since_lr_change += 1
                cooldown_active = (
                    self.n_lr_changes > 0 and self.epochs_since_lr_change <= self.cooldown_epochs
                )
                if cooldown_active:
                    action, reason = Action.HOLD, Reason.COOLDOWN
                else:
                    self._advance_counters(regime)
                    action, reason, rule_changed = self._act(regime)

        applied_changed = self._apply_budget()
        if not rule_changed and applied_changed:
            action, reason = Action.ENVELOPE, Reason.ENVELOPE
        record = self._record(
            val_loss=loss,
            val_accuracy=accuracy,
            velocity=velocity,
            ratio=ratio,
            regime=regime,
            action=action,
            reason=reason,
            lr_changed=rule_changed or applied_changed,
        )
        self.history.append(record)
        return record

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": ALGORITHM_VERSION,
            "hyperparameters": self.hyperparameters(),
            "lr": self.lr,
            "reactive_lr": self.reactive_lr,
            "base_lr": self.base_lr,
            "initial_lr_clipped": self.initial_lr_clipped,
            "n_steps": self.n_steps,
            "n_lr_changes": self.n_lr_changes,
            "epochs_since_lr_change": self.epochs_since_lr_change,
            "ema_loss": self.ema_loss,
            "ema_accuracy": self.ema_accuracy,
            "ema_loss_history": list(self.ema_loss_history),
            "reference_velocity": self.reference_velocity,
            "healthy_count": self.healthy_count,
            "slow_count": self.slow_count,
            "stagnant_count": self.stagnant_count,
            "last_reduction_ema": self.last_reduction_ema,
            "lr_before_reduction": self.lr_before_reduction,
            "recoveries_since_reduction": self.recoveries_since_reduction,
            "history": deepcopy(self.history),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != ALGORITHM_VERSION:
            raise ValueError(
                f"unsupported scheduler state version {state.get('version')!r}; "
                f"expected {ALGORITHM_VERSION}"
            )
        required = (
            "lr",
            "n_steps",
            "n_lr_changes",
            "epochs_since_lr_change",
            "ema_loss",
            "ema_accuracy",
            "ema_loss_history",
            "reference_velocity",
            "healthy_count",
            "slow_count",
            "stagnant_count",
            "last_reduction_ema",
            "lr_before_reduction",
            "recoveries_since_reduction",
            "history",
            "hyperparameters",
        )
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"scheduler state is missing keys: {missing}")
        # Restore the hyperparameters that governed this run so a later default
        # change cannot alter an in-progress resume.
        for key, value in state["hyperparameters"].items():
            setattr(self, key, value)
        self.lr = float(state["lr"])
        self.reactive_lr = float(state.get("reactive_lr", state["lr"]))
        self.base_lr = float(state.get("base_lr", state["lr"]))
        self.initial_lr_clipped = bool(state.get("initial_lr_clipped", False))
        self.n_steps = int(state["n_steps"])
        self.n_lr_changes = int(state["n_lr_changes"])
        self.epochs_since_lr_change = int(state["epochs_since_lr_change"])
        self.ema_loss = state["ema_loss"]
        self.ema_accuracy = state["ema_accuracy"]
        self.ema_loss_history = list(state["ema_loss_history"])
        self.reference_velocity = state["reference_velocity"]
        self.healthy_count = int(state["healthy_count"])
        self.slow_count = int(state["slow_count"])
        self.stagnant_count = int(state["stagnant_count"])
        self.last_reduction_ema = state["last_reduction_ema"]
        self.lr_before_reduction = state["lr_before_reduction"]
        self.recoveries_since_reduction = int(state["recoveries_since_reduction"])
        self.history = deepcopy(state["history"])
        self._write_lr(self.lr)

    def hyperparameters(self) -> dict[str, Any]:
        return {
            "ema_alpha": self.ema_alpha,
            "velocity_window": self.velocity_window,
            "reference_beta": self.reference_beta,
            "healthy_ratio": self.healthy_ratio,
            "slow_ratio": self.slow_ratio,
            "slow_patience": self.slow_patience,
            "stagnant_patience": self.stagnant_patience,
            "healthy_patience": self.healthy_patience,
            "slow_factor": self.slow_factor,
            "stagnant_factor": self.stagnant_factor,
            "recover_factor": self.recover_factor,
            "recover_margin": self.recover_margin,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "warmup_epochs": self.warmup_epochs,
            "cooldown_epochs": self.cooldown_epochs,
            "max_recoveries": self.max_recoveries,
            "min_reference_velocity": self.min_reference_velocity,
            "enable_recovery": self.enable_recovery,
            "envelope": self.envelope,
            "total_epochs": self.total_epochs,
            "envelope_eta_min": self.envelope_eta_min,
        }

    def _update_ema(self, loss: float, accuracy: float | None) -> None:
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.ema_loss
        self.ema_loss_history.append(float(self.ema_loss))
        if accuracy is None:
            return
        if self.ema_accuracy is None:
            self.ema_accuracy = accuracy
        else:
            self.ema_accuracy = self.ema_alpha * accuracy + (1.0 - self.ema_alpha) * self.ema_accuracy

    def _velocity(self) -> float | None:
        window = self.velocity_window
        if len(self.ema_loss_history) < window + 1:
            return None
        past = self.ema_loss_history[-(window + 1)]
        current = self.ema_loss_history[-1]
        return relative_improvement(past, current)

    def _ratio_against_reference(self, velocity: float) -> float:
        if self.reference_velocity is None:
            if velocity > 0:
                return 1.0
            return velocity / self.min_reference_velocity
        scale = max(self.reference_velocity, self.min_reference_velocity)
        return velocity / scale

    def _update_reference(self, velocity: float) -> None:
        if velocity <= 0:
            return
        if self.reference_velocity is None:
            self.reference_velocity = max(velocity, self.min_reference_velocity)
            return
        updated = (
            self.reference_beta * velocity
            + (1.0 - self.reference_beta) * self.reference_velocity
        )
        self.reference_velocity = max(updated, self.min_reference_velocity)

    def _advance_counters(self, regime: str) -> None:
        if regime == "healthy":
            self.healthy_count += 1
            self.slow_count = 0
            self.stagnant_count = 0
        elif regime == "slowing":
            self.slow_count += 1
            self.healthy_count = 0
            self.stagnant_count = 0
        else:
            self.stagnant_count += 1
            self.healthy_count = 0
            self.slow_count = 0

    def _act(self, regime: str) -> tuple[str, str, bool]:
        if self.stagnant_count >= self.stagnant_patience:
            return self._decay(self.stagnant_factor, Action.DECAY_STAGNANT, Reason.STAGNATION)
        if self.slow_count >= self.slow_patience:
            return self._decay(self.slow_factor, Action.DECAY_SLOW, Reason.SLOWING)
        if regime == "healthy":
            return self._maybe_recover()
        return Action.HOLD, Reason.PATIENCE, False

    def _decay(self, factor: float, action: str, reason: str) -> tuple[str, str, bool]:
        current = self.reactive_lr
        proposed = clip_lr(current * factor, self.min_lr, self.max_lr)
        if proposed >= current - 1e-15:
            return Action.HOLD, Reason.MIN_LR, False
        self._commit_lr(proposed, reduction=True, lr_before=current)
        return action, reason, True

    def _maybe_recover(self) -> tuple[str, str, bool]:
        if not self.enable_recovery or self.last_reduction_ema is None:
            return Action.HOLD, Reason.HEALTHY, False
        if self.healthy_count < self.healthy_patience:
            return Action.HOLD, Reason.HEALTHY, False
        if self.recoveries_since_reduction >= self.max_recoveries:
            return Action.HOLD, Reason.RECOVERY_LIMIT, False
        assert self.ema_loss is not None
        relative_gain = relative_improvement(self.last_reduction_ema, self.ema_loss)
        if relative_gain < self.recover_margin:
            return Action.HOLD, Reason.RECOVERY_MARGIN, False
        current = self.reactive_lr
        cap = self.max_lr
        if self.lr_before_reduction is not None:
            cap = min(cap, self.lr_before_reduction)
        proposed = min(clip_lr(current * self.recover_factor, self.min_lr, self.max_lr), cap)
        if proposed <= current * (1.0 + 1e-12):
            return Action.HOLD, Reason.RECOVERY_CAP, False
        self._commit_lr(proposed, reduction=False, lr_before=current)
        self.recoveries_since_reduction += 1
        return Action.RECOVER, Reason.RECOVERY, True

    def _commit_lr(self, new_lr: float, *, reduction: bool, lr_before: float) -> None:
        self.reactive_lr = float(new_lr)
        self.n_lr_changes += 1
        self.epochs_since_lr_change = 0
        self.healthy_count = 0
        self.slow_count = 0
        self.stagnant_count = 0
        if reduction:
            self.last_reduction_ema = self.ema_loss
            self.lr_before_reduction = lr_before
            self.recoveries_since_reduction = 0

    def _apply_budget(self) -> bool:
        """Publish min(reactive rate, envelope). Returns whether the applied rate moved."""
        if self.envelope == "linear":
            self.envelope_lr = linear_decay_lr(
                self.n_steps,
                self.total_epochs,
                self.base_lr,
                self.envelope_eta_min,
            )
            applied = min(self.reactive_lr, self.envelope_lr)
        else:
            self.envelope_lr = None
            applied = self.reactive_lr
        applied = clip_lr(applied, self.min_lr, self.max_lr)
        changed = abs(applied - self.lr) > 1e-12
        self.lr = float(applied)
        self._write_lr(self.lr)
        return changed

    def _write_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _record(
        self,
        *,
        val_loss: float | None,
        val_accuracy: float | None,
        velocity: float | None,
        ratio: float | None,
        regime: str,
        action: str,
        reason: str,
        lr_changed: bool,
    ) -> dict[str, Any]:
        return {
            "epoch": self.n_steps,
            "lr": self.lr,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "ema_loss": self.ema_loss,
            "ema_accuracy": self.ema_accuracy,
            "velocity": velocity,
            "reference_velocity": self.reference_velocity,
            "normalized_progress": ratio,
            "regime": regime,
            "action": action,
            "reason": reason,
            "lr_changed": lr_changed,
            "reactive_lr": self.reactive_lr,
            "envelope_lr": self.envelope_lr,
            "healthy_count": self.healthy_count,
            "slow_count": self.slow_count,
            "stagnant_count": self.stagnant_count,
            "epochs_since_lr_change": self.epochs_since_lr_change,
            "recoveries_since_reduction": self.recoveries_since_reduction,
        }


class ConstantLR:
    """Epoch stepper that leaves the optimizer learning rate unchanged."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def step(self, metric: float | None = None) -> None:
        del metric

    def state_dict(self) -> dict[str, Any]:
        return {"lr": float(self.optimizer.param_groups[0]["lr"])}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        lr = float(state["lr"])
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


class LinearDecayLR:
    """Decay the learning rate linearly from the optimizer's initial rate to ``eta_min``.

    Timing matches :class:`torch.optim.lr_scheduler.CosineAnnealingLR`: the
    first epoch trains at the peak, and each ``step`` sets the rate for the
    following epoch. The rate is ``eta_min`` after ``total_epochs`` steps.
    """

    def __init__(self, optimizer: Optimizer, total_epochs: int, eta_min: float) -> None:
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1")
        if eta_min < 0:
            raise ValueError("eta_min must be >= 0")
        if not optimizer.param_groups:
            raise ValueError("optimizer has no parameter groups")
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.eta_min = float(eta_min)
        self.base_lr = float(optimizer.param_groups[0]["lr"])
        self.steps_taken = 0

    def get_lr(self) -> float:
        return linear_decay_lr(self.steps_taken, self.total_epochs, self.base_lr, self.eta_min)

    def step(self, metric: float | None = None) -> None:
        del metric
        self.steps_taken += 1
        self._write(self.get_lr())

    def state_dict(self) -> dict[str, Any]:
        return {
            "steps_taken": self.steps_taken,
            "base_lr": self.base_lr,
            "eta_min": self.eta_min,
            "total_epochs": self.total_epochs,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.steps_taken = int(state["steps_taken"])
        self.base_lr = float(state["base_lr"])
        self.eta_min = float(state["eta_min"])
        self.total_epochs = int(state["total_epochs"])
        self._write(self.get_lr())

    def _write(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)


def _milestones(total_epochs: int, fractions: list[float]) -> list[int]:
    points = set()
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"milestone fraction must be in (0, 1), got {fraction}")
        epoch = int(round(fraction * total_epochs))
        if 0 < epoch < total_epochs:
            points.add(epoch)
    return sorted(points)


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    config: dict[str, Any],
    total_epochs: int,
) -> Any:
    """Build one of the compared schedulers from a campaign config mapping.

    ``ablr`` and ``ablr_recovery`` share the ``ablr`` hyperparameter block.
    Recovery is forced off for ``ablr`` and on for ``ablr_recovery``.
    ``ablr_hybrid`` uses that same block, forces recovery off, and caps the
    rate with the linear envelope from the ``linear`` block. The hybrid peak
    is the optimizer's current learning rate, even if ``ablr.max_lr`` is lower.
    """
    if name not in SCHEDULER_NAMES:
        raise KeyError(f"unknown scheduler {name!r}; expected one of {SCHEDULER_NAMES}")
    if name == "constant":
        return ConstantLR(optimizer)
    if name == "step":
        step_cfg = config["step"]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=_milestones(total_epochs, list(step_cfg["milestone_fractions"])),
            gamma=float(step_cfg["gamma"]),
        )
    if name == "cosine":
        cosine_cfg = config["cosine"]
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_epochs)),
            eta_min=float(cosine_cfg["eta_min"]),
        )
    if name == "linear":
        return LinearDecayLR(
            optimizer,
            total_epochs=int(total_epochs),
            eta_min=float(config["linear"]["eta_min"]),
        )
    if name == "plateau":
        plateau_cfg = config["plateau"]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(plateau_cfg["factor"]),
            patience=int(plateau_cfg["patience"]),
            threshold=float(plateau_cfg["threshold"]),
            threshold_mode=str(plateau_cfg["threshold_mode"]),
            min_lr=float(plateau_cfg["min_lr"]),
        )
    ablr_cfg = dict(config["ablr"])
    ablr_cfg["enable_recovery"] = name == "ablr_recovery"
    if name == "ablr_hybrid":
        peak = float(optimizer.param_groups[0]["lr"])
        ablr_cfg["enable_recovery"] = False
        ablr_cfg["envelope"] = "linear"
        ablr_cfg["total_epochs"] = int(total_epochs)
        ablr_cfg["envelope_eta_min"] = float(config["linear"]["eta_min"])
        ablr_cfg["max_lr"] = max(float(ablr_cfg["max_lr"]), peak)
    return AdaptiveBounceLR(optimizer, **ablr_cfg)


def simulate_ablr(
    losses: list[float],
    accuracies: list[float | None] | None = None,
    *,
    lr: float = 0.1,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Step ABLR on a hand-built metric sequence.

    The sequence is not training data. It exists so the decision rule can be
    inspected without fitting a network. Results from this helper must not be
    reported as model accuracy.
    """
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=lr)
    scheduler = AdaptiveBounceLR(optimizer, **kwargs)
    if accuracies is None:
        accuracies = [None] * len(losses)
    if len(accuracies) != len(losses):
        raise ValueError("accuracies and losses must have the same length")
    for loss, accuracy in zip(losses, accuracies):
        scheduler.step(loss, accuracy)
    return scheduler.history


def step_scheduler(scheduler: Any, name: str, val_loss: float, val_accuracy: float | None) -> Any:
    """Step a scheduler with validation metrics.

    Plateau and ABLR receive the validation loss. ABLR also receives accuracy
    for logging. Epoch schedules (constant, step, cosine, linear) ignore the
    metrics. The test set is never passed here.
    """
    if isinstance(scheduler, AdaptiveBounceLR):
        return scheduler.step(val_loss, val_accuracy)
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
        return None
    scheduler.step()
    return None
