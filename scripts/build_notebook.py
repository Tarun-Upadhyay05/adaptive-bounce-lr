"""Build notebooks/adaptive_bounce_lr.ipynb. Run from the project root."""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip() + "\n"))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip() + "\n"))


md(
    """
# Adaptive Bounce LR

Experimental comparison of a validation-velocity learning-rate rule against constant, step, cosine, and plateau schedules on CIFAR-10.

ABLR is a hypothesis, not a result. This notebook runs the scheduler on synthetic losses, checks the unit tests, trains a **truncated** demonstration, and then loads whatever full campaigns are already saved on disk. Synthetic traces are labeled as synthetic. Missing campaigns are left missing.
"""
)

code(
    """
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = Path.cwd().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import Image, Markdown, display

from src.data import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    class_counts,
    compute_channel_stats,
    load_display_dataset,
    one_example_per_class,
    stratified_indices,
)
from src.evaluate import format_results_markdown
from src.experiment import load_completed_summaries, load_config, run_campaign, write_aggregate
from src.models import count_trainable_parameters, resnet18_cifar
from src.plotting import plot_campaign, plot_class_examples, plot_decision_timeline
from src.scheduler import simulate_ablr
from src.utils import capture_environment

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
print("project root:", ROOT)
environment = capture_environment()
print(environment["python"], environment["torch"], environment["gpu_name"])
"""
)

md(
    """
## Motivation

A fixed schedule spends the same learning rate whether the loss is falling quickly or not moving. A plateau rule waits until progress has already stopped, then only cuts. The bouncing-ball picture is an attempt to use the speed of improvement as well: a large step while the ball is moving, a smaller step as it slows, a harder cut if it stops, and at most a small, capped step back up if it is clearly moving again. The upward step is optional, because putting energy back into the ball can just as easily kick it off the slope.
"""
)

md(
    """
## Research question and hypotheses

**Question.** On CIFAR-10 with a small-image ResNet-18, does a scheduler driven by the relative velocity of smoothed validation loss improve validation-selected test accuracy, or the time to reach a validation-accuracy target, relative to constant, step, cosine, and plateau schedules under a shared training recipe?

**Hypotheses, all falsifiable.**

1. Relative loss velocity, with smoothing and patience, can reduce the learning rate when progress slows without reacting to a single noisy epoch.
2. A stronger cut after a sustained stall is different in effect from the gradual cut. The unit tests lock that distinction. Whether it helps a network is a separate question.
3. A capped recovery after the loss starts falling again may help or may undo a useful cut. The two ABLR variants exist so this can be measured rather than assumed.
4. None of these hypotheses claim a new state of the art. A short run that does not beat cosine annealing is a valid outcome.
"""
)

md(
    """
## Related scheduler concepts

| Scheduler | What it watches | What it does |
| --- | --- | --- |
| Constant | Nothing | Keeps the initial SGD learning rate of 0.1. |
| Step decay | The epoch index | Multiplies the rate by 0.1 at 50% and 75% of the budget. |
| Cosine annealing | The epoch index | Follows a cosine from 0.1 down to 1e-5 across the budget. |
| ReduceLROnPlateau | Validation loss | Cuts by 10× after patience epochs without a sufficient improvement. It does not raise the rate. |
| ABLR | Smoothed validation-loss velocity | Holds, decays gradually, or decays sharply. Recovery is a separate variant. |

Cosine annealing changes the rate every epoch by design. Counting those updates next to ABLR's occasional cuts is not a stability comparison unless that difference is stated.
"""
)

md(
    r"""
## Mathematical formulation of ABLR

Let \(\ell_t\) be the validation loss. Accuracy is recorded and is not part of the update.

\[
\tilde{\ell}_t = \alpha \ell_t + (1-\alpha)\tilde{\ell}_{t-1}
\]

\[
v_t = \frac{\tilde{\ell}_{t-W}-\tilde{\ell}_t}{\max(|\tilde{\ell}_{t-W}|,\epsilon)}
\]

\[
\rho_t = \frac{v_t}{\max(r_{\mathrm{prev}}, \epsilon_v)}
\]

The reference \(r\) then moves only when \(v_t>0\), with a small \(\beta\), so it does not immediately forget that earlier epochs improved faster. \(\rho\) at or above `healthy_ratio` is healthy, between the two thresholds is slowing, and below `slow_ratio` is stagnant. Each regime must persist for its patience. Warm-up and cooldown do not change the rate.

Recovery, when enabled, requires a prior cut, an expired cooldown, a healthy streak, and a relative drop in smoothed loss since the cut of at least `recover_margin`. The new rate cannot exceed the rate from before that cut or `max_lr`. The implementation and a longer derivation are in `src/scheduler.py` and the README.
"""
)

md(
    """
## Architecture and methodology

ResNet-18 with a 3×3 stride-1 stem and no max-pool. SGD, momentum 0.9, weight decay 5e-4, initial learning rate 0.1, batch size 128. Ten percent of the official 50,000 training images are a stratified validation split with seed 42. Scheduler decisions and checkpoint selection use that split. The 10,000-image official test set is evaluated only after training, on the checkpoint with the highest validation accuracy (ties broken by lower validation loss).

Seeds cover Python, NumPy, and PyTorch. Deterministic CUDA kernels are requested in warn-only mode, so a remaining nondeterministic kernel does not abort the run and bit-identical GPU results are not guaranteed.
"""
)

md("## Dataset exploration")

code(
    """
display_train = load_display_dataset(ROOT / "data", train=True)
display_test = load_display_dataset(ROOT / "data", train=False)
targets = display_train.targets
train_idx, val_idx = stratified_indices(targets, val_fraction=0.1, seed=42)
print(f"official train {len(display_train)} | train after split {len(train_idx)} | val {len(val_idx)} | official test {len(display_test)}")
counts = pd.DataFrame({
    "train": class_counts(targets, train_idx),
    "validation": class_counts(targets, val_idx),
})
display(counts)
examples = one_example_per_class(display_train)
figure_path = plot_class_examples(examples, CIFAR10_CLASSES, ROOT / "figures" / "cifar10_examples")
display(Image(filename=str(figure_path)))
sample_mean, sample_std = compute_channel_stats(display_train, limit=2000)
print("published training-set mean/std used by the loaders:", CIFAR10_MEAN, CIFAR10_STD)
print("mean/std on the first 2000 training images (audit only):", sample_mean, sample_std)
"""
)

md("## Model setup")

code(
    """
model = resnet18_cifar()
print("conv1:", model.conv1)
print("maxpool:", model.maxpool)
print("trainable parameters:", f"{count_trainable_parameters(model):,}")
print("logits for a 32x32 batch:", tuple(model(torch.randn(2, 3, 32, 32)).shape))
"""
)

md(
    """
## Unit tests and scheduler sanity checks

The tests feed synthetic losses to ABLR. They check bounds, warm-up, healthy progress, gradual decay, strong decay, noise that alternates every epoch, minimum-rate saturation, recovery, the recovery cooldown, and checkpoint restore. A separate test trains one synthetic batch with every scheduler and checks that metrics and checkpoints are written. Those accuracies are not CIFAR-10 results.
"""
)

code(
    """
completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=line"], cwd=ROOT)
print("pytest exit code:", completed.returncode)
if completed.returncode != 0:
    raise RuntimeError("pytest failed")
"""
)

md(
    """
## Synthetic demonstration

The losses below are handwritten. They fall, sit still, then fall again. This is not a trained network. The figure shows whether the rule cuts during the stall and whether the recovery variant is willing to step back up afterward.
"""
)

code(
    """
synthetic_losses = [4.0, 2.0, 1.2, 1.2, 1.2, 1.2, 1.2, 0.9, 0.7, 0.55, 0.45, 0.38]
synthetic_settings = dict(
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
    cooldown_epochs=1,
    max_recoveries=1,
    min_reference_velocity=1e-4,
)
without = pd.DataFrame(simulate_ablr(synthetic_losses, enable_recovery=False, **synthetic_settings))
with_recovery = pd.DataFrame(simulate_ablr(synthetic_losses, enable_recovery=True, **synthetic_settings))
print("SYNTHETIC metrics, not a trained model")
print("without recovery, final lr", without["lr"].iloc[-1], "reasons", without["reason"].tolist())
print("with recovery, final lr", with_recovery["lr"].iloc[-1], "reasons", with_recovery["reason"].tolist())
timeline = plot_decision_timeline(
    with_recovery,
    ROOT / "figures" / "synthetic_ablr_recovery_timeline",
    title="SYNTHETIC metric sequence: ABLR with recovery",
)
display(Image(filename=str(timeline)))
display(with_recovery[["epoch", "val_loss", "velocity", "normalized_progress", "regime", "lr", "reason"]])
"""
)

md(
    """
## Truncated training demonstration

The next cell trains constant LR and ABLR for one epoch on 40 training batches and evaluates 10 validation batches, then reads the test loader once. It is a wiring demonstration. It is not the quick campaign and it is not the benchmark. Full data runs are loaded in the following section.
"""
)

code(
    """
demo_config = load_config(ROOT / "configs" / "notebook_demo.yaml")
demo_aggregate = run_campaign(demo_config, plot=False)
print("Truncated notebook_demo results (40 train batches, 10 eval batches, 1 epoch):")
display(Markdown(format_results_markdown(demo_aggregate, config_name="notebook_demo")))
demo_history = pd.read_csv(ROOT / "results" / "runs" / "notebook_demo" / "ablr" / "seed_0" / "history.csv")
display(demo_history)
"""
)

md(
    """
## Completed campaigns on disk

The quick campaign is three full-data epochs and one seed. The benchmark campaign is twenty full-data epochs and three seeds. Either file may be absent. Absence is reported. It is not filled with the truncated demo above.
"""
)

code(
    """
def show_campaign(name: str) -> None:
    summaries = load_completed_summaries(ROOT / "results", name)
    if summaries.empty:
        print(f"{name}: no completed runs on disk. This campaign was not executed, or it has not finished.")
        return
    aggregate = write_aggregate(ROOT / "results", name, summaries)
    print(f"{name}: loaded {len(summaries)} completed run(s) from disk")
    display(Markdown(format_results_markdown(aggregate, config_name=name)))
    plot_campaign(ROOT / "results", ROOT / "figures", name)

for campaign_name in ("quick", "benchmark"):
    show_campaign(campaign_name)
"""
)

md(
    """
## Visualization

Figures below are redrawn from the completed quick campaign when it exists, and from the benchmark campaign when that exists too. Axes show epoch or wall-clock seconds as labeled. Single-seed plots have no standard-deviation band. Time-to-target bars include only seeds that reached the target.
"""
)

code(
    """
def show_figures(campaign: str, names: list[str]) -> None:
    folder = ROOT / "figures" / campaign
    if not folder.exists():
        print(f"no figures for {campaign}")
        return
    print(campaign)
    for name in names:
        path = folder / f"{name}.png"
        if path.exists():
            display(Markdown(f"**{campaign} / {name}**"))
            display(Image(filename=str(path)))
        else:
            print("missing figure", path.name)

figure_names = [
    "loss_curves",
    "accuracy_curves",
    "learning_rates",
    "overlay_ablr",
    "overlay_ablr_recovery",
    "best_val_accuracy",
    "test_accuracy",
    "per_seed_test_accuracy",
    "time_to_target",
    "ablation_recovery",
    "ablr_decision_timeline",
    "ablr_recovery_decision_timeline",
]
for campaign_name in ("quick", "benchmark"):
    show_figures(campaign_name, figure_names)
"""
)

md(
    """
## Ablation: recovery on and off

Compare `ablr` and `ablr_recovery` in the tables above. They share every setting except the recovery switch. If only one seed finished, the difference is a single-run anecdote. If the benchmark campaign is missing, this notebook does not estimate it. A recovery that raises test accuracy in the truncated 40-batch demo would still not be evidence, because that demo does not see the full training set.
"""
)

code(
    """
def ablation_line(name: str) -> None:
    path = ROOT / "results" / "runs" / name / "aggregate.csv"
    if not path.exists():
        print(f"{name}: no aggregate file, so no ablation numbers are shown.")
        return
    frame = pd.read_csv(path)
    pair = frame.loc[frame["scheduler"].isin(["ablr", "ablr_recovery"])]
    columns = [
        "scheduler",
        "n_seeds",
        "epochs",
        "best_val_accuracy_mean",
        "best_val_accuracy_std",
        "test_accuracy_mean",
        "test_accuracy_std",
    ]
    print(name)
    display(pair[columns])

for campaign_name in ("notebook_demo", "quick", "benchmark"):
    ablation_line(campaign_name)
"""
)

md(
    """
## Limitations, failure cases, and future work

- `reference_beta` too large and the baseline chases the slowdown, so the rule rarely decays. Too small, and early fast progress makes the rest of training look stagnant until the rate sits on the floor.
- Recovery with a tiny margin can undo a cut that was helping. That is why variant A leaves it off.
- Patience avoids reacting to one noisy epoch and also delays a real stall.
- Twenty epochs on one laptop is not a 200-epoch CIFAR comparison. An unreached accuracy target stays unreached.
- One model and one dataset cannot support a general claim.
- CUDA determinism is best effort.
- A useful follow-up is a longer budget, more seeds, and a second architecture, with the same ban on reading the test set during training. It is not a search for a plot that makes ABLR look best.
"""
)

md(
    """
## Conclusion and reproducibility

Read the loaded tables as descriptions of the runs that finished. Do not promote ABLR over cosine annealing, step decay, or `ReduceLROnPlateau` unless those tables, with more than one seed, show a difference large enough to matter for the question you care about. The quick configuration is a pipeline check.

From the project root, after installing `requirements.txt`:

```powershell
python -m pytest
python scripts/run_experiments.py --config configs/quick.yaml
python scripts/run_experiments.py --config configs/benchmark.yaml
python scripts/plot_results.py --config-name benchmark
jupyter notebook notebooks/adaptive_bounce_lr.ipynb
```

Finished runs are skipped. Interrupted runs resume from `final.pt` when the configuration fingerprint matches. Details, the formula, and the hyperparameter table are in the README.
"""
)

nb.cells = cells
path = Path("notebooks/adaptive_bounce_lr.ipynb")
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, path)
print(f"wrote {path} with {len(cells)} cells")
