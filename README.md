# Adaptive Bounce LR

[![Tests](https://github.com/Tarun-Upadhyay05/adaptive-bounce-lr/actions/workflows/tests.yml/badge.svg)](https://github.com/Tarun-Upadhyay05/adaptive-bounce-lr/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-c6f432.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-35d9e8.svg)](pyproject.toml)

Experimental learning-rate scheduler that reacts to the velocity of validation-loss improvement, compared with standard schedules on CIFAR-10.

Adaptive Bounce LR (ABLR) is a small research project, not a drop-in replacement for cosine annealing or `ReduceLROnPlateau`. It asks whether a scheduler that measures how fast validation loss is still falling can train a CIFAR-10 ResNet-18 more effectively than schedules that ignore that signal. The answer is empirical and is reported only from runs that actually finished.

**Portfolio note.** The technical contribution is the separation of three decisions that are easy to tangle: ABLR adapts the learning rate from smoothed validation loss, checkpoints are chosen by validation accuracy, and the official test set is read once at the end. A second variant adds a single capped learning-rate recovery so the bounce itself can be ablated.

## Problem

Most learning-rate schedules are functions of the epoch index. `ReduceLROnPlateau` is the common exception, and it only knows how to cut the rate after a stall. ABLR is an experiment in using the *rate* of improvement, not only the presence of a plateau:

- start at the usual CIFAR learning rate of 0.1 so early progress is fast
- decay gradually when improvement slows relative to recent progress
- decay more strongly when the stall persists
- optionally take one small step back up if, after a cut, smoothed loss is clearly falling again
- never leave the configured minimum and maximum

That last upward step is the "bounce." It is off by default. Turning it on does not make the method better unless a completed ablation says so.

## Intuition

Picture a ball released near the top of a hill. A large learning rate is the steep part of the roll. As the slope flattens, the same step size overshoots or wastes updates, so the rate comes down. A long flat stretch is not the same event as a merely slower slope, so the cut is larger. If the ball then finds a slope again, a little of the removed step size can be restored, but not more than was just taken away, and not above the maximum. Noise in the height measurement should not reverse the ball on every bump, which is why the rule smooths the loss, waits for a streak, and then cools down.

## Mathematical formulation

Validation loss \(\ell_t\) is the only quantity that can change the learning rate. Validation accuracy is smoothed and stored so it can be plotted. It is not an input to the rule, because accuracy can sit still while the loss is still falling, and it often ticks up after a cut that should stay in place.

Smoothed loss:

\[
\tilde{\ell}_t = \alpha \ell_t + (1-\alpha)\tilde{\ell}_{t-1}
\]

Relative improvement over a window of \(W\) epochs:

\[
v_t = \frac{\tilde{\ell}_{t-W} - \tilde{\ell}_t}{\max(|\tilde{\ell}_{t-W}|, \epsilon)}
\]

Positive \(v_t\) means the smoothed loss fell. Dividing by the loss level stops late training from looking "stagnant" only because absolute loss changes are small.

The reference \(r\) is a slow average of earlier positive improvements. The current velocity is compared with the reference *before* the reference is updated:

\[
\rho_t = \frac{v_t}{\max(r_{\text{prev}}, \epsilon_v)}
\]

\[
r \leftarrow \beta v_t + (1-\beta) r \quad \text{when } v_t > 0
\]

A small \(\beta\) remembers the earlier, faster pace. A large \(\beta\) redefines "healthy" so quickly that the scheduler rarely reacts.

| Regime | Condition |
| --- | --- |
| Healthy | \(\rho_t \ge \rho_{\text{healthy}}\) |
| Slowing | \(\rho_{\text{slow}} \le \rho_t < \rho_{\text{healthy}}\) |
| Stagnant | \(\rho_t < \rho_{\text{slow}}\) |

A regime must hold for its patience count before the learning rate changes. During the warm-up, statistics update and the rate does not. After any change, the next `cooldown_epochs` epochs cannot change the rate, and patience counters do not accumulate during that cooldown.

| Action | Update, then clip to \([\eta_{\min}, \eta_{\max}]\) |
| --- | --- |
| Slowing | \(\eta \leftarrow \eta \cdot \gamma_{\text{slow}}\) |
| Stagnation | \(\eta \leftarrow \eta \cdot \gamma_{\text{stag}}\), with \(\gamma_{\text{stag}} \le \gamma_{\text{slow}} < 1\) |
| Recovery, optional | \(\eta \leftarrow \min(\eta_{\max}, \eta_{\text{before}}, \eta \cdot \gamma_{\text{recover}})\) |

Recovery runs only when all of the following are true:

1. The recovery variant is enabled.
2. A reduction has already happened.
3. The cooldown after the last change has expired.
4. The healthy regime has held for `healthy_patience` consecutive decision epochs since that cooldown.
5. Smoothed loss has improved since the reduction by a relative margin \(\delta\).
6. Fewer than `max_recoveries` bounces have been used since that reduction.
7. The capped proposal is strictly larger than the current rate.

A rise in validation accuracy is not one of these conditions.

Missing or non-finite losses are ignored. They do not move the averages, the counters, or the rate. If a cut would fall below \(\eta_{\min}\), the rate is set to the floor once. Further cuts while the rate is already there are recorded as saturation and do not count as new changes.

Hyperparameters and what they do:

| Name | Role |
| --- | --- |
| `ema_alpha` | Higher tracks the latest epoch more closely and admits more noise. |
| `velocity_window` | Longer windows ignore brief spikes and react later. |
| `reference_beta` | Higher accepts a new slower pace as normal. Lower keeps cutting because early progress was faster. |
| `healthy_ratio`, `slow_ratio` | Boundaries between healthy, slowing, and stagnant. |
| `slow_patience`, `stagnant_patience`, `healthy_patience` | Hysteresis. A single noisy epoch cannot act. |
| `slow_factor`, `stagnant_factor` | Multiplicative cuts. Stagnation is at least as strong as slowing. |
| `recover_factor`, `recover_margin`, `max_recoveries` | Size, evidence, and count of a bounce. |
| `min_lr`, `max_lr` | Hard bounds. |
| `warmup_epochs` | Epochs that only collect statistics. |
| `cooldown_epochs` | Epochs after a change during which another change is suppressed. |

Variant A is `ablr` (`enable_recovery: false`). Variant B is `ablr_recovery` (the same block with recovery forced on).

`linear` ignores validation loss. It decays from the optimizer's initial learning rate to `eta_min` in equal steps across the epoch budget. The first epoch uses the peak. The rate reaches `eta_min` on the step after the last epoch, the same timing as cosine annealing.

`ablr_hybrid` is variant A capped by that line: after every epoch, including warm-up and cooldown, the published rate is `min(reactive_lr, linear_envelope)`. The reactive rule can cut earlier than the line. It cannot leave the rate at the peak through the end of the budget. Recovery is off. The peak is the optimizer learning rate, so a campaign can start at 0.15 without editing `max_lr` up by hand. Momentum stays 0.9. A larger momentum widens the stationary bounce around the minimum, which is the floor this cap is there to get under.

## Architecture

CIFAR-10, 10-way classification. The network is torchvision ResNet-18 with the ImageNet stem replaced by a 3×3 stride-1 convolution and no max-pool, so the 32×32 input is not immediately downsampled. Training augmentation is `RandomCrop(32, padding=4)` and `RandomHorizontalFlip`. Inputs are normalized with the usual CIFAR-10 training-set statistics, mean `(0.4914, 0.4822, 0.4465)` and std `(0.2023, 0.1994, 0.2010)`, computed on the full official training set before the validation split.

The optimizer for every scheduler in the benchmark is SGD with learning rate 0.1, momentum 0.9, and weight decay `5e-4`. `configs/envelope.yaml` uses the same momentum and weight decay with learning rate 0.15. The validation split is 10% of the official training set, stratified, with split seed 42 frozen across schedulers and training seeds. The official test set is 10,000 images and is not used for scheduler steps, checkpoint selection, or threshold timing.

Checkpoint selection maximizes validation accuracy and breaks ties with lower validation loss. The reported test accuracy is that checkpoint, not the weights from the final epoch. Threshold times use the first epoch whose validation accuracy crosses the target. A target that is never crossed is reported as not reached.

```
configs/          quick, smoke, benchmark, envelope, notebook demo
src/scheduler.py  ABLR and the baseline builders
src/data.py       CIFAR-10, stratified split, augmentation
src/models.py     CIFAR ResNet-18
src/train.py      epoch loop (train and validation only)
src/evaluate.py   test metrics and aggregation
src/experiment.py multi-seed runner, resume, skip
src/plotting.py   figures and the data behind them
notebooks/        narrative notebook
tests/            scheduler rule and training pipeline
```

## Installation

Python 3.11 or newer. This project was developed on Windows 11 with Python 3.13 and PyTorch built for CUDA, on an RTX 4060 Laptop GPU (8 GB). CPU is used automatically when CUDA is absent.

```powershell
python -m pip install -r requirements.txt
```

Install a CUDA build of PyTorch from the official wheel index if `torch.cuda.is_available()` is false and you have an NVIDIA GPU. Do not expect the default PyPI CPU wheel to use the RTX 4060.

CIFAR-10 is downloaded on first use into `data/`. The official archive MD5 is `c58f30108f718f92721af3b95e74349a`. If the Toronto host is slow, download that same archive from a mirror and extract it so `data/cifar-10-batches-py/` exists before training:

```powershell
python scripts/download_cifar10.py --url https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz
```

The script refuses to extract an archive whose checksum does not match.

## Usage

From the project root:

```powershell
python -m pytest
python scripts/run_experiments.py --config configs/smoke.yaml
python scripts/run_experiments.py --config configs/quick.yaml
python scripts/run_experiments.py --config configs/benchmark.yaml
python scripts/run_experiments.py --config configs/envelope.yaml
python scripts/plot_results.py --config-name quick
jupyter notebook notebooks/adaptive_bounce_lr.ipynb
```

`smoke.yaml` is one epoch of constant learning rate, used to time the machine. `quick.yaml` is three epochs and one seed for every scheduler, used to prove the pipeline. `benchmark.yaml` is twenty epochs and seeds 0, 1, and 2 at learning rate 0.1. `envelope.yaml` is the same budget with the peak learning rate at 0.15, comparing constant, cosine, linear decay, and ABLR under the linear envelope. Finished runs are skipped. An interrupted run resumes from `final.pt` when the configuration fingerprint matches. `--force` deletes that run and starts over. `--epochs`, `--seeds`, and `--schedulers` override the file.

Quick overrides:

```powershell
python scripts/run_experiments.py --config configs/quick.yaml --epochs 1 --schedulers constant ablr
```

Use a new `name` in the YAML if you change the epoch budget. The fingerprint includes the budget, so a checkpoint from a different budget will not be reused.

## Experimental methodology

Shared by every scheduler: model, optimizer, batch size 128, augmentation, normalization, epoch budget, validation split, mixed precision on CUDA, and the test-evaluation protocol. Step decay fires at 50% and 75% of the epoch budget with \(\gamma = 0.1\). Cosine annealing runs for the whole budget down to `1e-5`. `ReduceLROnPlateau` watches validation loss with a relative threshold of 0.01. Its patience is 1 in the quick file (so a short run can move) and 4 in the benchmark file. ABLR settings for the benchmark are in `configs/benchmark.yaml`: warm-up 3, velocity window 4, cooldown 2, slow factor 0.5, stagnation factor 0.1, recovery factor 1.1 and margin 0.02 when recovery is on.

Seeds are set for Python, NumPy, and PyTorch. cuDNN benchmarking is off and deterministic algorithms are requested with `warn_only=True`, so a kernel without a deterministic implementation warns instead of aborting the run. That means bit-identical CUDA results are not guaranteed. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set by the experiment script before PyTorch is imported. DataLoader workers default to 0 on Windows so worker seeding is not an extra source of variation.

Training may use automatic mixed precision on CUDA. Validation and test losses are computed in float32 so the scheduler does not react to autocast rounding. The learning rate logged for an epoch is the rate used during that epoch. The scheduler updates it afterward, and the new rate is what the next epoch uses.

Aggregation reports the mean and the sample standard deviation across seeds. The standard deviation is omitted when only one seed finished. Time-to-target means include only seeds that reached the target, and the table shows how many seeds that was. Wall-clock time and epoch index are both recorded because a run can be fast in epochs and slow in seconds, or the reverse.

Package versions and the GPU name are written to `results/runs/<campaign>/environment.json`.

## Experimental results

Numbers below come from completed runs on this machine. A campaign that is not listed here had not finished when the README was written. Synthetic scheduler traces in the notebook are labeled as synthetic and are not in these tables.

The campaign writer also saves `results/runs/<name>/RESULTS.md` from the same summaries. Re-running `python scripts/plot_results.py --config-name <name>` refreshes figures without training.

### Smoke

One epoch, constant learning rate, seed 0, RTX 4060 Laptop. This only checks that training, validation, and the final test evaluation run.

| Scheduler | Seeds | Best val accuracy | Test accuracy | Test loss | Time (s) | LR changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 1 | 0.4338 | 0.4403 | 1.5199 | 54.5 | 0.00 |

Validation accuracy 0.40 was reached at epoch 1. Validation accuracy 0.60 was not reached.

### Quick

Three epochs, one seed, every scheduler. Epoch 1 is identical across schedulers, which is the expected shared initialization. These numbers are a pipeline check. They are not a ranking.

| Scheduler | Seeds | Best val accuracy | Test accuracy | Test loss | Time (s) | LR changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 1 | 0.6516 | 0.6533 | 0.9968 | 136.7 | 0.00 |
| step | 1 | 0.7070 | 0.7042 | 0.8386 | 156.0 | 1.00 |
| cosine | 1 | 0.6844 | 0.6823 | 0.8966 | 119.5 | 3.00 |
| plateau | 1 | 0.6516 | 0.6533 | 0.9968 | 155.6 | 0.00 |
| ablr | 1 | 0.6516 | 0.6533 | 0.9968 | 115.7 | 0.00 |
| ablr_recovery | 1 | 0.6516 | 0.6533 | 0.9968 | 158.9 | 0.00 |

No scheduler reached validation accuracy 0.80. Constant, plateau, ABLR, and ABLR with recovery kept learning rate 0.1 for all three epochs, so their metrics match. Step decay dropped to 0.01 after epoch 2 because the quick budget puts a milestone at half of three epochs. Cosine annealing changed the rate every epoch because that is what a three-epoch cosine does. Plateau did not fire because validation loss was still falling. ABLR did not fire because epoch 1 is warm-up and the later epochs were still healthy progress, short of the patience of 2. Figures are in `figures/quick/`.

### Benchmark

Twenty epochs, seeds 0, 1, and 2, RTX 4060 Laptop. Test accuracy uses the checkpoint with the best validation accuracy. Sample standard deviation is across the three seeds.

| Scheduler | Seeds | Best val accuracy | Test accuracy | Test loss | Time (s) | LR changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 3 | 0.8430 ± 0.0073 | 0.8379 ± 0.0109 | 0.4904 ± 0.0402 | 968.5 ± 27.0 | 0.00 ± 0.00 |
| step | 3 | 0.9126 ± 0.0009 | 0.9077 ± 0.0025 | 0.2788 ± 0.0045 | 872.3 ± 160.0 | 2.00 ± 0.00 |
| cosine | 3 | 0.9256 ± 0.0046 | 0.9190 ± 0.0016 | 0.2573 ± 0.0034 | 885.5 ± 199.5 | 20.00 ± 0.00 |
| plateau | 3 | 0.8905 ± 0.0390 | 0.8825 ± 0.0407 | 0.3500 ± 0.1248 | 658.1 ± 0.5 | 0.67 ± 0.58 |
| ablr | 3 | 0.8430 ± 0.0073 | 0.8379 ± 0.0109 | 0.4904 ± 0.0402 | 1010.5 ± 25.2 | 0.33 ± 0.58 |
| ablr_recovery | 3 | 0.8430 ± 0.0073 | 0.8379 ± 0.0109 | 0.4904 ± 0.0402 | 885.2 ± 166.7 | 0.33 ± 0.58 |

On this budget, cosine annealing had the highest test accuracy and the smallest spread. Step decay was next. `ReduceLROnPlateau` was higher than a constant rate on average and more variable across seeds. ABLR, with and without recovery, matched the constant-rate runs. The only ABLR cut was on seed 2, after epoch 20 had already been trained: validation loss was classified as sustained stagnation and the next rate was set to 0.01. That rate was never used, and the selected checkpoint was earlier (epoch 14). Recovery therefore had no cut to undo. Validation accuracy 0.85 was reached by step and cosine on all three seeds, by plateau on 2 of 3, and was not reached by constant or either ABLR variant. Figures are in `figures/benchmark/`.

This does not show that ABLR is better than the baselines. With these hyperparameters it mostly stayed at 0.1 while validation loss was still noisy but trending down, so it behaved like a constant schedule. Cosine's 20 learning-rate updates are the cosine schedule itself, not instability.

## Figures

Plot scripts write PNG and PDF, plus a CSV of the plotted values, under `figures/<campaign>/`.

- `loss_curves` and `accuracy_curves`: training and validation, mean across seeds, with a band when at least two seeds exist
- `learning_rates`: learning rate used during each epoch
- `overlay_<scheduler>`: validation loss and learning rate for the lowest seed, with change points marked
- `best_val_accuracy` and `test_accuracy`: mean and sample standard deviation
- `per_seed_test_accuracy`: every seed, so a mean is not the only view
- `time_to_target`: wall-clock time for seeds that reached each validation-accuracy target
- `ablation_recovery`: ABLR with and without recovery
- `ablr_decision_timeline`: normalized progress, regime, and learning rate for the lowest ABLR seed

Cosine annealing changes the learning rate every epoch by construction. Its change count is not a stability score against event-driven schedules.

## Limitations and failure cases

- ABLR has more knobs than cosine annealing. A bad `reference_beta` either never reacts or walks the rate down to the floor because early progress was faster than late progress.
- Recovery can undo a useful cut if `recover_margin` is too small. That is why it is a separate variant.
- Validation loss is noisy. Patience and cooldown reduce chatter; they also delay the response.
- The benchmark budget is 20 epochs on one laptop GPU, not a 200-epoch CIFAR leaderboard run. Thresholds such as 85% validation accuracy may be reported as not reached. That is a result, not a missing value to fill in.
- One architecture and one dataset cannot support a general claim.
- CUDA deterministic mode is best effort. A warning during training means some kernels did not have a deterministic implementation.
- Resuming restores model, optimizer, scheduler, and RNG state. Changing workers, batch size, or the epoch budget changes the fingerprint and will not silently continue an old run.
- This repository does not claim that ABLR is new, state of the art, or better than the baselines. Related ideas include step decay, cosine annealing (including warm restarts), and plateau reduction. The specific rule is an experiment.

## Reproducibility

```powershell
python -m pytest
python scripts/run_experiments.py --config configs/benchmark.yaml
python scripts/plot_results.py --config-name benchmark
```

Recorded with each campaign: the resolved config, the environment, the frozen split counts, per-epoch history, the ABLR decision trace, the best validation checkpoint, and the final resume checkpoint. Test metrics are in `summary.json`.

Open `notebooks/adaptive_bounce_lr.ipynb` from the project root, or launch Jupyter as shown above. The notebook runs the unit tests, a synthetic plateau-and-recovery demonstration, a short truncated training demo, and then loads whatever full campaigns are already on disk. It does not invent a finished benchmark.
