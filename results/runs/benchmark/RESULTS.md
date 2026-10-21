Results below are computed from completed `benchmark` runs only.

| Scheduler | Seeds | Best val accuracy | Test accuracy | Test loss | Time (s) | LR changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 3 | 0.8430 ± 0.0073 | 0.8379 ± 0.0109 | 0.4904 ± 0.0402 | 968.5 ± 27.0 | 0.00 ± 0.00 |
| step | 3 | 0.9126 ± 0.0009 | 0.9077 ± 0.0025 | 0.2788 ± 0.0045 | 872.3 ± 160.0 | 2.00 ± 0.00 |
| cosine | 3 | 0.9256 ± 0.0046 | 0.9190 ± 0.0016 | 0.2573 ± 0.0034 | 885.5 ± 199.5 | 20.00 ± 0.00 |
| plateau | 3 | 0.8905 ± 0.0390 | 0.8825 ± 0.0407 | 0.3500 ± 0.1248 | 658.1 ± 0.5 | 0.67 ± 0.58 |
| ablr | 3 | 0.8430 ± 0.0073 | 0.8379 ± 0.0109 | 0.4904 ± 0.0402 | 1010.5 ± 25.2 | 0.33 ± 0.58 |
| ablr_recovery | 3 | 0.8430 ± 0.0073 | 0.8379 ± 0.0109 | 0.4904 ± 0.0402 | 885.2 ± 166.7 | 0.33 ± 0.58 |

Validation accuracy thresholds. Time is the mean only over seeds that reached the target.

| Scheduler | Threshold | Reached | Epochs (reached seeds) | Wall time s (reached seeds) |
| --- | ---: | ---: | ---: | ---: |
| constant | 0.7 | 3/3 | 5.00 | 243.3 |
| constant | 0.8 | 3/3 | 8.00 | 385.5 |
| constant | 0.85 | 0/3 | not reached | not reached |
| step | 0.7 | 3/3 | 5.00 | 218.9 |
| step | 0.8 | 3/3 | 8.00 | 360.3 |
| step | 0.85 | 3/3 | 11.00 | 482.8 |
| cosine | 0.7 | 3/3 | 4.00 | 180.7 |
| cosine | 0.8 | 3/3 | 7.33 | 321.2 |
| cosine | 0.85 | 3/3 | 12.67 | 569.9 |
| plateau | 0.7 | 3/3 | 5.00 | 164.7 |
| plateau | 0.8 | 3/3 | 8.00 | 263.4 |
| plateau | 0.85 | 2/3 | 16.00 | 526.5 |
| ablr | 0.7 | 3/3 | 5.00 | 241.7 |
| ablr | 0.8 | 3/3 | 8.00 | 395.1 |
| ablr | 0.85 | 0/3 | not reached | not reached |
| ablr_recovery | 0.7 | 3/3 | 5.00 | 209.6 |
| ablr_recovery | 0.8 | 3/3 | 8.00 | 354.7 |
| ablr_recovery | 0.85 | 0/3 | not reached | not reached |

A standard deviation is reported only when at least two seeds finished. A single seed is a point estimate, not a distribution.
Cosine annealing changes the learning rate every epoch by design, so its LR-change count is not a stability comparison against event-driven schedules.
