Results below are computed from completed `quick` runs only.

| Scheduler | Seeds | Best val accuracy | Test accuracy | Test loss | Time (s) | LR changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 1 | 0.6516 | 0.6533 | 0.9968 | 136.7 | 0.00 |
| step | 1 | 0.7070 | 0.7042 | 0.8386 | 156.0 | 1.00 |
| cosine | 1 | 0.6844 | 0.6823 | 0.8966 | 119.5 | 3.00 |
| plateau | 1 | 0.6516 | 0.6533 | 0.9968 | 155.6 | 0.00 |
| ablr | 1 | 0.6516 | 0.6533 | 0.9968 | 115.7 | 0.00 |
| ablr_recovery | 1 | 0.6516 | 0.6533 | 0.9968 | 158.9 | 0.00 |

Validation accuracy thresholds. Time is the mean only over seeds that reached the target.

| Scheduler | Threshold | Reached | Epochs (reached seeds) | Wall time s (reached seeds) |
| --- | ---: | ---: | ---: | ---: |
| constant | 0.4 | 1/1 | 1.00 | 45.4 |
| constant | 0.6 | 1/1 | 3.00 | 136.7 |
| constant | 0.8 | 0/1 | not reached | not reached |
| step | 0.4 | 1/1 | 1.00 | 52.3 |
| step | 0.6 | 1/1 | 3.00 | 156.0 |
| step | 0.8 | 0/1 | not reached | not reached |
| cosine | 0.4 | 1/1 | 1.00 | 34.4 |
| cosine | 0.6 | 1/1 | 3.00 | 119.5 |
| cosine | 0.8 | 0/1 | not reached | not reached |
| plateau | 0.4 | 1/1 | 1.00 | 51.7 |
| plateau | 0.6 | 1/1 | 3.00 | 155.6 |
| plateau | 0.8 | 0/1 | not reached | not reached |
| ablr | 0.4 | 1/1 | 1.00 | 48.4 |
| ablr | 0.6 | 1/1 | 3.00 | 115.7 |
| ablr | 0.8 | 0/1 | not reached | not reached |
| ablr_recovery | 0.4 | 1/1 | 1.00 | 51.4 |
| ablr_recovery | 0.6 | 1/1 | 3.00 | 158.9 |
| ablr_recovery | 0.8 | 0/1 | not reached | not reached |

A standard deviation is reported only when at least two seeds finished. A single seed is a point estimate, not a distribution.
Cosine annealing changes the learning rate every epoch by design, so its LR-change count is not a stability comparison against event-driven schedules.
