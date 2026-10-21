Results below are computed from completed `smoke` runs only.

| Scheduler | Seeds | Best val accuracy | Test accuracy | Test loss | Time (s) | LR changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 1 | 0.4338 | 0.4403 | 1.5199 | 54.5 | 0.00 |

Validation accuracy thresholds. Time is the mean only over seeds that reached the target.

| Scheduler | Threshold | Reached | Epochs (reached seeds) | Wall time s (reached seeds) |
| --- | ---: | ---: | ---: | ---: |
| constant | 0.4 | 1/1 | 1.00 | 54.5 |
| constant | 0.6 | 0/1 | not reached | not reached |

A standard deviation is reported only when at least two seeds finished. A single seed is a point estimate, not a distribution.
Cosine annealing changes the learning rate every epoch by design, so its LR-change count is not a stability comparison against event-driven schedules.
