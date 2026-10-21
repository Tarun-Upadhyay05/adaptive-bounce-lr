Results below are computed from completed `envelope` runs only.

| Scheduler | Seeds | Best val accuracy | Test accuracy | Test loss | Time (s) | LR changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 3 | 0.8152 ± 0.0030 | 0.8091 ± 0.0012 | 0.5809 ± 0.0304 | 1044.4 ± 8.4 | 0.00 ± 0.00 |
| cosine | 3 | 0.9197 ± 0.0016 | 0.9142 ± 0.0031 | 0.2630 ± 0.0088 | 1176.7 ± 31.4 | 20.00 ± 0.00 |
| linear | 3 | 0.9165 ± 0.0029 | 0.9136 ± 0.0025 | 0.2644 ± 0.0109 | 17517.6 ± 28264.3 | 20.00 ± 0.00 |
| ablr_hybrid | 3 | 0.9165 ± 0.0029 | 0.9136 ± 0.0025 | 0.2644 ± 0.0109 | 1168.3 ± 3.6 | 20.00 ± 0.00 |

Validation accuracy thresholds. Time is the mean only over seeds that reached the target.

| Scheduler | Threshold | Reached | Epochs (reached seeds) | Wall time s (reached seeds) |
| --- | ---: | ---: | ---: | ---: |
| constant | 0.7 | 3/3 | 6.67 | 351.0 |
| constant | 0.8 | 3/3 | 15.67 | 819.8 |
| constant | 0.85 | 0/3 | not reached | not reached |
| cosine | 0.7 | 3/3 | 6.33 | 357.9 |
| cosine | 0.8 | 3/3 | 10.33 | 598.8 |
| cosine | 0.85 | 3/3 | 13.33 | 778.8 |
| linear | 0.7 | 3/3 | 6.67 | 16714.9 |
| linear | 0.8 | 3/3 | 10.67 | 16958.7 |
| linear | 0.85 | 3/3 | 14.00 | 17159.4 |
| ablr_hybrid | 0.7 | 3/3 | 6.67 | 390.5 |
| ablr_hybrid | 0.8 | 3/3 | 10.67 | 624.6 |
| ablr_hybrid | 0.85 | 3/3 | 14.00 | 818.8 |

A standard deviation is reported only when at least two seeds finished. A single seed is a point estimate, not a distribution.
Cosine annealing changes the learning rate every epoch by design, so its LR-change count is not a stability comparison against event-driven schedules.
