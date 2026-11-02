
### PRE-WEEKEND (no target-weekend data) - season 2023, 22 races
| model       |   winner |   podium |   top10 |   pairwise |   mae |
|:------------|---------:|---------:|--------:|-----------:|------:|
| ens         |    0.864 |    0.53  |   0.764 |      0.734 | 3.741 |
| form_d3     |    0.864 |    0.515 |   0.755 |      0.723 | 3.836 |
| pre_lgb     |    0.818 |    0.5   |   0.768 |      0.73  | 3.755 |
| pre_logit   |    0.864 |    0.515 |   0.777 |      0.729 | 3.782 |
| prev_podium |    0.727 |    0.53  |   0.659 |      0.654 | 4.527 |
| standings   |    0.864 |    0.545 |   0.759 |      0.725 | 3.805 |
final model (ens) podium = 0.530  95% CI [0.470, 0.591];  races >= 75%: 0/22;  ORACLE ceiling (true race pace known, diagnostic) = 0.742

### PRE-WEEKEND (no target-weekend data) - season 2024, 24 races
| model       |   winner |   podium |   top10 |   pairwise |   mae |
|:------------|---------:|---------:|--------:|-----------:|------:|
| ens         |    0.333 |    0.569 |   0.796 |      0.764 | 3.402 |
| form_d3     |    0.375 |    0.528 |   0.783 |      0.755 | 3.461 |
| pre_lgb     |    0.333 |    0.528 |   0.792 |      0.76  | 3.393 |
| pre_logit   |    0.292 |    0.528 |   0.796 |      0.758 | 3.49  |
| prev_podium |    0.167 |    0.417 |   0.733 |      0.706 | 4.043 |
| standings   |    0.375 |    0.556 |   0.796 |      0.756 | 3.477 |
final model (ens) podium = 0.569  95% CI [0.458, 0.667];  races >= 75%: 3/24;  ORACLE ceiling (true race pace known, diagnostic) = 0.736

### PRE-WEEKEND (no target-weekend data) - season 2025, 24 races
| model       |   winner |   podium |   top10 |   pairwise |   mae |
|:------------|---------:|---------:|--------:|-----------:|------:|
| ens         |    0.208 |    0.653 |   0.729 |      0.716 | 4.005 |
| form_d3     |    0.292 |    0.625 |   0.696 |      0.698 | 4.188 |
| pre_lgb     |    0.292 |    0.569 |   0.708 |      0.709 | 4.013 |
| pre_logit   |    0.25  |    0.625 |   0.717 |      0.718 | 3.954 |
| prev_podium |    0.333 |    0.556 |   0.675 |      0.658 | 4.58  |
| standings   |    0.167 |    0.667 |   0.696 |      0.697 | 4.235 |
final model (ens) podium = 0.653  95% CI [0.583, 0.722];  races >= 75%: 3/24;  ORACLE ceiling (true race pace known, diagnostic) = 0.736

### PRE-WEEKEND (no target-weekend data) - season 2026, 19 races
| model       |   winner |   podium |   top10 |   pairwise |   mae |
|:------------|---------:|---------:|--------:|-----------:|------:|
| ens         |    0.368 |    0.544 |   0.779 |      0.771 | 3.689 |
| form_d3     |    0.263 |    0.491 |   0.742 |      0.745 | 3.919 |
| pre_lgb     |    0.316 |    0.579 |   0.768 |      0.756 | 3.732 |
| pre_logit   |    0.263 |    0.561 |   0.784 |      0.77  | 3.603 |
| prev_podium |    0.368 |    0.509 |   0.674 |      0.683 | 4.617 |
| standings   |    0.368 |    0.526 |   0.758 |      0.755 | 3.876 |
final model (ens) podium = 0.544  95% CI [0.439, 0.632];  races >= 75%: 1/19;  ORACLE ceiling (true race pace known, diagnostic) = 0.789

### Does accuracy improve across the season? (final model, pooled over evaluated seasons)
| stage   |   mean |   size |
|:--------|-------:|-------:|
| R1-5    |  0.6   |     20 |
| R6-11   |  0.583 |     24 |
| R12+    |  0.563 |     45 |
slope of podium accuracy per round = -0.0018 (p=0.64); per-season stage means:
|   season |   R1-5 |   R6-11 |   R12+ |
|---------:|-------:|--------:|-------:|
|     2023 |  0.667 |   0.611 |  0.424 |
|     2024 |  0.667 |   0.5   |  0.564 |
|     2025 |  0.667 |   0.667 |  0.641 |
|     2026 |  0.4   |   0.556 |  0.625 |

### 2026 per round
|   round |   ens |   standings |   cum_ens |
|--------:|------:|------------:|----------:|
|       1 | 0     |       0.333 |     0     |
|       2 | 0.667 |       0.667 |     0.334 |
|       3 | 0.667 |       0.667 |     0.445 |
|       4 | 0.333 |       0.333 |     0.417 |
|       5 | 0.333 |       0.333 |     0.4   |
|       6 | 0.667 |       0.667 |     0.445 |
|       7 | 0.667 |       0.667 |     0.476 |
|       8 | 0.667 |       0.667 |     0.5   |
|       9 | 0.667 |       0.667 |     0.519 |
|      10 | 0.333 |       0.333 |     0.5   |
|      11 | 0.333 |       0.333 |     0.485 |
|      12 | 0.667 |       0.667 |     0.5   |
|      13 | 0.667 |       0.667 |     0.513 |
|      14 | 0.333 |       0.333 |     0.5   |
|      15 | 0.333 |       0.333 |     0.489 |
|      16 | 0.667 |       0.667 |     0.5   |
|      17 | 0.667 |       0.333 |     0.51  |
|      18 | 0.667 |       0.333 |     0.519 |
|      19 | 1     |       1     |     0.544 |