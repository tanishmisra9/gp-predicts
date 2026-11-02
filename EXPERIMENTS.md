# Experiment log

## E0 — Metric pre-registration (written BEFORE any model was run)

Prediction time = after qualifying/sprint and all practice, before lights-out (grid is public).

Grid-position baseline (no model), measured first:

| metric (per-race mean) | 2026 R2-19 | 2022-25 |
|---|---|---|
| winner correct | 0.667 | 0.554 |
| **podium overlap** (|pred top3 ∩ true top3|/3) | **0.611** | 0.674 |
| top-10 set overlap | 0.756 | 0.772 |
| pairwise ordering accuracy | 0.779 | 0.749 |
| exact podium set | 0.222 | 0.207 |

Top-10 overlap and pairwise accuracy are already >=75% from grid order alone => a 75% target on them is **trivial and
would say nothing about the model**. Rejected as the target.

**Primary target metric: podium accuracy = mean over races of |predicted top-3 ∩ actual top-3| / 3**, ranking drivers
using only pre-race information. Must also clearly beat the grid baseline (not just cross 75%).
Winner accuracy, exact-podium, top-10, pairwise, MAE are reported as secondary.

Protocol: dev seasons 2023-2025 (walk-forward, tune here only); **2026 is the held-out headline test** (walk-forward, 19 rounds,
no tuning on it). Every look at 2026 is recorded below.

## E1 — First models, Jolpica-only features (dev = 2023-25 walk-forward, 70 races; 2026 untouched)
Features: grid/quali, decayed driver+team form (hl 3 & 10 races), team quali gap, DNF rates, circuit grid-correlation, era handling.
Leakage test `tests/test_no_leak.py` passes and was mutation-checked (fails if a leak is injected).

| model | podium | winner | top10 | pairwise | MAE |
|---|---|---|---|---|---|
| grid baseline | 0.686 | 0.586 | 0.786 | 0.762 | 3.32 |
| lgb regress(order) | 0.643 | 0.543 | 0.797 | 0.776 | 3.17 |
| lgb podium clf | 0.690 | 0.614 | 0.780 | 0.764 | 3.35 |
| lgb lambdarank | 0.676 | 0.614 | 0.799 | 0.771 | 3.23 |
| logit podium | 0.652 | 0.614 | 0.803 | 0.781 | 3.14 |
| rank-avg ensemble | 0.676 | 0.614 | 0.800 | 0.782 | 3.14 |

Learned: models improve overall ordering a little (MAE/pairwise) but do NOT beat the grid on podium. The grid already holds
most pre-race information; standard error on 70 races is ~0.03 so differences <0.03 are noise. 75% podium is ~9 points above baseline.

## E2 — Compact (16-feature) logistic podium model vs. grid, dev 2022-25 (92 races)
grid: podium 0.674 / top10 0.772 / pairwise 0.749 / MAE 3.47. compact logit (C=0.3): podium 0.688 / top10 0.800 / pairwise 0.773 / MAE 3.22.
Small, consistent gain over grid on rank-quality; podium +1.4 pts (inside noise). Fewer features did not hurt (50-feature GBMs were no better).

## E3 — History weighting (old-era multiplier x training half-life), dev 2022-25
12 settings, podium range 0.678-0.692, all inside noise. 2022 regulation reset (the 2026 proxy) is also flat across settings.
Learned: how history is windowed/decayed is NOT a lever for this task. Kept defaults era_mult=(0.3 driver, 0.1 team), train half-life=100 races.
Next information source: OpenF1 lap-derived pace (FP/race clean-lap pace) — the only new signal beyond grid/results.

## E4 — OpenF1 lap-derived pace features (FP best/top5/long-run gap, decayed race clean-lap pace), dev 2023-25
Added to compact logit / LGBM podium. podium: logit compact 0.690 -> with laps 0.667 (C=.3) / 0.686 (C=.05); lgbm 0.657 -> 0.662. Grid 0.686.
Learned: **no gain.** Practice pace is noisy (fuel/engine modes/programmes) and quali+grid already captures car pace. Negative result.
Note: fastf1 unusable (3.9 needs F1 credentials; 3.3.8 cannot parse 2026) -> OpenF1 used instead.
Diagnosis of dev errors (compact logit, 2022-25): 87 of 276 predicted podium slots miss; 28% of false picks were retirements (noise),
and the true podium drivers we miss mostly started P4-P6 -> the open question is race-trim pace among the front-midfield.
