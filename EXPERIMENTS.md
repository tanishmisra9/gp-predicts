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

## E5 — Structural changes, dev 2022-25 (92 races). podium / pairwise / MAE
| variant | podium | pairwise | MAE |
|---|---|---|---|
| grid | 0.674 | 0.749 | 3.47 |
| compact logit (base) | 0.685 | 0.773 | 3.22 |
| A: train on finishers only, P(podium given finish) | 0.681 | 0.771 | 3.24 |
| B: A x P(finish) reliability model | 0.681 | 0.771 | 3.24 |
| C: + within-race rank & form-rank-minus-grid features | 0.688 | 0.773 | 3.22 |
| D: pairwise Bradley-Terry linear utility | 0.667 | 0.771 | 3.25 |
| D+rank features | 0.678 | 0.771 | 3.26 |

Learned: **plateau at ~0.685 (+~1 pt over grid) regardless of model class, label handling, features or history window.**
Four independent families (E2/E3 history, E4 lap pace, E5 labels/ranks/pairwise) all land inside the +-0.03 noise band.
The residual is race-day randomness (28% of false podium picks are retirements; strategy/safety cars/first-lap incidents).
Decision: freeze the simplest model (`compact_logit`) as FINAL before touching 2026; further tinkering would only fit noise.

## E6 — HELD-OUT 2026 walk-forward (single look; model frozen from E5; no tuning afterwards)
`python run.py` -> results/summary.md. 19 rounds, each trained only on strictly earlier races.

| 2026 (19 races) | compact_logit | grid | standings | prev_finish |
|---|---|---|---|---|
| **podium (target)** | **0.649** [0.526, 0.754] | 0.614 | 0.544 | 0.509 |
| winner | 0.684 | 0.684 | 0.474 | 0.368 |
| exact podium set | 0.263 | 0.211 | 0.053 | 0.053 |
| top-10 overlap | 0.768 | 0.753 | 0.763 | 0.674 |
| pairwise ordering | 0.792 | 0.775 | 0.763 | 0.683 |
| MAE (positions) | 3.33 | 3.44 | 3.77 | 4.61 |

Podium minus grid: +0.035, bootstrap CI [0.000, +0.088]. Races >= 75% podium: 5/19.
Dev seasons for context (podium, model vs grid): 2023 .667/.636, 2024 .667/.667, 2025 .736/.750.

**Outcome: the 75% podium target is NOT reached** (0.649 on 2026; 0.69 on 2023-25). Pairwise ordering is >75% (0.792) but the
grid alone gets 0.775, so that number reflects the metric, not the model (see E0).
Season-cycle: accuracy does not climb with more in-season data; early rounds (R1-5) are no worse than late (R12+),
because quali position already encodes the current car pace from round 1. In-season rounds help model vs grid only marginally.

## E7 — NEW SETTING (user request): predict the podium with NO data from the target weekend
Inputs allowed: results/qualifying/laps of all EARLIER races, the entry list (driver->team) and the circuit. Not allowed: grid, qualifying,
sprint, practice of the target weekend. Metric unchanged (podium overlap, E0). Target: >= 75%. Data source switched to FastF1 (see f1/ff1.py).
Model-free pre-weekend baselines, measured before any model:

| podium overlap | dev 2023-25 | 2026 |
|---|---|---|
| previous race's podium | 0.500 | 0.509 |
| championship standings top-3 | 0.590 | 0.526 |
| decayed driver form (hl=3) | 0.557 | 0.491 |
| decayed team form + driver | 0.552 | 0.509 |

Argument for feasibility: pre-weekend info is (roughly) a subset of what the post-qualifying model used (E6: 0.65-0.69). A model with
less information beating a better-informed one by 6-10 pts would be surprising => prior says 75% is unlikely. To be tested, not assumed.
The user's hypothesis (accuracy improves across a season as in-season data accumulates) is tested explicitly via accuracy-by-round curves.
Telogify's FastF1 cache covers only 14 weekends of 2026 (3.6 GB, mostly telemetry) -> reused for those; everything else downloaded.

## E8 — Pre-weekend (no target-weekend data) models, FastF1 race-pace features (dev 2022-25, then ONE look at 2026)
Data: FastF1 is now used for race pace (clean green-flag lap median gap, 2018-2026; corr 0.993 with the OpenF1 version on 1,718 driver-races).
Official results stay on Jolpica: FastF1's grid/points/status come from Jolpica internally (NaN when rate-limited, observed in many races when
4 fetchers ran in parallel) and its timing-based positions ignore post-race penalties (e.g. 2019 Canada), so it is a worse label source.
Leak tests: tests/test_preweekend.py wipes ALL target-weekend data and asserts the pre-weekend features are bit-identical; mutation-checked.

Dev 2022-25 (92 races), podium: standings 0.583 | logit(pace-aware, 34 feats) 0.554 | lgbm ~0.53 | pace-only scores 0.51-0.54 | **rank-avg(standings, logit) 0.583**.
Frozen: `ens` = rank-average(standings, logit). Held-out 2026 (19 races), single look:

| 2026 pre-weekend | ens | standings | logit | lgbm | prev podium |
|---|---|---|---|---|---|
| podium | **0.544** [0.439, 0.632] | 0.526 | 0.561 | 0.579 | 0.509 |
| winner | 0.368 | 0.368 | 0.263 | 0.316 | 0.368 |
| pairwise | 0.771 | 0.755 | 0.770 | 0.756 | 0.683 |

Races >= 75% podium: 1/19 (2023-25: 0/22, 2/24, 3/24). **75% is NOT reached; best pooled pre-weekend podium accuracy ~0.57.**
ORACLE UPPER BOUND (diagnostic, uses post-race info): knowing each driver's TRUE race pace in the target race gives podium overlap
0.74 (2022-25: .742/.742/.736/.736; 2026: .789). So the target is at/above what perfect pace knowledge delivers; strategy, safety cars,
first-lap incidents and retirements cap it. A forecast made before the weekend cannot beat that bound.

Hypothesis "accuracy improves across the season": pooled slope -0.0018/round (p=0.64). By stage R1-5 .60 / R6-11 .58 / R12+ .56.
2023-25 show no gain (2025 flat at .667). 2026 (new regulations) does rise: R1-5 .40 -> R6-11 .56 -> R12+ .63 (n=19; noisy but
consistent with the idea that in-season data matters when the car pecking order is reshuffled).

## E9 — User-directed information expansion (pre-weekend only; all from PAST weekends/sessions)
Built (all leak-tested by tests/test_profile_leak.py, which replaces every target-weekend telemetry/session/strategy/circuit row with noise):
- f1/telemetry.py: fastest-quali-lap telemetry -> per-driver CORNER PROFILES. A reference lap defines corners (speed minima); every lap is resampled
  onto the same track-position grid. Per corner: apex speed, braking decel into the corner, distance to full throttle after apex, speed gained over the
  next 150 m, top speed on the following straight; gaps vs best car averaged by corner class (slow<130, med<220, fast km/h).
  ENGINE proxy: full-throttle acceleration at 200-280 km/h with DRS closed, plus straight-end speed gaps. Circuit layout traits from the reference lap.
- f1/sessions.py: all sessions (FP1-3, Q, SQ, S, R): best/long-run pace, sector gaps, speed-trap gaps, tyre-degradation slope.
- f1/strategy.py + f1/pits.py: per race stops/stint length/compound mix/wet share/safety-car share; pit-lane time per driver.
- f1/profile.py: CAR profile (team mean, decayed hl=6 races -> follows in-season development), DRIVER STYLE (driver minus teammate in the same car, portable
  across teams), circuit character from PAST visits (layout, typical stops, SC share, DNF/collision rate, podium "chaos"), car x track "fit" scalars,
  strategy-execution residual (finish rank vs race-pace rank), pit-lane time, power-unit-pooled reliability/power (supplier map hand-entered: ASSUMPTION).
First look (dev 2023-25): adding the 99-feature profile block did not improve anything: pace-order Spearman 0.70-0.72 for every variant, podium 0.50-0.55 vs
standings 0.59. Ridge got worse (0.682).

## E10 — Does the premise ("tracks favour different cars") exist in the data?  (diagnostics, 2019-26)
- A team's deviation from its own season-average race pace = 32% of pace variance, BUT the same team at the same circuit one year later: corr(dev) = +0.044
  (n=1234); consecutive races: -0.007. Cars change too much year to year for "team X is good at track Y" to carry over.
- Within a season, deviation correlation by circuit-trait similarity (z-scored layout/straight/vmax/low-speed share): most similar +0.055, similar +0.080,
  different -0.025, most different -0.089 (the common negative offset is a mean-zero artifact). A real but weak gradient (~0.12 vs ~-0.04 after offset).
- As a forecaster, track-similarity-weighted team form is WORSE than plain recency-weighted form: team pace-order Spearman 0.774 (recency) vs 0.766 (tau=2.5)
  vs 0.709 (tau=0.7). The signal is too weak to pay for the data it discards.
- Pace -> podium mapping (corrupt TRUE race pace with noise): rho 1.00 -> podium 0.746; 0.96 -> 0.66; 0.86 -> 0.60; 0.76 -> 0.50.
  => even a PERFECT race-pace forecast gives 0.746 (<75%). Our pre-weekend pace forecasts reach ~0.72 driver-level / 0.77 team-level.
  Reaching 75% would need pace forecast better than perfect, i.e. predicting retirements, safety cars and strategy luck as well.

## E11 — Information blocks x learners, dev 2023-25 (70 races), pre-weekend. podium / pace-order Spearman (vs TRUE race-pace order)
Base = 38 history features. Each block added alone, then all, with logistic-podium / ridge-pace / extra-trees-pace. Standings baseline: 0.590 / 0.691.

| block added to base | logit | ridge | extra-trees | pace rho (best of 3) |
|---|---|---|---|---|
| (none: base) | .557 | .548 | .557 | .719 |
| strategy / attrition / SC / wet | .557 | .548 | .576 | .719 |
| circuit layout (corner mix, straights, vmax) | .557 | .548 | .567 | .720 |
| car x track fit | .552 | .519 | .552 | .717 |
| driver style x track fit | .557 | .538 | .548 | .718 |
| telemetry car profile (corners + engine proxy) | .543 | .519 | .562 | .716 |
| driver style (teammate-relative) | .548 | .543 | .548 | .721 |
| pit / execution / deg / sectors | .543 | .538 | .576 | .717 |
| power-unit pooled reliability + power | .557 | .533 | .552 | .719 |
| ALL | .543 | .490 | .571 | .717 |

**No block beats the base; ALL is worse for ridge (overfitting 100+ noisy features on ~2,000 rows). Nothing beats championship standings on podium (.590).**
Why (E10): the car-x-track effect the features try to capture is a tiny share of variance and does not persist (r=+0.04 year to year at the same circuit),
so these features add noise faster than signal. Learner type (linear / trees / MLP-free ensembles) did not matter either.
No new 2026 look was taken: dev showed no gain, so the frozen E8 model (`ens`) stays the deliverable and 2026 stays at 0.544.

## Conclusion of the pre-weekend search (E2-E11)
Tried: history windows/decay, practice+race lap pace, label handling, reliability model, rank features, pairwise Bradley-Terry, GBM/lambdarank, ridge, extra-trees,
telemetry corner profiles, engine/power proxies, power-unit pooling, driver-style x track, strategy typicality, attrition/SC, pit execution, track-similarity forecasting.
Best pre-weekend podium accuracy stays ~0.57-0.58 (dev) / 0.544 (2026). Pace-forecast quality is the binding constraint (rho~0.72) AND the pace-based ceiling is
0.746 (E10), so 75% is not reachable from pace information alone, and the remaining outcome variance (incidents, SC, strategy luck, DNFs) has no pre-weekend predictor in the data.
