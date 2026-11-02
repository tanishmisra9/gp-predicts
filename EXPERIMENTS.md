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

# ROUND 2 — audit-driven exhaustive search (E12-E21). Dev window widened to 2020-2025 (131 races); inner = 2020-22, outer = 2023-25.

## E12 — Yardstick fixes (audit A)
- FOUND: pre-weekend evaluation broke ties using the TARGET-WEEKEND GRID (`metrics()` sorted by [score, gridn]). Measured impact: 22/89 races had a tie at the 3rd/4th cut but podium
  accuracy was identical (0.577 grid-tie vs 0.577 name-tie). Fixed anyway: `tie=None` (driver-name tie-break) for every pre-weekend run; results regenerated (2026 unchanged at 0.544).
- Added NDCG@3, paired bootstrap, label-shuffle NULL test and random baseline. NULL (shuffled labels) podium 0.181, random 0.165 -> pipeline is not leaking labels.
- Frozen ens vs standings on 131 races: podium 0.606 vs 0.598, paired diff +0.008 [-0.013, +0.028] (not significant).

## E13 — State-space (Kalman) team pace + hindsight headroom
KF (tuned q,qs,r) team race-pace order Spearman 0.760 vs recency 0.759 (quali gap 0.618 vs 0.583). ORACLE with hindsight (team's leave-one-race-out SEASON mean, uses future races): 0.775.
=> at most +1.5 pts of team-order correlation is available from ANY "persistent level" model. Better form-tracking is a dead end.

## E14 — Do circuit traits predict team pace deviations? (hindsight-generous, LOO within season incl. future races)
Ridge per team-season on 9 circuit traits, no intercept (an earlier version with an intercept gave spurious negative correlations: demeaned-LOO artifact, my test bug, fixed).
corr(pred, actual deviation): race pace +0.09..+0.12, quali +0.02..+0.06; out-of-sample R^2 vs zero -0.01..-0.11. Real but tiny with this trait vector. (Limit: crude 9-number trait vector.)

## E15 — Where do podium errors come from? (131 races, 393 podium slots, frozen ens)
Team-order Spearman 0.78; teammate head-to-head accuracy 0.64. 157 missed slots: predicted rank of the missed driver was 4-6 for 104 of them (near misses).
157 false picks: 54% finished P4-6, 23% P7+, 23% RETIRED. Retirements cost ~9.2 pts of podium accuracy.

## E16 — Weather (Open-Meteo): ORACLE test
Forecast archive (Previous Runs API, 3-day lead) exists only from 2024 (67 races); actual race-day rain/temp/wind pulled for all 192 races (f1/weather.py).
Giving the model the TRUE rain (+temp, wind, a past-only driver wet-skill rating and interactions): logit podium 0.595 -> 0.593-0.595; lgbm 0.570 -> 0.570-0.583. No gain even with perfect weather knowledge.
Wet (>1 mm, n=31) races did not show more upsets (4.5% vs 4.7% of podium slots from outside the top-3 grid). Forecast-based features therefore cannot help; dropped.

## E17 — INFORMATION-STAGE CURVE (logit podium, 131 races): how much is each kind of information worth?
history only (the user's no-weekend-data setting) 0.595 | + practice pace 0.585 | + grid & qualifying 0.687 | + this weekend's own quali corner/engine telemetry 0.679 | grid alone 0.679.
Even with EVERYTHING known before the start, podium accuracy is ~0.69. Only grid/qualifying moves it; practice and telemetry add nothing. => 75% exceeds the best achievable at ANY pre-race stage.

## E18 — Monte Carlo race simulation with retirement model (4,000 sims/race; pace noise sigma x spread of predicted pace; Bernoulli retirement)
Retirement model (logistic on team/driver/PU/circuit attrition): modelled DNF rate 14.7% vs actual 15.0%. Podium 0.585-0.590 (sigma 0.3-1.5) vs logit 0.595, standings 0.598. Brier of P(podium) best at sigma=0.6
(0.0773 inner / 0.0853 outer). Better probabilities, same top-3.

## E19 — Learners, tuning, stacking (outer window = honest)
12 logit (C x half-life) settings + lgbm podium/rank + ridge/ET/lgbm pace + ET order + standings: INNER 0.60-0.65, OUTER 0.52-0.59. Standings OUTER 0.590 beats every learner.
Nested tuning: best-on-inner logit (C=0.01,hl=100) inner 0.645 -> outer 0.548 (hand-set 0.557). Stacking: best subsets by inner reach outer 0.567-0.586; across all 98 subsets of up to 4 models
outer podium median 0.581, 95th pct 0.600, max 0.605; equal-weight average of all 0.586; standings 0.590. Learned models do not beat standings out of sample.
TabPFN: BLOCKED (weights need a Prior Labs account/licence acceptance; not done on the user's behalf). Packages uninstalled again.

## E20 — Telemetry reliability audit (why E9/E11 found nothing)
Over 19 corner/engine metrics (2022-26): median week-to-week persistence of a team's value 0.04; between-team share of variance ~10%; median Spearman with race pace 0.17.
Only apex_med/apex_fast (0.59/0.63, essentially lap speed) relate to pace; the engine proxy acc_gap is the most persistent (0.34) but unrelated to race pace (-0.01).
=> single-fastest-quali-lap corner metrics are NOT a stable car signature. A better build (many laps, race trim, position-aligned corners) is untested and would need far more telemetry.

## E21 — Remaining routes
Dedicated teammate head-to-head model: 62.8% accuracy (ens-implied 63.9%); blending lowers podium (outer 0.586 -> 0.581/0.571/0.543 for lambda .1/.2/.4). Longer history (train from 2016 / 2014 instead of 2019): outer ens 0.567/0.562 vs 0.571 - no gain.

## Metric variants for the frozen pre-weekend ensemble (131 races; DIFFERENT, looser questions - not a pass of the pre-registered metric)
strict podium overlap 0.606 | >=2 of 3 podium drivers 0.702 | all 3 any order 0.160 | podium covered by top-4/5/6 picks 0.697/0.791/0.865 | winner within top-1/2/3 picks 0.473/0.679/0.824 |
per-driver binary "on podium?" accuracy 0.882 vs always-say-no 0.850 (hobby-project "80-91% podium accuracy" claims are most likely this metric).

## ROUND 2 CONCLUSION
Plateau ~0.58-0.61 pre-weekend (131 races). The remaining headroom is closed from three independent sides: (1) perfect hindsight "persistent form" adds <=1.5 pts of team-order correlation (E13);
(2) full pre-race information (practice, qualifying, grid, telemetry) tops out at ~0.69 (E17); (3) even a perfect race-pace forecast gives 0.746 (E10). 75% strict podium overlap is above all three.

# ROUND 3 — 36 new experiments, reasoned BACKWARD from the result (the real podiums) to the pipeline (E22-E57)
Protocol: pre-weekend only; dev 2020-25 (131 races); INNER = 2020-22, OUTER = 2023-25 (the honest number); paired outer diff vs the frozen `ens`. Reference outer podium: standings 0.590, ens 0.586, logit 0.557.
**INCIDENT (caught by the audit habit):** E30/E31 first showed 0.71 outer podium. Cause: the models used the harness-supplied default feature list, which contained the TARGET WEEKEND's grid/quali gaps and the target
race's own pace labels (y_pace, y_q). Fixed at the root: in pre-weekend mode (`tie=None`) walk_forward now defaults to history-only columns and RAISES if any forbidden column is passed
(`W.assert_pre_weekend`, `tests/test_guard.py`). All earlier scripts were audited by grep: they bound feature lists explicitly, so no earlier result was affected. Corrected E30/E31: 0.548 / 0.552.

## A. Result-side (backward diagnostics - what did the true podium need that the pipeline did not know?)
| # | Experiment | Result | Verdict |
|---|---|---|---|
| E22 | Oracle ladder: give the model ONE true fact at a time | ens 0.606 -> knowing retirers 0.646 (+4.0); true teammate order 0.656 (+5.0); true team race PACE 0.669 (+6.3); **true TEAM finishing order 0.743 (+13.7)**; true driver pace 0.740 | Biggest unknown = team order (incl. strategy/reliability luck), then teammate order, then retirements |
| E23 | Hindsight fixed effects (whole-season mean incl. future, minus the race) | driver season-mean 0.552; team season-mean + ens within team 0.463 (both BELOW the dynamic ens 0.606) | Static strength is worse than recent form: within-season dynamics are real signal the model already uses |
| E24 | Single-feature within-race AUC for podium | best: d3_q_pos .905/.887 (inner/outer), d10_pts_n .904/.895, d10_order, d_season_pts .899/.892 | Many near-identical correlated features; no hidden single killer feature |
| E25 | Podium team-pattern | real: 2+1 (two cars of one team) 62%, 1+1+1 38%. Predicted: 2+1 in 85%. Pattern matched 61% | Model over-concentrates on the top team |
| E27 | Is race difficulty predictable from pre-race descriptors? | corr(podium overlap, x): stops_2plus -0.175, circ_grid_corr +0.135, c_chaos -0.078, season_races -0.087 | Weak: 2-stop tracks and early-season are a bit harder |
| E56 | How accurate must TEAM-ORDER prediction be? (noise-corrupt the true team order) | perfect 0.743; Spearman 0.97 -> 0.705; 0.89 -> 0.613; 0.79 -> 0.529. Mine: 0.748 | 75% needs a better-than-perfect team order PLUS better teammate/retirement calls |
| E57 | Selective prediction: abstain when unconfident (confidence = sum of top-3 probs) | accuracy at coverage 100/60/40/25/15%: .606/.616/.597/.566/.550; corr(conf, acc)=+0.10; nested outer confident .583 vs rest .586 | Does NOT work: the model does not know when it is right (first attempt had a NaN-threshold bug; fixed) |

## B. Structure-side (models built around what A showed)
| # | Experiment | outer / inner podium | dOuter vs ens | Verdict |
|---|---|---|---|---|
| E28 | Team-then-driver: predict TEAM best-finish order, within-team order from logit | .543 / .617 | -0.043 | worse (forces a 2+1 pattern) |
| E29 | Diversity penalty on the weaker teammate (delta .05/.1/.2) | .576/.548/.519 | -0.010/-0.038/-0.067 | fix for over-concentration backfires; monotone worse with delta |
| E30 | Plackett-Luce top-3 likelihood, linear utility (CORRECTED) | .548 / .634 | -0.038 | no gain |
| E31 | All-pairs Bradley-Terry, top-6 focus (CORRECTED) | .552 / .617 | -0.033 | no gain |
| E32 | Within-race rank-transformed features (logit / lgbm) | .557 / .548 | -0.029 / -0.038 | no gain |
| E33 | Elo, team+driver additive, pairwise updates, K and season-shrink tuned on INNER only (Kt40 Kd10 shrink.8) | .548 / .634 | -0.038 | no gain; ndcg3 .626 |

## C. Label-side (change what the model is trained to predict)
| # | Experiment | outer / inner | dOuter | Verdict |
|---|---|---|---|---|
| E34 | Label = "would-be podium by TRUE race pace" (immune to DNF/strategy), evaluated on real podium | .567 / .634 | -0.019 | no gain |
| E35 | Ridge on log(finishing position) | .581 / .623 | -0.005 | ties ens; best learned model of the round |
| E36 | Ridge on log(1+points) | .571 / .628 | -0.014 | no gain |
| E37 | Down-weight retired rows x0.3 | .548 / .634 | -0.038 | no gain |
| E38 | Soft labels (P4-5 = 0.4) | .543 / .645 | -0.043 | no gain, highest inner |

## D. Feature-side (new past-only information)
| # | Experiment | outer / inner | dOuter | Verdict |
|---|---|---|---|---|
| E39 | Trend: recent-vs-long form for team and driver (pace and results) | .557 / .634 | -0.029 | identical to base: nothing added |
| E40 | Cleaner quali history from Q1 gaps (all drivers set a Q1 time) | .557 / .639 | -0.029 | nothing |
| E41 | Start skill: decayed lap-1 places gained | .557 / .639 | -0.029 | nothing |
| E42 | Dominance: decayed laps-led share (driver/team) | .557 / .634 | -0.029 | nothing |
| E43 | Past sprint-race form | .557 / .639 | -0.029 | nothing |
| E44 | Decayed-points standings family, half-life 2/4/8 races | .552/.552/.567 vs standings .590 | -0.033/-0.033/-0.019 | plain season standings is better than every decayed version |
| E45 | Circuit continuity: this driver/team's podium at this track last visit | .552 / .634 | -0.033 | nothing |
| E46 | ALL new features together | .552 / .617 | -0.033 | more features, lower both splits |

## E. Algorithm sweep (BASE features unless noted)
| # | Experiment | outer / inner | dOuter |
|---|---|---|---|
| E47 | SVM (RBF, 10 key features) | .571 / .623 | -0.014 |
| E48 | kNN k=60 | .557 / .590 | -0.029 |
| E49 | Gaussian naive Bayes | .571 / .628 | -0.014 |
| E50 | AdaBoost | .557 / .639 | -0.029 |
| E51 | HistGradientBoosting | .543 / .579 | -0.043 |
| E52 | Random forest | .562 / .623 | -0.024 |
| E53 | Bagged logistic (25 bags) | .571 / .634 | -0.014 |
| E54 | Monotone-constrained LightGBM | .562 / .601 | -0.024 |
| E55 | Top-8 features chosen from training data only, logit | .576 / .623 | -0.010 |

## WHAT WORKED / WHAT DIDN'T
- Worked (relative): championship standings (.590) and the frozen standings+logit ens (.586), ridge on log-position (.581), top-8-feature logit (.576). None statistically separable from each other (paired CIs span 0).
- Did not work: every one of 30+ alternatives. Not one beat standings on the outer window. Flexible models fit 2020-22 (inner .58-.65) and drop on 2023-25 (outer .54-.58); standings barely moves (.607 -> .590).
- Falsified ideas: diversity correction, selective prediction, team-then-driver, pace-based labels, retired-row down-weighting, trend/start/laps-led/Q1/sprint features, Elo/PL/BT ranking likelihoods, SVM/kNN/NB/RF/GBM/monotone GBM.
## OVERALL OBSERVATIONS (backward chain: result -> information gap -> pipeline)
1. The unknowns that cost accuracy, in order: TEAM ORDER of the race (13.7 pts), teammate order (5.0), retirements (4.0). Pace per se is a smaller piece. Team order includes strategy, safety cars, first-lap and reliability luck, which no pre-weekend feature observed here predicts.
2. Pipelines can only re-estimate form; E13/E23/E33-E46 all hit the same wall: recent form is already extracted by standings/decayed means, and extra feature classes (start skill, laps led, trend, circuit continuity) carry no marginal information.
3. Model flexibility hurts: the inner-outer gap grows with flexibility. With ~130 races the dominant error is variance, so simpler is better; E35/E55 (smooth target / few features) are the least bad learned models.
4. The model's confidence is uninformative (E57), so there is no safe subset of races where 75% is reached.
5. Even a hypothetical perfect team order gives only 0.743 (E56): 75% strict podium is above the oracle for every single-source pre-race information I could construct.

# ROUND 4 — 35 more experiments (E58-E92), following the Opus audit routes, plus web research and new data sources
Protocol unchanged (pre-weekend only, harness guard on). Reference podium (outer 2023-25 / pooled 2019-25): standings .590 / .596, ens .586 / .607, logit .557 / .599.
Statistical context discovered this round: with 131 races the smallest detectable real gain is ~4.5 pts (70 races: ~6 pts) -> "no gain" means "no gain larger than ~5 pts".

## A. Yardstick and regime (audit route A)
| # | Experiment | Result | Verdict |
|---|---|---|---|
| E58 | Regime shift vs overfitting: per-season (learned - standings) | 2020 +.039, 2021 +.076, 2022 -.016, 2023 -.030, 2024 -.028, 2025 -.042; corr with training size -0.84 | Learned models WIN in dominant-team eras and LOSE in competitive eras: a regime effect, not classic overfitting |
| E59 | Equal-length rolling training windows (last 44/66/110 races) | outer .552/.543/.557 (all-history .557); standings .590 | Window length is not the cause (supports E58) |
| E60 | Power analysis | MDE (80% power): ~0.085 @70 races unpaired / ~0.061 paired; ~0.045 paired @131 | All "no gain" claims are bounded at ~±5 pts |
| E61 | Polymarket "Driver Podium Finish" markets: coverage + price quality | 20 completed races have markets (3 in 2025, 17 in 2026). Resolved markets return EMPTY price history for most races; pre-weekend price exists for only 3 races and long-shot quotes are untraded placeholders (sum 6-8.6 instead of 3) | BLOCKED: no usable historical benchmark/feature |
| E62 | Polymarket LIVE quotes for Brazil/Las Vegas 2026 | Brazil: volume 0, bid .51 / ask .99; mids Antonelli .75, Verstappen .75, Leclerc .645, Hamilton .645; Las Vegas: no quotes | Thin, but the quoted four equal my model's top four |
| E63 | Pooled evaluation 2019-25 (152 races, train from 2018) | ens .607, logit .599, standings .596, rank-prior .566 | Same ordering with 21 more races |
| E64 | Brier / log-loss of P(podium) (selection metric) | logit Brier .0758 (19-22) / .0832 (23-25) vs standings-rank prior .0831 / .0900; log-loss .244/.264 vs .298/.296 | **Learned model carries real probabilistic information the top-3 overlap hides** |
| E65 | Reversed windows: tune on 2023-25, test on 2019-22 | best-outer setting inner .646 vs standings .602 | learned beats standings in the early window, loses in the late one (mirror of E58) |
| E66 | Metric unit tests (ties, tiny fields, perfect/reversed) | pass (`tests/test_metrics.py`) | Harness verified |
| E67 | Stale-cache guard on the feature table | table() now rebuilds when any data/code file is newer | engineering fix |
| E68 | Season-block bootstrap of (ens - standings) podium | +0.011, 95% CI [-0.009, +0.032] | Statistical tie |

## B. New information blocks (logit and extra-trees variants; dOuter vs ens)
| # | Block | outer logit / ET | Verdict |
|---|---|---|---|
| E69 | Championship context (gap to leader, margin, rounds left, title pressure) | .557 / .562 | nothing |
| E70 | Rookie flag, experience, team-prior imputation | .557 / .552 | nothing |
| E71 | Calendar (days since last race, back-to-back) | .557 / .567 | nothing |
| E72 | Within-team strength share | .557 / .562 | nothing |
| E73 | Season-stage blend of short/long-run form (+ all blocks together .562) | .557 / .538 | nothing |

## C. Models
| # | Experiment | outer / inner podium | Verdict |
|---|---|---|---|
| E74 | Gaussian process, team race pace, time kernel vs time x circuit-trait kernel (2022-25, team-level Spearman) | recency .747; GP time-only .739; **GP time x traits .732** | adding the track kernel makes the forecast WORSE |
| E75 | Monte Carlo with team-shared shock (strategy/reliability luck), 3 settings | .548 / .634 | no gain over independent noise |
| E76 | Online Hedge ensemble (regime adaptive; weights from past races only) | .552-.581 / .623-.628 | no better than equal weights (.586) |
| E77 | Dominance-conditioned model choice (threshold from inner) | inner .661 -> outer .557 | overfits the threshold |
| E78 | Multinomial position bins -> P(P1)+P(P2-3) | .548 / .639 | no gain |
| E79 | Circuit-cluster-conditioned team form (k-means on layout traits) | .562 / .645 | no gain |
| E80 | Two-layer Kalman (team + driver offset) ranked by forecast pace | .576 / .612; ndcg3 outer .639 vs standings .643 | ties standings; best pure-pace model |
| E81 | Race-trim strength (race pace minus quali gap, decayed) | .562 / .623 | no gain |
| E82 | Sprint races as extra labelled training races | .562 / .639 | no gain |

## D. Telemetry done better, and pre-season testing
| # | Experiment | Result | Verdict |
|---|---|---|---|
| E83 | Telemetry v2 (mean of each driver's 3 fastest laps) reliability audit | median persistence .04 -> .07, validity .19 -> .29, between-team share .08 -> .08 | better, still mostly noise |
| E84 | v2 full profile as features (2022-25) | ET .562 vs base .543; logit .536 vs .554; ridge .514 vs .554 | no robust gain |
| E85 | v2 persistent-only metrics (persistence>.15: apex_slow, apex_med, gain_slow, gain_med, acc_gap) | ET .540, logit .558, ridge .547 (base .543/.554/.554) | no gain |
| E86 | Pre-season testing pace vs actual rounds 1-3 team race pace (2022-26) | team-order Spearman: testing best-lap .716, long-run .562, last-year pace .752, **testing + last-year .813**; in 2026 (new regs) testing .855/.882 vs last-year .750 | **A real signal, only for the first rounds** |
| E87 | Blend testing strength into ens for rounds 1-3 | podium .648 -> .685 at weight .25 (n=18 races, 12 with testing data) | suggestive, underpowered; affects ~12% of races |

## E. Backward / analysis
| # | Experiment | Result |
|---|---|---|
| E88 | Target family with 95% CIs (131 races): strict overlap ens .606 [.565,.646]; >=2 of 3 .702 [.618,.786]; top-4 cover .697; top-5 cover .791 [.751,.830]; winner in top-2 .679; per-driver binary .882 (always-no .850) | ens and standings indistinguishable on every variant; no variant of "podium accuracy" reaches 75% except top-5 coverage |
| E89 | Can pre-race descriptors predict which races go well? (leave-one-season-out AUC) | AUC .531 -> not predictable |
| E90 | Leave-one-block-out ablation of the logit (outer): -prev_order +.024, circuit stats 0, era/field 0, t10 0, t3 -.010, d10 -.014, d3 -.014, **season standings -.019** | Season standings is the most informative block; last-race result is slightly harmful (but inner -.005: not robust) |
| E91 | Sign test ens vs standings per race | ens better in 10, standings in 7, tied in 114 of 131 (p=.63) | the two pick the SAME podium in 87% of races |
| E92 | Stickiness: consecutive-race podium Jaccard | predicted .88 vs actual .39; corr of changes +.116 | reality is volatile; a persistent-state model cannot track it |

## WHAT WORKED
- Statistically supported: (1) Brier/log-loss shows the learned logit beats a standings-rank prior (E64): real but small probabilistic information; (2) pre-season testing pace improves the early-round team order (.752 -> .813; clearly better in the 2026 reset) (E86), with a suggestive podium lift (E87, n=18).
- Everything else tied standings/ens within noise: the two-layer Kalman filter (E80) and top-8/log-position models (E55/E35) were the closest.
## WHAT DID NOT WORK
- All new information blocks (E69-E73, E79, E81, E82), track-aware GP (E74, worse), telemetry v2 (E84/E85), team-shock Monte Carlo (E75), regime-adaptive ensembles (E76/E77), multinomial bins (E78), market data (blocked: illiquid, no history).
## OVERALL OBSERVATIONS (round 4)
1. The learned-vs-standings gap flips with the competitive regime (E58/E65): learned models gain 4-8 pts in dominant eras (2020-21) and lose 3-4 in competitive ones (2022-25). A fixed pipeline cannot be uniformly better than standings; regime detection (E77) overfit.
2. ens and standings choose the same podium in 87% of races (E91). The remaining differences are small and are not statistically resolvable at ~131 races (E60, E68).
3. Probabilistic metrics see more than overlap (E64). If a score must be optimized, Brier/log-loss is the sharper selection tool than podium overlap.
4. Track/car-specific structure keeps failing in the cleanest tests (GP E74, hindsight E14, similarity E10): the physical premise is plausible, but its signal is small relative to race-to-race noise.
5. The one genuine information gain found is temporal: before the season starts, testing pace carries information (E86), most valuable after regulation changes. Its ceiling is limited to ~3 of ~23 races per season.
6. Volatility (E92): real podiums share only 39% of members from one race to the next; 75% would require predicting exactly the part of that churn that is luck.

## ROUND 5 (user: "so much of what happened in that race is up to data" - classify the errors, then attack the biggest class)
| # | Experiment | Result | Verdict |
|---|---|---|---|
| E93 | Cause of every podium error, 2023-26 (89 races, 226 errors), using race data (true race-pace rank, retirements, lap-1 gain, stops) | **61% pace-order errors**: missed a driver who HAD podium pace 71 (31%), picked a driver WITHOUT podium pace 67 (30%). **39% execution/luck**: podium without podium pace (strategy/SC/start/attrition ahead) 42 (19%), picked driver lost his pace in the race 25 (11%), picked driver retired 21 (9%). Podiums reached without podium pace: median grid 4 -> finish 3, 81% started top-6, median 0 lap-1 gain, 24% gained >=3 places | The data-addressable error class is race-pace ORDER (needs a better pace forecast); execution is mostly luck near the front |
| E94 | Race-pace MEASUREMENT variants (median, p25, fuel-corrected, 1st-stint, best 8-lap, tyre-wear-corrected), driver-level, forecasting next race | fuel-corrected is most persistent (.750 vs .710 same-measure) but improves forecast of FINISHING order only .726 -> .733; p25 and best-stint are worse (.700/.691) | tiny gain; not a lever on its own |
Note: Mexico 2026 race laps (R19) were not yet loadable in FastF1, so its pace is missing from the pace history until it appears.
