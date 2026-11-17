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
| E69 | Championship context (gap to leader, margin, rounds left, title pressure) | .557 / .562 | historical timing-invalid variant: final-calendar dependency found in E187; do not treat as admissible pre-weekend evidence |
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

## ROUND 6 — the three untried data avenues, pre-weekend only (user: "try 1-3, prioritize no-info pre-weekend")
| # | Experiment | Result | Verdict |
|---|---|---|---|
| E95 | Power-unit supplier x ALTITUDE (Open-Meteo elevation; >600 m = Mexico 2234 m, Interlagos 779, Madring 683, Red Bull Ring 681, Las Vegas 649): supplier's pace deviation at high vs low tracks, cross-year persistence | corr **-0.58** (n=27 supplier-seasons) | no positive persistence: engines do not carry an altitude edge year to year |
| E96 | Same at TEAM level | corr **-0.28** (n=73) | none |
| E97 | 2026-style energy-deployment signature from the fastest quali lap (full-throttle acceleration 200-260 and 280-330 km/h, taper, clipping share), persistence / between-team share / validity vs race pace | persistence: 2022-25 cars .11-.20, **2026 cars acc_mid .42**, taper .20; between-team share 2026 acc_mid .17; corr with race pace -0.15..+0.06 in both regimes | a stable car/PU characteristic that does NOT tell you who is fast |
| E98 | Static circuit descriptors as features: altitude, high-altitude flag, historical race-day temperature/rain, historical tyre hardness, + interactions with team/driver form | podium outer: logit .557 (base .557), lgbm .548 (.533), ET-pace .562 (.557) | historical global-median interaction imputation timing issue noted in E188; not certified pre-weekend evidence |
| E99 | Reliability-weighted FUSION of up to 6 weak team-strength indicators (race pace, fuel-corrected pace, Q1 gap, FP long-run, points, finishing position) in a multi-observation Kalman filter; forecast of TEAM finishing order (Spearman vs actual best-car order) | single indicator inner .7546 / outer .7274; fused (noise tuned on inner) inner .7773 / outer .7311; **fused equal weights outer .7403** | historical timing-invalid scaling; E188 found full-sample standardization, so gain is unverified |
| E100 | Fused strength -> PODIUM: alone / as logit feature / as third member of the ensemble | outer .529 / .557 / .548 (ens .586, standings .590) | historical fusion path timing-invalid per E188; not certified pre-weekend evidence |
Historical observation (superseded in part by E188): E13 measured a particular hindsight persistent-form benchmark around .775, not a proven bound. E188 found that the cited .740 fusion result used full-sample indicator scales, so it cannot establish a causal pre-weekend gain or saturation claim.

## ROUND 7 — permitted-source follow-up after E100

### E101 — Cause-specific mechanical-failure exposure (A4 in RESEARCH_AVENUES.md)

**Hypothesis:** driver's and team's prior named mechanical failures per completed racing lap, plus whether the team had a failure in the last three races, might predict another mechanical retirement and improve the pre-weekend podium ranking. This differs from E18's all-cause DNF model. `experiments/a4_mechanical.py` builds three history features from strictly earlier result rows (10-race half-life, fixed prior), fits a regularized cause-specific logistic hazard on earlier races, and multiplies the frozen pre-weekend logit's podium probabilities by predicted survival. A generic-DNF-feature mechanical hazard is the comparator. No target-weekend status/laps are features; `tests/test_mechanical_leak.py` scrambles target-and-later rows and mutation-checks a prior row.

**Data-quality discovery:** Jolpica `status` in this dataset names 85 mechanical failures over 2020–22, but **zero** over 2023–25. In the latter three seasons, 153 rows say only `Retired` and cannot be assigned a cause; 174 rows in total have unknown or otherwise excluded outcomes. Those rows were excluded from hazard training and scoring rather than treated as mechanical negatives. This label regime change makes outer-window mechanical-hazard accuracy unmeasurable from this source. The apparently low outer Brier score on named mechanical events would be meaningless, so it is not reported.

| walk-forward metric | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| standings strict podium overlap | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| frozen logit | 0.6393 | 0.5571 |
| logit × generic-hazard survival | 0.6393 | 0.5571 |
| logit × cause-specific-hazard survival | 0.6393 | 0.5571 |
| label-shuffled logit null | 0.1749 | 0.1857 |
| cause-specific vs generic hazard Brier, named outcomes only | 0.06588 vs 0.06620 (85 events / 1,211 eligible driver-races) | **not estimable:** zero named events |

Both survival variants made the same podium selections as the original logit. Outer difference from standings: -0.0333, paired race bootstrap 95% CI [-0.0714, +0.0048]. **Verdict: no podium gain; stop this cause-specific route with the current labels.** The small inner Brier difference does not validate mechanical-risk forecasting in the target era, and the unclassified `Retired` rows could contain many mechanical failures. This is a source limitation, not evidence that failure risk cannot be forecast in principle. All existing leak/guard/metric tests and the new source-specific leak test passed. No new 2026 evaluation was run.

### E102 — Clean-air, same-lap race-pace measurement (A2 pilot)

**Hypothesis:** past race laps observed in cleaner air and compared with other drivers on the same lap might produce a more useful next-race pace signal than a driver's median lap. `experiments/a2_matchedpace.py` extracted prior-race timing from the existing FastF1 offline cache, requiring green/accurate laps, no pit in/out, at least five usable drivers per lap, and an estimated finish-line gap over 1.5 seconds to the car ahead (or the leader). It used each driver's median lap-time residual against the same-lap field median. Both this measure and the existing clean-lap median were forecast with the **same** 10-race half-life from strictly earlier races. No race car/position telemetry was present in the cache; this is a timing-only first test of A2, without compound/tyre-age matching or position-aligned corners. `tests/test_matchedpace_leak.py` scrambles target-and-later source rows on synthetic and actual cached data.

Offline extraction produced 3,311 driver-race rows from 172 races in 2018–25 (one 2018 race unavailable); 3,112 rows had matched-lap estimates. Target-race matched-forecast coverage was 99.3% on inner and 99.2% on outer scored driver-races. Scoring used all 61 inner and 70 outer races with the same deterministic missing-value fallback and no target-weekend tie-break.

| walk-forward metric | inner 2020–22 | outer 2023–25 |
|---|---:|---:|
| standings strict podium overlap | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| past median-pace order | 0.6175 | 0.5429 |
| past matched-lap-pace order | 0.6557 | 0.5619 |
| label-shuffled logit null | 0.1749 | 0.1857 |
| predicted vs true next-race pace Spearman, median / matched | 0.7083 / 0.7063 | 0.7004 / 0.6959 |

Matched-lap order beat median-pace order by +0.0383 inner (paired 95% CI [+0.0109,+0.0710]) and +0.0190 outer ([0,+0.0429]), while its outer difference from **standings** was -0.0286 ([-0.0667,+0.0095]). Its pace correlation was slightly worse in both splits. **Verdict:** the timing-only matched-lap filter does not beat standings or improve the general pace forecast; do not promote it to the frozen model. Its apparent improvement over the weak pure-pace ranking is small and may reflect top-contender selection rather than better pace measurement. The estimated clean-air gap uses lap-end timing/position and can misclassify lapped or traffic-affected cars. Raw race telemetry was unavailable in the existing cache, and this process had no F1 credentials, so the many-lap X/Y pilot A1 remains untested. All existing guards and the source-specific leak test passed. No 2026 evaluation was run.

### E103 — Does matched pace change the third-place boundary? (A8 follow-up)

Pre-specified one simple blend after E102: equal rank average of standings, the frozen logit, and E102's matched-pace order. No weight search or new data extraction. The same source-specific leak test and pre-weekend guard apply. The label-shuffled logit null remained 0.1749 inner / 0.1857 outer.

| strict overlap | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| `ens` plus matched pace, equal three-way rank average | 0.6503 | 0.5857 |
| podium selections changed versus `ens` | 8/61 races | 15/70 races |

The blend gained +0.0219 over `ens` in the inner window (paired 95% CI [0,+0.0492]), but **zero** in the outer window ([-0.0238,+0.0238]). Outer difference versus standings was -0.0048 ([-0.0335,+0.0238]). **Verdict: no held-out-era gain.** The new measure sometimes reaches the podium boundary, but its outer changes cancel. This is another example of better-looking inner results failing the competitive 2023–25 regime; it does not move the 75% gate. No 2026 evaluation was run.

### E104 — Matched-compound, teammate-relative tyre degradation (A3 measurement pilot)

The strict podium reference remains standings 0.5905 and frozen `ens` 0.5857 on outer 2023–25; **this pilot did not produce a new podium ranking**, so there is no candidate strict-overlap score to compare. The first gate was whether a driver-specific tyre-wear measurement from earlier races predicts later behaviour well enough to justify such a model.

`experiments/a3_tyre_degradation.py` read cached green, accurate, non-pit race laps, removed large lap-time outliers, measured each stint's lap-time residual slope versus tyre age (at least eight laps and six ages), and compared teammate slopes only where both ran a common compound. It excluded implausible absolute slopes over 1 second/lap, then carried the driver's last observed teammate-relative slope into the next race. Source coverage: 2,978 driver-race style rows from 171 races in 2018–25; one 2018 race was unavailable and one 2021 race had no comparable stints. `tests/test_tyre_style_leak.py` verifies target-and-later source edits leave target forecasts unchanged on synthetic and cached data.

| diagnostic | inner 2020–22 | outer 2023–25 |
|---|---:|---:|
| usable previous/current driver-race style pairs | 1,030 | 1,217 |
| previous-vs-current style correlation | 0.0709 | 0.0659 |
| within-race style Spearman | 0.0767 | 0.0579 |
| teammate pairs with two style histories | 512 | 604 |
| lower predicted degradation finished ahead | 0.4785 | 0.4719 |
| standings picked the teammate who finished ahead, on those same pairs | 0.6094 | 0.6258 |
| within-pair shuffled-label null | 0.5000 | 0.4996 |

**Verdict: stop this A3 measurement route before fitting a podium model.** Its prior value barely repeats, and the implied teammate call is slightly below chance and far below standings. Lap-end residuals, traffic, pit timing, non-identical car setup, and tyre-use choices still confound the slope; this is not proof that intrinsic tyre-management skill does not exist. The initial extractor run failed because its FastF1 cache path was not initialized, and a first report failed on pandas' `DataFrame.style` property; both were fixed before these numbers. The source-specific leak test and existing guards passed. No 2026 evaluation was run.

### E105 — Comparable medium-tyre pre-season testing laps (A6 follow-up)

**Strict podium overlap first (2023–25 rounds 1–3; nine races):** standings **0.7037**, frozen `ens` 0.6667, `ens` plus medium-window testing 0.5185, `ens` plus the existing testing best-lap measure 0.6667. The equal three-way rank average was fixed in advance; no blend weight was tuned. The medium blend versus standings was -0.1852, paired race bootstrap 95% CI [-0.2963,-0.0741]; versus `ens` it was -0.1481 [-0.2593,-0.0370]. The label-shuffled logit null scored 0.2593 on this tiny nine-race set. Per-season medium-blend overlap was 0.6667 / 0.4444 / 0.4444 for 2023 / 2024 / 2025.

`experiments/a6_testing_medium.py` read the existing offline FastF1 testing cache for three days in each of 2023–25. It used accurate, non-pit, medium-tyre laps at tyre ages 4–12, requiring at least six laps per driver-day, and averaged day-relative pace into the entry-list team for rounds 1–3. Coverage: 155 driver-day rows, 20 distinct drivers per year, nine of nine days. The measure's team-order Spearman versus actual early-round race pace was 0.3025, compared with 0.6629 for the existing testing best-lap measure on the same 90 team-races. The medium-only podium order scored 0.3333; best-lap-only 0.4815. `tests/test_testing_medium_leak.py` scrambles target-weekend labels and future testing rows while holding target features fixed.

**Verdict: negative.** Age/compound restriction did not make testing long-run pace useful; unknown fuel, programmes, and track evolution remain plausible noise sources. This evaluates just nine repeatedly inspected development races, and 2022 raw testing laps were not available in the cache for an independent tuning window. E86/E87's broader early-round suggestion should not be overturned by a tiny subset, but this specific medium-window measurement is worse. No 2026 evaluation was run.

### E106 — Race-trim X/Y speed-profile feasibility pilot (A1)

**Strict podium overlap:** no candidate ranking was generated. The existing 2023–25 outer reference remains standings **0.5905** and frozen `ens` 0.5857; no claim of improvement toward 0.750 follows from this measurement pilot. An inner 2020–22 race-telemetry test is unavailable through OpenF1, whose historical feed begins in 2023. No 2026 data was read. A paired podium bootstrap and label-shuffled podium null are inapplicable without candidate predictions; the profile-identity permutation below is the diagnostic null.

`experiments/a1_race_xy_pilot.py` used the project's existing OpenF1 feed to fetch five fixed early-stint race laps (laps 6–10) for Verstappen, Pérez, Leclerc, and Sainz at Bahrain in 2023 and 2024. It timestamp-matched car speed to X/Y location within 400 ms, projected samples onto a same-race reference lap, and took median speed in 50 distance bins. This yielded 1,527–1,623 matched samples per driver-race, with 49 or 50 bins covered in both lap halves. The 2024 reference lap is used only to measure the 2024 outcome here; it must never become a feature in a 2024 pre-weekend prediction. The pilot made no prediction and added no model feature, so there is no new target-source leak path to scramble. The existing leak, pre-weekend, profile-leak, guard, and metric tests passed.

Raw first-two-versus-last-three-lap profile correlations were 0.990–0.997, and 2023-versus-2024 correlations were 0.971–0.985. These mostly reflect the circuit's common speed curve. After subtracting the four-driver field median at each bin, within-race split-half correlations ranged from **-0.077 to 0.706** (mean approximately 0.374); same-driver cross-year correlations were **0.462, -0.097, 0.006, 0.323** (mean 0.174). Among all 24 permutations matching 2024 driver identities to 2023, four had a mean residual correlation at least as large as the true identity matching (exact permutation tail 4/24 = 0.167). The four-driver field reference is small and team-skewed, and seasonal car changes, fuel, tyre wear, and traffic were not controlled. This is a feasibility sample, not a reliable estimate of signal across the grid.

**Verdict: technically feasible, no predictive evidence yet.** Position alignment and many laps do not by themselves establish a persistent car-specific signature. Do not build a podium feature from these correlations. A larger, prespecified multi-circuit measurement and next-event residual test would be required before A1 merits a walk-forward podium backtest; the lack of 2020–22 race telemetry makes model selection on an untouched inner period especially difficult.

### E107 — Past on-track passing conversion (B2)

**Hypothesis:** a driver's past conversion of close, on-track passing opportunities may add execution information to standings and the frozen podium model. `experiments/b2_passing.py` extracted from existing offline FastF1 race timing. An opportunity is a follower within 1.5 seconds of the immediately preceding car at a lap line; both must have accurate, green-flag, non-pit laps at that line and the next. A pass is a reversal of their relative race positions at the next line. The source contains 42,700 opportunities and 3,809 inferred passes across 171 races in 2018–25 (one 2018 race unavailable). For each target race, a beta-smoothed driver rate `(past passes + 2)/(past opportunities + 20)` used only strictly earlier races. An initial extraction failed because FastF1 was using its empty default cache; importing the project's cache initializer fixed it before scoring. A later merge/`k` implementation error also failed loudly and was fixed before results were recorded.

**Strict podium overlap**, walk-forward, all races included with the prior fallback for drivers with no opportunities:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| past passing rate alone | 0.2186 | 0.3524 |
| `ens` plus 10% passing-rate rank, fixed 45/45/10 weights | 0.6339 | 0.5810 |
| label-shuffled logit null | 0.1749 | 0.1857 |

The passing blend changed the equivalent of one correct podium pick in each split: +0.0055 versus `ens` inner (paired race bootstrap 95% CI [0,+0.0164]) and -0.0048 outer ([-0.0238,+0.0143]). Its outer difference from standings was -0.0095 ([-0.0381,+0.0190]). The weight is one fixed exploratory choice, **not** a tuned optimum; both development eras have already been inspected repeatedly. The approximately ±5-point detection limit still applies to broad model comparisons.

At the opportunity level (60 inner races with opportunities, 70 outer), the driver-specific prior versus a strictly past global pass-rate prior had race-averaged Brier **0.087982 vs 0.087756** inner (driver minus global +0.000226, 95% CI [-0.000213,+0.000758]) and **0.082730 vs 0.083138** outer (-0.000408, [-0.000726,-0.000090]). Log-loss was 0.319823 vs 0.319134 inner and 0.303608 vs 0.306063 outer. These tiny, era-dependent gains could reflect team pace and changing exposure, not an intrinsic driver passing skill. Spot-checked Bahrain 2023 swaps include plausible Pérez–Leclerc and Alonso–Hamilton moves, but timing alone cannot independently verify every event or correct for all lapped-car effects. `tests/test_passing_leak.py` scrambles target-and-later pass outcomes on synthetic and cached data; all existing pre-weekend leak, profile, guard, and metric checks passed. No 2026 evaluation was run.

**Verdict: no podium gain.** The event-level signal is interesting for probabilities but does not overcome standings at the podium boundary in this test. Stop this simple attacker-rate feature; a future B2 variant would need validated pair-level exposure and pace advantage before another podium backtest.

### E108 — Matched teammate race segments (A5)

**Hypothesis:** teammate-relative pace on the same clean race laps, compound, and similar tyre age can forecast the next teammate ordering more faithfully than broad driver form. `experiments/a5_teammate_segments.py` used the existing offline FastF1 race cache. It kept accurate, green, non-pit laps with a finish-line gap over 1.5 seconds to the car ahead, then paired teammates only on the same lap and compound within three tyre-life laps, requiring at least five paired laps. Each race yielded a symmetric teammate pace gap; its 10-race-half-life, lap-count-weighted forecast used only strictly earlier races with the **same team and teammate**. Coverage was 1,516 driver-race rows across 171 races in 2018–25 (one 2018 race unavailable, one 2021 race with no eligible matched pair). The result does not use target-weekend timing as a prediction feature.

**Strict podium overlap**, walk-forward, with an all-race score. The candidate assigned each team's two frozen-ensemble scores to the teammate predicted faster/slower by past matched pace; team score slots remained fixed. This is one deterministic intervention, with no threshold or weight search.

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| `ens` with matched-teammate score swaps | 0.6339 | 0.5857 |
| label-shuffled logit null | 0.1749 | 0.1857 |

The swap changed the top-three set in only **2/61 inner** and **4/70 outer** races. Versus frozen `ens`, strict-overlap difference was +0.0055 inner (paired race bootstrap 95% CI [0,+0.0164]) and **0.0000 outer** ([-0.0143,+0.0143]). Outer difference versus standings was -0.0048 ([-0.0333,+0.0238]). The sample still cannot reliably distinguish gains smaller than roughly five points.

The measurement itself transfers better than E104's degradation slope: past/current matched-gap correlation was 0.438 inner (516 driver rows) and 0.380 outer (548 rows). On the 567/655 covered teammate pairs, its faster-driver call was 0.621/0.666 correct, versus frozen `ens` 0.596/0.648 on the same pairs. For pairs with an actual podium finisher (an **evaluation-only** subset, not a feature), it was 0.735/0.768 versus `ens` 0.723/0.723 on 136/164 pairs. The race-bootstrap difference in pair accuracy was +0.019 outer overall (95% CI [-0.014,+0.052]) and +0.036 on outer podium-involved race averages ([-0.019,+0.093]); these do not establish a reliable teammate-order gain. A first bootstrap attempt produced impossible differences because boolean/object values were subtracted; the comparison was converted to numeric and rerun before logging these numbers. `tests/test_teammate_segments_leak.py` scrambles target-and-later source gaps on synthetic and cached data. Existing leak, pre-weekend, profile, guard, and metric tests passed. No new 2026 evaluation was run.

**Verdict:** promising measurement repeatability, **no outer podium gain**. The hard teammate swap rarely changes which driver occupies the third podium slot. Larger claims about driver skill or a path to 0.750 are unsupported; traffic estimates, tyre state, and sparse paired-lap selection remain limitations. Any next use of this feature should show how it changes marginal podium probabilities at the P3/P4 boundary, rather than optimizing teammate accuracy in isolation.

### E109 — Matched-teammate gap as one podium-model feature (A5/A8 follow-up)

**Strict podium overlap**, walk-forward over the complete development windows:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` (standings + fixed logit) | 0.6284 | 0.5857 |
| fixed logit + E108 prior teammate gap | 0.6448 | 0.5714 |
| standings + augmented logit rank average | 0.6339 | 0.5857 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/a5_podium_feature.py` merged E108's strictly past, same-team/same-teammate pace gap into the existing feature table. It added **one** feature to the frozen regularized podium logit, keeping `C=0.03`, 100-race half-life, the same training rows, and the existing equal-rank standings blend. Missing gaps were handled by the logit's existing median imputer; all races remained scored. The prior gap was present for 92.95% of inner and 93.71% of outer driver-races. The augmented blend changed the top-three set in 2/61 inner and 7/70 outer races. Its strict-overlap difference versus frozen `ens` was +0.0055 inner (paired race bootstrap 95% CI [0,+0.0164]) and **0.0000 outer** ([-0.0238,+0.0238]); outer difference versus standings was -0.0048 ([-0.0239,+0.0143]). No weight, imputation, or feature variant was selected on the outer result. These repeatedly inspected development races cannot establish a small gain.

The new gap modestly changed the logit's probabilities: driver-race Brier/log-loss were 0.07719/0.24968 to 0.07690/0.24942 inner, and 0.08314/0.26293 to 0.08261/0.26140 outer. The race-paired outer Brier difference was -0.000531 (95% CI [-0.001212,+0.000210]); log-loss difference -0.001532 ([-0.003140,+0.000192]). Both intervals include zero. `tests/test_teammate_segments_leak.py` now also scrambles the target-and-later raw gaps and target outcomes through the final feature merge; the feature remains unchanged. All existing pre-weekend leak, profile, guard, and metric checks passed. No new 2026 evaluation was run.

**Verdict: no strict podium gain.** A repeatable teammate-race measurement carries weak probabilistic information but does not displace standings at the third-place boundary. It does not advance the 0.750 gate. Do not promote this augmented model as the new frozen predictor.

### E110 — Forecast strategy exposure from team stop tendency × circuit pit cost (B3)

**Strict podium overlap**, all races in walk-forward order:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| logit + strategy exposure | 0.6393 | 0.5619 |
| standings + augmented logit rank average | 0.6284 | 0.5857 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/b3_strategy_exposure.py` used existing pit-stop records, not target-weekend inputs. On each completed earlier race it calculated the team's average stop count among drivers who completed at least 90% of race distance, relative to the field median, and the circuit's median pit-lane time. For each target race, `prior_team_stops` was a 10-race-half-life forecast from strictly earlier team results, and `prior_pit_cost` was the median of up to three earlier visits to the circuit (past global median fallback). Their product was the sole added feature to the frozen regularized logit; `C=0.03`, the 100-race training half-life, and the equal-rank standings ensemble stayed fixed. The original broad strategy/profile blocks in E9/E11 used average stop, pit, and circuit features; this specifically tests whether the **interaction of team stop tendency and circuit pit cost** is forecastable and useful. Missing feature values used the existing median imputer, with no race dropped.

The track cost itself was forecastable: prior-versus-actual pit-lane-time correlation was 0.824 inner and 0.879 outer. The team-specific stop tendency was not: correlation **0.088 inner, -0.015 outer**, despite 99.51%/99.86% target team coverage. The augmented ensemble's race-paired strict-overlap difference from frozen `ens` was **0 in both splits** (bootstrap interval [0,0]); outer difference from standings was -0.0048 (95% CI [-0.0381,+0.0238]). The standalone logit gained one equivalent correct podium pick in the outer split but remained below standings. A persistent circuit cost alone cannot separate teams if the strategy exposure does not persist. `tests/test_strategy_exposure_leak.py` scrambles target-and-later stop tendencies and pit costs on synthetic and cached data; the target features are unchanged. Existing pre-weekend leak, profile, guard, and metric tests passed. No new 2026 evaluation was run.

**Verdict: no podium gain; stop this historical stop-count interaction.** It does not support a path to the 0.750 strict target. It does not rule out a genuinely earlier-issued, race-specific strategy forecast, but no such target-weekend document or feed is used here.

### E111 — Joint traffic, compound, and tyre-age race-pace correction (A2)

**Strict podium overlap**, walk-forward across all development races:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| earlier clean-air matched pace (E102) | 0.6557 | 0.5619 |
| jointly corrected pace alone | 0.6557 | 0.5714 |
| standings + frozen logit + joint pace, equal rank average | 0.6557 | 0.5905 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/a2_jointpace.py` used cached historical FastF1 timing from 2018–25. Within each completed race, it removed pit, neutralized, inaccurate, and large outlier laps; expressed lap times relative to the same-lap field median; and fitted one ridge correction with driver indicators, compound indicators, compound-specific tyre-age slopes, and a close-traffic indicator based on lap-line gap. It removed the non-driver effects before pooling each driver's corrected residual pace. This differs from E94's separate scalar corrections and E102's clean-air selection. There was **one fixed** ridge penalty (50), traffic cutoff (1.5 seconds), and prior forecast half-life (10 races), with no outer-window parameter search. Historical source coverage was 3,291 driver-race rows from 171 races, of which 3,240 had valid corrected pace; one 2018 race was unavailable. For each target race, only strictly earlier source rows contributed to the driver forecast. `tests/test_jointpace_leak.py` scrambles target-and-later source pace on synthetic and cached data.

The joint forecast's next-race pace Spearman was 0.7139 inner / 0.7028 outer, versus E102 matched pace 0.7063 / 0.6959 on almost identical target coverage (~99%). On common races, the race-paired correlation difference was +0.0076 inner (bootstrap 95% CI [-0.0009,+0.0169]) and +0.0074 outer ([-0.0030,+0.0174]); neither interval excludes zero. The pure joint ranking gained +0.0095 strict overlap over matched pace outer ([-0.0143,+0.0333]) but trailed standings by -0.0190 ([-0.0524,+0.0143]). The three-way blend improved +0.0048 over frozen `ens` outer ([-0.0143,+0.0238]) and tied standings exactly in mean (paired CI [-0.0286,+0.0286]), changing the top-three set in 12/70 races. Its outer season scores were 0.5455, 0.5417, and 0.6806 in 2023, 2024, and 2025; the gains are not uniform. The approximately ±5-point podium detection limit remains relevant. Existing leak, pre-weekend, profile, guard, and metric tests and the new source-specific test passed. No new 2026 evaluation was run.

**Verdict: small, uncertain pace-order improvement; no established podium gain over standings.** This correction moved one equivalent correct pick over the frozen ensemble on outer races, far below the roughly 0.45 additional correct picks per race needed for 0.750. Lap-line gap can misclassify traffic, and compound choice is correlated with car and strategy, so the fitted nuisance effects are not causal estimates. Keep it as a measurement candidate, not as a new frozen predictor or evidence that the gate is reached.

### E112 — Joint pace as one regularized podium feature (A2/A8 follow-up)

**Strict podium overlap**, walk-forward over the same complete development windows:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| fixed logit + E111 past joint-pace forecast | 0.6393 | 0.5571 |
| standings + augmented logit rank average | 0.6284 | 0.5905 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/a2_jointpace_feature.py` added **one** E111 past-only joint-pace forecast to the frozen logit's feature set, keeping `C=0.03`, the 100-race training half-life, the same rows, and the equal-rank standings blend. The source value was available for 99.26%/99.28% of inner/outer driver-races; existing median imputation handled missing values. This is distinct from E111's separate three-way ranked vote. It changed the ensemble's top-three set in 0/61 inner and 2/70 outer races. The outer ensemble gained one equivalent correct podium pick over frozen `ens`: +0.0048 (paired race bootstrap 95% CI [0,+0.0143]), but merely **tied standings** in mean (paired CI [-0.0286,+0.0286]). The standalone logit's strict overlap did not change in either split.

The added feature did not improve outer probability quality: Brier 0.08314 to 0.08315 (race-paired difference +0.000013, 95% CI [-0.000069,+0.000097]) and log-loss 0.26293 to 0.26306 (+0.000126, [-0.000051,+0.000313]). Inner Brier/log-loss moved slightly downward, with intervals including zero. `tests/test_jointpace_leak.py` checks target-and-later source scrambles through the exact `prior_joint` feature builder; the pre-weekend guard rejects forbidden feature columns. All required leak, pre-weekend, profile, guard, and metric tests passed. No new 2026 evaluation was run. This is another exploratory look on repeatedly inspected development races, not an independent confirmation of the one-pick difference.

**Verdict: no gain over standings and no credible progress toward 0.750.** The modest pace measurement improvement from E111 does not become a material podium-model improvement when used as a regularized feature. Keep the frozen model unchanged.

### E113 — Multi-circuit race-trim X/Y speed-profile repeatability (A1 follow-up)

**Strict podium overlap:** no new podium prediction was made. The pre-weekend reference remains plain standings **0.6066 inner (2020–22, 61 races) / 0.5905 outer (2023–25, 70)**, frozen `ens` 0.6284 / 0.5857, and the label-shuffled logit null 0.1749 / 0.1857. The diagnostic source starts in 2023, so an untouched 2020–22 inner model-selection score is unavailable. No 2026 evaluation was run.

`experiments/a1_multitrack_xy.py` extended E106 to Bahrain, Austria, and Monza in 2023 and 2024, with six drivers (Verstappen, Pérez, Leclerc, Sainz, Hamilton, Russell). It used the first five laps of each driver's lap-6-to-10 race stint, aligned OpenF1 car speed to X/Y position, projected onto a same-race reference lap, and took median speed in 50 distance bins. It subtracted the six-driver, same-race, same-half field median at each bin before comparing each driver's 2023 and 2024 signatures. The extracted 3,597 bin records cover all 36 driver-races; every driver-half has 49–50 bins. These target-race measurements were used **only as retrospective labels for the repeatability diagnostic**, never as pre-weekend features.

| circuit | within-race split-half residual correlation, mean | same-driver 2023–24 residual correlation, mean | identity-matching rank among 720 driver permutations | rank among 8 teammate-only swaps |
|---|---:|---:|---:|---:|
| Bahrain | 0.384 | 0.148 | 106/720 | 1/8 |
| Austria | 0.327 | 0.076 | 258/720 | 3/8 |
| Monza | 0.225 | 0.200 | 54/720 | 3/8 |

The combined same-driver residual correlation is **0.141** (median 0.174; 13/18 positive); the identity matching ranks **45/720** when the same permutation is applied across all three circuits, but only **3/8** against teammate-only swaps. Raw speed-curve split-half correlations averaged 0.990–0.995 by circuit, mostly reflecting shared track geometry. The six drivers are only three front-running teams, and 2023-to-2024 car changes, tyres, fuel, and traffic remain uncontrolled. The 18 correlations are not independent draws, and the permutation ranks are exploratory diagnostics rather than a calibrated p-value for podium improvement. An OpenF1 timestamp-format mismatch initially stopped Austria 2024 extraction; accepting both ISO timestamp forms in the E106 helper fixed it before any result was calculated. A separate calculation from the saved CSV reproduced the combined mean and both permutation ranks. Existing pre-weekend leak, profile, guard, and metric tests passed; no new predictive source feature or source-specific leak path was introduced.

**Verdict: no basis for a podium-model feature or progress toward 0.750.** Multi-lap X/Y alignment is feasible, but the persistent driver-specific residual is weak and not convincingly distinguishable from teammate swaps. Stop this early-stint speed-profile variant; a future race-trim profile would need a stronger pre-registered transfer test across seasons and teams before a walk-forward podium backtest.

### E114 — Past-only forecast of the target grid (B8 feasibility gate)

**Strict podium overlap**, complete walk-forward development windows:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| forecast grid order alone | 0.6284 | 0.5476 |
| standings + frozen logit + forecast grid, equal rank average | 0.6393 | 0.5762 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/b8_grid_forecast.py` trained one ridge regression (`alpha=300`, 100-race half-life) on **strictly earlier** actual race-grid positions, using only the existing pre-weekend history features. Every development race was scored. It asked whether forecast qualifying/grid order supplies information missing from the direct podium model; the grid prediction was evaluated alone and as one fixed equal-rank vote alongside standings and the frozen logit. The target grid was an evaluation label only; target qualifying was unused. The grid forecast improved race-mean Spearman against actual grid over standings by +0.0115 inner (paired 95% CI [-0.0035,+0.0274]) and **+0.0453 outer** ([+0.0232,+0.0683]). Within the **pre-weekend standings top six**, the corresponding gains were +0.0084 ([-0.0628,+0.0796]) and +0.0310 ([-0.0384,+0.1012]); these are too uncertain to establish useful contender-order information.

The forecast-grid-only ranking lost **0.0429** outer strict overlap versus standings (paired race bootstrap 95% CI [-0.0810,-0.0048]); the three-way blend lost 0.0095 versus frozen `ens` ([-0.0286,+0.0095]) and 0.0143 versus standings ([-0.0524,+0.0238]). Thus the better full-field grid forecast did not become a better podium forecast. `tests/test_grid_forecast_leak.py` reverses target-grid and qualifying labels and confirms unchanged predictions, then changes earlier training grids and confirms sensitivity; the existing feature scramble, pre-weekend, profile, guard, and metric checks passed. No 2026 evaluation was run. These development windows have been inspected repeatedly, and the usual roughly five-point detection limit remains relevant.

**Verdict: no strict podium gain; stop using mean forecast grid order as a podium rank vote.** This is a feasibility gate, not a test of a calibrated forecast-grid *distribution* or a correctly trained conditional podium model. Any such B8 follow-up must use out-of-fold historical grid forecasts for its own training rows and justify added complexity with a contender-specific gain, rather than using the true grid in training and a forecast grid at prediction time.

### E115 — Out-of-fold forecast grid as one podium feature (B8 follow-up)

**Strict podium overlap**, all development races scored:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| fixed logit + prior grid forecast | 0.6393 | 0.5571 |
| standings + augmented logit rank average | 0.6284 | 0.5857 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/b8_grid_feature.py` reran E114's fixed ridge grid forecast for **each historical race**, training on only earlier races; it then added that out-of-fold forecast as one feature to the fixed podium logit (`C=0.03`, 100-race half-life) and kept the equal-rank standings blend. This avoids training the podium model on true historical grids while asking it to interpret a forecast grid at the target race. The 2019 training rows used 2018 and earlier-2019 grids, and all 420/1,220/1,398 driver-race rows in the 2019/inner/outer periods had a forecast. The feature added no new source and used no target-weekend data. The augmented ensemble changed **0/61 inner and 0/70 outer** top-three sets. Its strict-overlap difference from frozen `ens` was exactly zero (paired bootstrap interval [0,0] in both splits); its outer difference from standings was -0.0048 (95% CI [-0.0381,+0.0238]).

Standalone logit Brier/log-loss were 0.07719/0.24968 to 0.07718/0.24957 inner, and 0.08314/0.26293 to 0.08312/0.26280 outer. The outer race-paired differences were -0.000012 for Brier (95% CI [-0.000068,+0.000045]) and -0.000129 for log-loss ([-0.000265,+0.000020]); neither excludes zero. `tests/test_grid_forecast_leak.py` now checks the nested forecast feature as well: changing the target grid leaves it identical, while changing an earlier grid changes it. The existing pre-weekend feature scramble and guard tests also passed. No 2026 evaluation was run. This is another look at the same repeatedly inspected development windows.

**Verdict: no podium gain from the time-aligned mean-grid forecast.** E114's full-field grid correlation does not translate into a useful third-place decision through this simple conditional model. A calibrated grid distribution remains untested, but the weak contender-specific grid gain in E114 and zero selection changes here do not justify promoting this feature or tuning many variants on these races.

### E116 — Portable driver pace from the matched teammate graph (B6)

**Strict podium overlap**, full walk-forward development windows:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| fixed logit + portable driver pace | 0.6393 | 0.5762 |
| standings + augmented logit rank average | 0.6339 | 0.5857 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/b6_portable_teammate.py` reused E108's 758 matched-lap teammate comparisons across 42 drivers in 2018–25. For each target race, it fitted one weighted driver graph from **strictly earlier** pair gaps: driver A minus teammate B, with lap-count weights, a fixed 10-race half-life, and ridge anchoring of 50. One resulting driver pace rating was added to the frozen regularized podium logit (`C=0.03`, training half-life 100), then rank-averaged with standings. This differs from E108–E109's exact same-team/same-teammate history: a connected graph can carry a driver's earlier contrasts into a new pairing. Missing ratings used the existing logit median imputer; feature coverage was 98.77% inner / 98.86% outer. The ensemble changed 2/61 and 8/70 top-three sets, but its outer strict-overlap difference from frozen `ens` was effectively **zero** (paired race bootstrap 95% CI [-0.0238,+0.0238]); versus standings it was -0.0048 ([-0.0239,+0.0143]).

The standalone logit's driver-race Brier/log-loss improved from 0.07719/0.24968 to 0.07659/0.24853 inner and from 0.08314/0.26293 to **0.08205/0.26024** outer. The outer race-paired differences were -0.001082 for Brier (95% CI [-0.002013,-0.000132]) and -0.002690 for log-loss ([-0.004847,-0.000447]); inner intervals included zero. The Brier difference was negative in each development season, ranging from -0.00023 to -0.00127. This probabilistic signal remains exploratory after many looks at the same outer races and does **not** pass the strict podium gate. For a narrow transfer check, team lineages were normalized across name changes in local results (e.g. Toro Rosso/AlphaTauri/RB, Renault/Alpine); only **6 inner and 7 outer** first-race team-change pairs had a matched same-race pace gap and prior ratings, and the graph picked the faster driver in 2/6 and 5/7. Those tiny samples cannot establish a transferable driver skill effect, and disconnected teammate graph components are only ridge-anchored, not absolutely comparable.

`tests/test_portable_teammate_leak.py` verifies the graph's sign on a synthetic pair and scrambles target-and-later matched gaps and counts on cached data; target ratings remain unchanged, while changing a prior gap changes the rating. Existing pre-weekend feature, profile, guard, and metric checks passed. No 2026 evaluation was run.

**Verdict: better probability calibration, no podium gain over standings or the frozen ensemble.** Keep the portable pace rating as a research measurement, not a replacement predictor. Before another podium model attempt, show that it separates P3 from P4 on genuinely later races; retuning on the inspected 2023–25 window would overstate the evidence.

### E117 — Does portable pace help among plausible podium contenders? (A8/B6 diagnostic)

**Strict podium overlap first:** this is a diagnostic of the unchanged E116 predictions, not a new selector. Plain standings remains **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, E116's portable-feature ensemble 0.6339 / **0.5857**, and the label-shuffled logit null 0.1749 / 0.1857. No 2026 evaluation was run.

`experiments/b6_portable_teammate.py` now fixes a **pre-weekend** contender set: the six highest-ranked drivers by plain standings in each race. It partitions E116's logit probability errors by that set and its complement, and computes within-contender AUC for separating actual podium finishers from non-finishers. Target-race podium labels are used only for these retrospective diagnostics. One inner race had no positive/negative mixture among the six and was omitted from AUC only; strict overlap and Brier retained all races.

| diagnostic, portable minus frozen logit | inner 2020–22 | outer 2023–25 |
|---|---:|---:|
| Brier among standings top six, race-paired | -0.002221, 95% CI [-0.004630,+0.000260] | **-0.003068**, [-0.006020,-0.000160] |
| Brier among remaining field, race-paired | +0.000091, [-0.000132,+0.000336] | -0.000233, [-0.000609,+0.000152] |
| top-six podium AUC, frozen → portable | 0.7386 → 0.7352 (60 races) | 0.7133 → 0.7323 (70 races) |
| top-six AUC difference, race-paired | -0.0033, [-0.0191,+0.0111] | +0.0189, [-0.0028,+0.0411] |

The outer probability gain is concentrated among plausible contenders, not merely backmarkers. The AUC gain is uncertain and was +0.0044/+0.0170/+0.0341 in 2023/24/25, while 2020–21 were negative. The frozen and augmented ensembles still had identical outer strict overlap, with eight different top-three sets balancing out in aggregate. This repeatedly inspected outer window does not provide an independent confirmation of the contender-order gain. The source-specific leak test, pre-weekend guard, and metric checks passed; the feature, model weights, and predictions did not change from E116.

**Verdict: potentially useful contender probabilities, no demonstrated path to 0.750 strict overlap.** Preserve the portable pace measurement for genuinely later evaluation, but do not select a new threshold or weight from these 70 outer races.

### E118 — Access and terms gate for historical market/fantasy data (C1/C2 feasibility)

**Strict podium overlap:** no new model or feature was scored. The development reference remains plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and the label-shuffled logit null 0.1749 / 0.1857. No 2026 evaluation was run.

The [Betfair Historical Data portal](https://historicdata.betfair.com/ExchangeHistoricalStore/) advertises timestamped exchange price data from 2015 onward, but its [developer support](https://support.developer.betfair.com/hc/en-us/articles/360008664937-Which-jurisdictions-is-Betfair-Exchange-Historical-Data-available-to) says access requires a registered Betfair.com account, and the [historical-data API guidance](https://support.developer.betfair.com/hc/en-us/articles/12859956891932-How-Can-I-Make-HTTP-Requests-to-the-Historical-Data-API) says downloads must first be purchased through the portal. Public pages did **not** establish that driver-podium markets exist for the required races, have pre-weekend quotes and traded volume, or have a permissible licence for this project. No account was created, purchase made, licence accepted, or file acquired. The [F1 Fantasy terms](https://fantasy.formula1.com/en/terms) prohibit automated extraction from the game; no prices or pick rates were fetched. These are source-access findings, not model data or evidence of predictive quality.

**Verdict: C1/C2 cannot presently supply a reproducible, permitted pre-weekend backtest.** Keep them conditional; do not substitute winner odds, late-weekend prices, or post-race fantasy snapshots. Continue with permitted in-repo structured sources unless the user separately authorizes and supplies a suitable dataset or account/purchase path. No leak test was needed because nothing was ingested or used as a feature.

### E119 — Correct past race pace for fresh versus used tyre sets (A2/B7)

**Strict podium overlap**, complete walk-forward development windows:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| E111 joint pace alone | 0.6557 | 0.5714 |
| joint pace plus fresh-tyre correction, alone | 0.6557 | 0.5667 |
| standings + frozen logit + fresh-corrected pace, equal rank average | 0.6557 | 0.5905 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/a2_freshtyre.py` reran E111's clean historical race-lap measurement with **one** extra nuisance indicator: a stint started on a previously used set (`FreshTyre=False`). Everything else was fixed: lap filters, same-lap field residual, compound indicators, compound-specific tyre-age slopes, traffic flag, ridge penalty 50, and 10-race forecast half-life. The [FastF1 lap-data specification](https://github.com/theoehrly/fast-f1/blob/main/docs/api_reference/timing_data.rst) defines `FreshTyre` this way and says `TyreLife` includes laps from earlier sessions on a used set. It also says modern `SOFT/MEDIUM/HARD` labels are **event-relative**, not underlying C1–C5 identities; this feed cannot support B7's absolute construction comparison. Source coverage matched E111: 3,291 driver-race rows across 171 races in 2018–25, 3,240 valid corrected gaps. Only completed, strictly earlier race rows contributed to target features; target-race laps were retrospective labels for diagnostics, never prediction features.

The new versus original corrected gap had correlation 0.9896 and mean absolute difference 0.0751 percentage points on 3,240 common driver-races. Next-race pace-order Spearman changed from **0.7139 to 0.7128 inner** (paired race difference -0.0011, 95% CI [-0.0060,+0.0033]) and from **0.7028 to 0.6988 outer** (-0.0040, [-0.0080,-0.0003]). Fresh-corrected pace alone lost 0.0048 outer strict overlap versus E111 joint pace (paired CI [-0.0238,+0.0143]). Its three-way blend gained 0.0048 over frozen `ens` outer ([-0.0143,+0.0238]) and **tied standings** (paired difference approximately zero, [-0.0286,+0.0238]). The inner blend's +0.0273 over `ens` did not carry into a material outer gain. These are repeatedly inspected development races, not an independent validation of a small difference.

`tests/test_freshtyre_leak.py` scrambles target-and-later corrected source gaps and target outcomes on synthetic and cached data; target forecasts remain unchanged, while editing an earlier gap changes the feature. Existing pre-weekend leak, profile, guard, and metric checks passed. No 2026 evaluation was run.

**Verdict: no podium gain; stop this single fresh-versus-used correction.** It slightly worsened the next-race pace forecast and did not beat standings at the strict podium boundary. Absolute compound allocations or construction changes remain unavailable from the permitted cached lap feed; do not infer C1–C5 from the relative labels.

### E120 — Short practice running in completed prior weekends as an attrition warning (A7)

**Strict podium overlap**, every development race scored with missing-feature fallback:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| frozen logit × generic all-cause survival | 0.6393 | 0.5619 |
| frozen logit × short-running all-cause survival | 0.6393 | 0.5619 |
| standings + generic survival logit, equal rank average | 0.6339 | 0.5905 |
| standings + short-running survival logit, equal rank average | 0.6339 | 0.5905 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/a7_prior_running.py` measured each driver's practice lap count relative to the same session's field median, then the team's fraction of driver-sessions below **half** that median. It retained only sessions with at least 15 timed drivers, did not call missing driver rows or absent whole-session feeds failures, and used only the **last three completed races** to forecast a team's target short-running tendency. Source data were existing local FastF1 session summaries for 2018/2020/part of 2021 and existing OpenF1 lap summaries for 2023–25. The overlapping 2023/25 driver-session ratios correlated 0.944 across feeds; the fixed <0.5 flag agreed on 99.89% of 1,853 matched rows. The derived source has 1,084 team-weekend rows over 109 races, 136 with nonzero short running. Target feature coverage was 94.12%/26.36%/**0%** in 2020/21/22 and 95.45%/99.16%/100% in 2023/24/25. Thus the inner period cannot establish a reliable gain from this signal. The sparse 2021 and absent 2019/2022 FastF1 practice cache were not filled with target-weekend information.

The candidate added **one** past-only short-running feature to a regularized all-cause DNF hazard using the same historical driver/team DNF rates and era feature as the generic comparator; both hazards weighted the frozen logit's podium probability by predicted survival. The all-cause DNF label was used because the outer years' Jolpica statuses do not identify mechanical causes (E101). Outer hazard Brier/log-loss changed from 0.11237/0.38472 to 0.11241/0.38498; race-paired differences were +0.000040 (95% CI [-0.000487,+0.000641]) and +0.000263 ([-0.001946,+0.002788]). Inner differences likewise included zero. The two hazard-adjusted logits and their standings ensembles selected identical top threes in both splits: **short-running minus generic overlap was exactly zero** (paired interval [0,0]). Both survival ensembles gained one equivalent correct podium pick over frozen `ens` outer (+0.0048, paired CI [0,+0.0143]) and **tied standings** (paired CI [-0.0286,+0.0286]); that one-pick change cannot be attributed to short running. This repeatedly inspected outer window is not independent confirmation of a small gain.

**Bug caught before reporting:** the first feature builder used decaying weights over *all* earlier available practice races. During missing-source years, normalization kept stale 2020/21 observations influential in 2023. The final builder limits source rows to the last three races and returns missing otherwise; only final numbers above are the experiment result. `tests/test_prior_running_leak.py` checks the three-race expiry and scrambles target-and-later source values and outcomes without changing target features. Existing pre-weekend leak, profile, guard, and metric checks passed. No 2026 evaluation was run.

**Verdict: no credible all-cause retirement forecast or strict podium gain.** Practice short running is rare and may reflect programmes, traffic, crashes, or missing timing rather than an unresolved fault. Do not promote this one-indicator A7 variant; cause-specific warnings remain untestable with the current outer status labels.

### E121 — Does a team's costly pit-stop tail persist? (B3 feasibility gate)

**Strict podium overlap:** no new podium selector was run because the source signal failed its next-event forecast gate. The unchanged development references are plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and label-shuffled logit null 0.1749 / 0.1857. No 2026 evaluation was run.

`experiments/b3_pit_tail.py` used the existing pit records from 172 completed races in 2018–25. A driver-race was marked as a costly stop only when its valid median pit-lane time exceeded the **same-race field median by more than three seconds**; missing or unpaired stops were unknown, not negatives. This fixed threshold was chosen from the source's time scale before testing future outcomes. There were 3,180 valid driver-races and 269 costly-stop labels. For each target race, a team's probability was estimated from strictly earlier valid rows with a 10-race half-life and a 20-observation prior centered on the strictly past global rate. This was compared with that global rate on the same observed target rows. The inner/outer evaluation had 1,108/1,306 driver-races and 99/96 costly stops, covering 60/70 races.

| costly-stop forecast, team minus past-global baseline | inner 2020–22 | outer 2023–25 |
|---|---:|---:|
| Brier, race-paired difference (95% CI) | +0.000515 [-0.000117,+0.001105] | +0.000168 [-0.000465,+0.000769] |
| log-loss, race-paired difference (95% CI) | +0.003121 [-0.000765,+0.006860] | +0.001198 [-0.003393,+0.005786] |

Outer Brier/log-loss were 0.068358/0.264613 for the team forecast versus **0.068188/0.263324** for the global rate. `tests/test_pit_tail_leak.py` checks that changing target-and-later pit outcomes leaves the target prior unchanged and a prior edit changes it; the existing pre-weekend leak, profile, guard, and metric tests passed. The records collapse multiple stops into each driver's median and mix crew delay, traffic, repair, and neutralization effects, so this is not a clean crew-performance label. The development races have also been inspected repeatedly.

**Verdict: no forecastable team-specific costly-stop tail from this source.** Stop before a podium ranking or undercut simulator based on this particular tail statistic; it offers no measured incremental execution signal over a global prior. This does not test richer stop-by-stop rejoin states or a race-specific strategy forecast.

### E122 — Freeze a genuinely prospective 2026 São Paulo prediction

**Strict podium overlap:** pending the 2026 round-20 result. No target outcome was scored and no model was changed. The development reference remains plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and label-shuffled logit null 0.1749 / 0.1857. The previously reported 2026 holdout was 0.544 over 19 races; while checking the results cutoff, round 19's actual top five (VER, HAM, ANT, LEC, NOR) was viewed again. This is a logged additional holdout look, with no tuning from it.

At **2026-11-03 02:58:49 UTC**, before the São Paulo weekend, `predict.py` used local results through Mexico City round 19 and the existing pace history to forecast round 20. The structured Jolpica calendar supplied only race name/date/circuit; the Mexico City lineup was carried forward as an explicit assumption. The frozen E8 `ens` predicted **antonelli, max_verstappen, hamilton** for the podium, in that order. Plain standings picked the same set and order. Top ten, input cutoff, parent commit, and evaluation status are frozen in `results/prospective_2026_r20.md`. No target-weekend sessions, grid, telemetry, labels, or outside-document text entered the model. `predict.py` now calls `assert_pre_weekend()` on its feature list and prints the standings comparator alongside the prediction.

**Verdict: prospective forecast registered, accuracy unknown.** This observation cannot establish 0.750; after the official result, score the immutable top-three set under the strict metric and log the look without changing this selection.

### E123 — Availability gate for pair-contact and matched-weather ideas

**Strict podium overlap:** no new selector or feature was scored. References remain plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and label-shuffled logit null 0.1749 / 0.1857. No further 2026 evaluation was run.

For B1, the [OpenF1 race-control schema](https://openf1.org/docs/) offers one `driver_number` and a free-text `message`, not a structured pair of incident participants. The local 2018–25 FastF1 cache has no race-control-message files. Parsing free-text steward/race-control statements would violate the current no-outside-documents input constraint, so the proposed driver-pair contact model has no eligible partner labels. A driver-only incident rate is already close to E18's broad retirement-risk work and does not test the pair hypothesis.

For B9, the local 2018–25 FastF1 cache has no weather-data files, and sample historical race sessions did not expose loaded weather. Existing `data/weather.csv` has race-level aggregates and forecast fields, not matched lap-time temperature/wind observations; E16 already tested the aggregate-weather avenue without a podium gain. A matched historical thermal/wind response cannot be measured from the current permitted cache. No weather feed was newly acquired.

**Verdict: park B1 pair-contact and B9 matched-weather modelling until a permitted structured, historically timestamped source exists.** This is a source-availability finding, not evidence that those physical effects are absent.

### E124 — Does pre-season testing mileage warn of early-season attrition? (A6 reliability follow-up)

**Strict podium overlap:** no new podium selector was run because the source failed its attrition-forecast gate. The unchanged development references are standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and label-shuffled logit null 0.1749 / 0.1857. No 2026 outcome was evaluated.

`experiments/a6_testing_mileage.py` counted recorded laps in the existing offline FastF1 pre-season testing sessions, then mapped driver codes to each season's round-one team entry list. All three testing days and 20 distinct drivers per year were available for 2023–25. This yielded 30/30 team-years with 267–456 laps per team-year, covering 179 driver-races and 28 all-cause DNFs in rounds 1–3. The one missing driver-race was not imputed. The fixed hypothesis was **lower** testing mileage predicting **more** early DNFs; the comparator was the existing past-only team 10-race DNF feature at round one. One renamed team's comparator was missing and received the same-season mean of the other nine teams' pre-weekend values. No target-race status entered a feature.

| Within-season Spearman with early team DNF rate | 2023 | 2024 | 2025 | Mean |
|---|---:|---:|---:|---:|
| low testing mileage as risk | -0.1562 | +0.0730 | +0.4095 | +0.1088 |
| past team DNF risk | +0.4752 | -0.4439 | +0.4045 | +0.1453 |

The mileage association changed sign across seasons and was below the past DNF comparator on average. A 5,000-draw label shuffle **within each season** had mean correlation +0.0005 and one-sided p=0.2835 for the observed +0.1088. This is only a 30-team-year, repeatedly inspected outer-window diagnostic. Raw 2020–22 testing laps are not present for an inner-period validation, and the all-cause DNF label mixes mechanical, collision, and other failures. `tests/test_testing_mileage_leak.py` verifies target labels and later-season testing edits leave a target year's mileage unchanged, while an earlier testing edit changes it. A paired podium bootstrap is inapplicable without candidate podium predictions.

**Verdict: no credible early-attrition signal from testing lap count.** Do not fit a podium adjustment from this statistic; it cannot establish progress toward 0.750. The pace information in E86 is a distinct finding and is not negated by this mileage result.

### E125 — Does relative first-stop timing persist by team? (B3 strategy gate)

**Strict podium overlap:** no new podium selector was run because the timing forecast failed its out-of-time error gate. Development references remain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and label-shuffled logit null 0.1749 / 0.1857. No 2026 outcome was evaluated.

`experiments/b3_first_stop.py` used the existing historical pit-stop extracts. For cars completing at least 90% of a race, it measured the first pit-in lap as a fraction of the race's winner-lap count, then subtracted that race's field median and averaged across each team's eligible drivers. This yielded 1,679 team-race measurements across 172 completed races in 2018–25. For each target, a team forecast used **strictly earlier** races, a fixed 10-race half-life, and 10 weighted observations of zero shrinkage; a missing history forecast zero. The comparator was the same zero relative-timing forecast for every team. Target race pit timing and finishing laps were used only to compute the retrospective outcome and eligible evaluation set.

| Past-only team forecast versus zero, first-stop residual | inner 2020–22 | outer 2023–25 |
|---|---:|---:|
| covered team-races / races | 594 / 61 | 686 / 70 |
| forecast-vs-actual Spearman | +0.1304 | +0.0654 |
| team forecast MSE | 0.009604 | 0.013971 |
| zero forecast MSE | 0.009629 | **0.013784** |
| race-paired MSE difference, 95% bootstrap CI | -0.000016 [-0.000295,+0.000252] | +0.000188 [-0.000090,+0.000471] |

Outer yearly Spearman was +0.1562/+0.0968/**-0.1077** in 2023/24/25, while team-forecast MSE was worse than zero in **each** outer season. A within-race shuffled-outcome null had mean rank correlation -0.0107 and one-sided p=0.023 for the pooled outer +0.0654; that small rank association is exploratory after many looks at this window and did not translate into lower timing error. `tests/test_first_stop_leak.py` scrambles target-and-later pit residuals on synthetic and cached data while leaving target forecasts unchanged. A paired podium bootstrap is inapplicable without a candidate podium selector.

**Verdict: stop this fixed team first-stop-timing forecast before a strategy simulator or podium feature.** The measurable rank association does not provide reliable timing magnitudes, and full-distance selection, neutralizations, tyre choices, and race-specific strategy remain confounders. This does not rule out other permitted, genuinely predictive execution data.

### E126 — Many-lap race speed-trap advantage as a pre-weekend feature (A1)

**Strict podium overlap**, full walk-forward development windows:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| frozen logit + past race SpeedST advantage | 0.6393 | 0.5571 |
| standings + augmented logit rank average | 0.6284 | 0.5857 |
| label-shuffled logit null | 0.1749 | 0.1857 |

`experiments/a1_race_speedtrap.py` read the existing offline FastF1 **race** lap cache, not target-weekend telemetry. It retained accurate, green, non-pit laps after lap one, excluded lap-time outliers above 107% of each driver's median, and required at least six cars on the same lap and eight retained laps per driver. For each lap it subtracted the field median SpeedST at that lap line, then took each driver's median residual and the team's mean. The source had **3,234 driver-race rows across 171 races in 2018–25**; one 2018 race was unavailable. A team's target feature was a 10-race-half-life mean from **strictly earlier** source races, shrunk toward zero by ten team-race observations. Feature availability was 99.1–100% by development season. The candidate added only that scalar to the frozen regularized podium logit (`C=0.03`, 100-race training half-life), then kept the equal-rank standings blend. This differs from E83–E85's qualifying-lap telemetry and E113's five early-race laps for six drivers. [FastF1's timing schema](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/timing_data.rst) identifies SpeedST as a circuit-dependent high-speed trap in km/h; documentation was consulted for measurement semantics, not supplied as model data.

The speed-augmented ensemble changed one outer top-three set and zero inner sets, but **no race's strict overlap**. Its paired strict-overlap difference versus frozen `ens` was exactly 0 in both splits (bootstrap interval [0,0]); versus standings it was -0.0048 outer (95% CI [-0.0381,+0.0238]). The past speed feature's mean within-race Spearman against *actual* target race pace was **-0.0571 inner / -0.0678 outer**, with the sign chosen so higher speed would imply faster pace. Thus the measured straight-speed tendency did not carry useful next-race pace order in this form. The standalone logit's Brier changed from 0.077192 to 0.077321 inner (race-paired difference +0.000128, 95% CI [-0.000081,+0.000341]) and from 0.083148 to 0.083177 outer (+0.000029, [-0.000063,+0.000124]); neither is a gain. `tests/test_race_speedtrap_leak.py` scrambles target-and-later speed summaries and target race labels without changing the target feature, while an earlier source edit changes it. Existing pre-weekend guard and feature leak checks passed. No 2026 outcome was evaluated.

**Verdict: no strict or probabilistic podium gain.** Speed-trap advantage can reflect DRS, setup drag, traffic, and circuit-specific trap placement, not just power or race pace. Stop this scalar feature. It does not rule out a genuinely many-stint, position-aligned corner profile with better control for stint state and traffic; that higher-cost measurement still needs its own transfer gate. The repeatedly inspected outer window and roughly five-point detection limit remain relevant.

### E127 — Multi-stint X/Y profile at Bahrain: measurement gate before expansion (A1)

**Strict podium overlap:** no podium selector or target-race feature was made. The development references remain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and shuffled-label logit null 0.1749 / 0.1857. No 2026 outcome was evaluated.

`experiments/a1_multistint_xy.py` kept E113's six drivers, Bahrain 2023/24 session pair, 50 position bins, same-race reference-lap alignment, and per-bin field-median subtraction. Its main measurement change was using clean race laps across stints rather than laps 6–10: lap one, pit-in/out, unusually slow field-median laps (a neutralization proxy), and >107%-of-driver-median lap-time outliers were removed by the existing OpenF1 lap cleaner. It also split the clean laps into early and late halves. Each driver-race contributed **34–52 clean laps** and both stint-spanning halves covered all 50 bins; 2023 Leclerc had the shortest run because he retired. Timestamp-matched car/position samples were mapped back to the selected lap and kept within its recorded duration. Each year's reference geometry was used only to measure that completed year's profile; target-year geometry was never a pre-weekend input.

The mean within-race split-half residual correlation was **0.570** over 12 driver-races, versus E113 Bahrain's 0.384 from five early laps. Same-driver 2023–24 residual correlation averaged **0.414** over six drivers (VER .546, PER .236, LEC .197, SAI .510, HAM .484, RUS .512), versus E113 Bahrain's 0.148. The actual identity mapping ranked **1/720** among six-driver permutations and **1/8** among teammate-only swaps, versus E113's 106/720 and 1/8. These are diagnostic permutation ranks, not independent-race p-values. Fuel, compounds, DRS, traffic, and driver-specific style are still mixed into the speed curves, and one circuit with six continuing drivers is far too small for a podium claim.

**Verdict: passes the pilot's measurement expansion gate** (cross-year mean >0.3 and teammate-swap rank 1/8). Extend this unchanged extraction to E113's Austria and Monza pairs before considering any pre-weekend predictor. No model promotion or 0.750 claim follows from Bahrain alone.

### E128 — Unchanged multi-stint X/Y profile across three circuits (A1 transfer gate)

**Strict podium overlap:** no new prediction or target-weekend feature was made. The unchanged references are standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and shuffled-label logit null 0.1749 / 0.1857. No 2026 outcome was evaluated. A paired podium bootstrap is inapplicable to this measurement-only diagnostic.

`experiments/a1_multistint_multitrack.py` applied E127's fixed clean-lap filter, 50-bin X/Y alignment, early/late halves, and same-race field-median subtraction to E113's **same six continuing drivers** at Bahrain, Austria, and Monza in 2023/24. Each of 36 driver-races contributed 34–65 clean laps, with all 50 bins present in both halves. The 2023 profile is treated as the earlier measurement and the 2024 profile only as a retrospective transfer outcome; no 2024 race data entered a 2024 pre-weekend prediction. Historical OpenF1 car and location data are [documented as freely accessible without authentication](https://openf1.org/docs/) and no outside-document text entered model inputs.

| circuit | within-race split-half residual correlation, mean | same-driver 2023–24 residual correlation, mean | true identity rank among 720 driver permutations | rank among 8 teammate swaps |
|---|---:|---:|---:|---:|
| Bahrain | 0.570 | 0.414 | 1/720 | 1/8 |
| Austria | 0.532 | 0.399 | 2/720 | 2/8 |
| Monza | 0.514 | 0.181 | 77/720 | 2/8 |

The combined same-driver residual correlation was **0.332** over 18 driver-circuit pairs (E113's five-lap counterpart: 0.141). Applying one identical driver permutation across circuits, the true mapping ranked **1/720** and **1/8** among teammate-only swaps (E113: 45/720 and 3/8). This is a clear improvement in **measurement persistence** for this fixed six-driver set, especially at Bahrain and Austria; Monza remains weak. The first combined-rank print incorrectly reported 0/720 because the actual and permutation means were reduced in different orders, yielding a floating-point tie discrepancy. The final calculation uses the same reduction order for both and reproduces rank 1/720. No model result was reported from the buggy rank.

**Verdict: promising transfer signal, still no podium evidence.** These 18 dependent pairs involve only three teams with the same drivers across two years; shared team hardware, driver style, setup, DRS, tyre/fuel state, and circuit evolution are not separated. The next gate is to repeat this exact profile on additional continuing midfield teams, then test whether profile information predicts *future pace residuals beyond the existing form/standings features* before building a podium model. The 2020–22 inner-period telemetry gap remains a validation limit.

### E129 — Replicate multi-stint X/Y profiles on three midfield teams (A1)

**Strict podium overlap:** no new podium selector was made; standings remains **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and label-shuffled logit null 0.1749 / 0.1857. No 2026 outcome was evaluated. There are no candidate podium predictions for a paired podium bootstrap.

Before extraction, three continuing 2023–24 lineups were fixed as a broader replication set: Aston Martin (Alonso/Stroll), Alpine (Ocon/Gasly), and Alfa Romeo/Sauber (Bottas/Zhou). These add six drivers to E128's six from Red Bull, Ferrari, and Mercedes. `experiments/a1_multistint_midfield.py` used **unchanged** clean-lap, stint-half, timestamp-matching, X/Y alignment, and 50-bin rules at the same Bahrain, Austria, and Monza race pairs. The added source passed a pre-extraction lap-count check without replacement: its 36 new driver-races had 32–65 clean laps and all 50 bins in both halves. Across all 12 drivers, the 72 driver-races had mean within-race split-half residual correlation **0.523**.

| circuit | same-driver 2023–24 residual correlation, all 12 | original front six | added midfield six |
|---|---:|---:|---:|
| Bahrain | 0.277 | 0.355 | 0.199 |
| Austria | 0.243 | 0.381 | 0.105 |
| Monza | 0.270 | 0.249 | 0.291 |
| three-circuit mean | **0.263** | 0.328 | 0.198 |

With the **12-driver field median** used consistently for both years, the true driver mapping ranked **1/64** among all independent teammate-pair swaps across the six teams. None of 10,000 random full-driver mappings matched its combined correlation (Monte Carlo upper-tail estimate 0.0001 with the +1 correction). These are exploratory identity diagnostics on dependent driver/circuit pairs, not a valid podium significance test; the 10,000-draw estimate is also bounded by its sampling resolution. A separate NumPy correlation calculation from the saved source reproduced combined mean **0.263103** and teammate rank **1/64**, and checked all 144 driver-race-half profiles had 50 unique bins with no duplicate keys. The first-six subgroup values differ slightly from E128 because the field median now includes 12 drivers. No target-year X/Y profile was used as a pre-weekend feature.

**Verdict: the physical profile shape repeats beyond the original front-running teams, but the added teams' signal is weaker and no strict podium gain has been shown.** This passes a measurement reliability gate, not the 0.750 outcome gate. The next costly step must show that earlier profile shape improves a *future* pace or P3/P4 forecast over standings and the existing pace/form inputs; repeating identity matching alone would not justify a podium model. OpenF1 still lacks 2020–22 race telemetry, so any full model would need a separately justified validation split or historical FastF1 acquisition.

### E130 — Does prior-year X/Y speed add future race-pace order? (A1 forecast gate)

**Strict podium overlap:** no full-field podium selector was run; the development references remain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and label-shuffled logit null 0.1749 / 0.1857. No 2026 outcome was evaluated. This diagnostic is only three 2024 races and the 12 continuing drivers in E129; it cannot establish a 0.750 strict result.

`experiments/a1_profile_future_pace.py` fixed one simple prospective scalar before inspecting the result: each driver's **2023 mean X/Y speed residual** across both halves and all 50 bins at the same circuit. The 2024 feature came only from that earlier profile; the 2024 race pace was a retrospective target. On the same 12 drivers at Bahrain, Austria, and Monza, it compared the speed order with pre-weekend standings (`d_season_pts`, with the existing `d10_order` tie-break), 2023 same-circuit race pace, and a fixed equal-rank average of standings and speed. No weight or threshold was tuned. `tests/test_profile_future_pace_leak.py` scrambles the 2024 source profile on synthetic and cached data without changing the 2024 prior feature, while a 2023 edit changes it.

| 2024 within-race Spearman with actual race pace | Bahrain | Austria | Monza | Mean |
|---|---:|---:|---:|---:|
| pre-weekend standings | 0.797 | 0.867 | **0.951** | **0.872** |
| prior-year mean X/Y speed | 0.755 | 0.657 | 0.769 | 0.727 |
| equal-rank standings + X/Y speed | 0.797 | 0.790 | 0.902 | 0.830 |
| prior-year same-circuit race pace | 0.511 | 0.531 | 0.818 | 0.620 |

The speed scalar forecasts some future pace order: a within-race shuffled-pace-label null had mean correlation +0.0003 and one-sided p=0.0002 for the observed +0.727. It beats the old same-circuit pace scalar on this small sample, **but loses to standings in all three races**. Its race-paired mean correlation difference versus standings was -0.145 (bootstrap 95% CI [-0.210,-0.042]); the equal blend was -0.042 ([-0.077,+0.000006]). With only three selected races, these intervals are descriptive and do not support generalization to the full calendar. The comparator uses 2024 pre-weekend standings; neither the 2024 grid nor any 2024 session data was a feature.

**Verdict: the profile's scalar speed level has predictive information but no incremental pace-order gain over standings, so do not build a podium selector from it.** E127–E129's repeatable *shape* is a distinct signal; showing its value beyond existing form would require more circuits/years and a locked forecast test, not a post-hoc weight search on these three races.

### E131 — Physics-weighted X/Y profile shape for future pace (A1)

**Strict podium overlap:** no full-field podium selector was run. The unchanged development references are standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and label-shuffled logit null 0.1749 / 0.1857. No 2026 outcome was evaluated. This remains a three-race, 12-driver future-pace diagnostic, not evidence for 0.750 strict podium overlap.

`experiments/a1_profile_time_weight.py` used the same saved profiles, drivers, circuits, and 2024 target pace labels as E130, with **one fixed physical projection** instead of a mean-speed scalar. The 50 bins are approximately equal distance, so a small speed advantage changes traversal time in proportion to `-Δspeed / field_speed²`; this gives more weight to slow bins without fitting class thresholds or weights from outcomes. The field speed and driver residuals came from **2023** only for each 2024 target. The candidate was ranked alone and in a fixed equal-rank average with pre-weekend standings; no podium model was fitted. `tests/test_profile_time_weight_leak.py` scrambles target-year profile speeds on synthetic and cached source data without changing the target feature, while an earlier source edit changes it.

| 2024 within-race Spearman with actual race pace | Bahrain | Austria | Monza | Mean |
|---|---:|---:|---:|---:|
| pre-weekend standings | 0.797 | 0.867 | **0.951** | **0.872** |
| prior-year time-weighted X/Y profile | 0.727 | 0.664 | 0.769 | 0.720 |
| equal-rank standings + time-weighted profile | 0.747 | 0.763 | 0.889 | 0.800 |

The physics-weighted scalar was slightly weaker than E130's unweighted mean-speed scalar (0.720 vs 0.727) and below standings on **all three** races. Its paired mean pace-correlation difference versus standings was -0.152 (bootstrap 95% CI [-0.203,-0.070]); the equal blend was -0.072 ([-0.104,-0.050]). These intervals are descriptive for three selected races only. A within-race shuffled-pace-label null had mean -0.0004 and one-sided p=0.0002 for the profile's +0.720, which shows non-random pace information but **not incremental information over standings**.

**Verdict: no reason to promote this profile projection to a podium predictor.** The repeatable 50-bin identity signal in E127–E129 has not yielded contender-order improvement through either of two fixed scalar summaries. More profile transformations on these same three races would be variant search; further A1 work requires broader time-ordered data and a locked incremental forecast test.

### E132 — Pre-weekend Kalshi podium-market prices (C1)

**Strict podium overlap first:** market-only **0.7222** (39/54 podium slots) versus plain standings **0.6852** (37/54), on the same **18 available 2025 races**. The race-paired market-minus-standings difference was **+0.0370**, bootstrap 95% CI **[0.0000,+0.0926]**, with 11.9% of bootstrap draws at or below zero. The shuffled-label pre-weekend logit null scored **0.1667** on those races. There is **no inner 2020–22 coverage** for this source; this is 18 races within the outer 2023–25 window, not a replacement for its 70-race standings result of 0.5905 or frozen `ens` result of 0.5857. No 2026 outcome or prospective prediction was evaluated.

`experiments/c1_kalshi.py` retrieved driver-podium contracts and hourly bid/ask candlesticks from [Kalshi's public event API](https://docs.kalshi.com/api-reference/events/get-events) and [historical candlestick API](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks), both documented without authentication. Each pre-registered market score is the **last bid/ask midpoint at or before Thursday 00:00 UTC**, three calendar days before the race date; higher midpoint ranks first. The [archived 2025 session schedule](https://github.com/theOehrly/f1schedule/blob/master/schedule_2025.json) places every such cutoff at least **24.5 hours before the first scheduled session** across rounds 3–24. Market-event to race-round mapping is fixed from calendar identity, not final close times. No target-weekend price, grid, session result, or outside-document text enters the ranking. `tests/test_kalshi_leak.py` scrambles after-cutoff bars and confirms no feature change, while changing an earlier bar does change it.

The API listed **22** 2025 podium events (rounds 3–24); rounds 1–2 had none. Four listed events (rounds **4, 5, 10, 11**) had no bid/ask pairs by the cutoff and were excluded *before scoring*; there was no outcome-based selection. Of **361** quoted driver contracts across the 18 eligible races, **216** had a recorded pre-cutoff trade. Median spread was **0.04**, 90th percentile **0.09**, maximum **1.00**; aggregate pre-cutoff event volume ranged from **236 to 126,045** contracts (median **19,440**). Thus late-season markets had substantial trading, while several early markets were sparse. The market and standings top-three sets differed in only two eligible races; both changes added one correct podium driver. An independent calculation directly from the saved price table and official result rows reproduced **0.7222**.

The first extraction used Friday 00:00 UTC. A schedule audit found this was only **30 minutes** before Las Vegas practice, so those results were discarded and the entire extraction rerun at Thursday 00:00 UTC. The final script also records markets that opened after the cutoff as unavailable; it does not backfill their later quotes. The discarded Friday CSV was removed. This fixes a cutoff-risk and a source-coverage bug before treating the score as evidence. [Betfair's own historical-data guide](https://www.betfair.com.au/hub/education/how-to-model/historical-data-sources/) additionally corrects E118's blanket purchase statement: its Basic tier is **free**, but has no volume; account access and actual F1 podium coverage are still unverified.

**Verdict: useful new pre-weekend market signal, but no demonstrated 75% accuracy or reliable incremental podium gain.** The 0.7222 result is below 0.750, spans only 18 races from one dominant season, and its +3.7-point lift is below the project's roughly ±5-point detection threshold even before accounting for repeated exploration. Do not tune a blend or declare a new model on this subset. Retain the timestamped source and test a fixed market comparator prospectively on genuinely new races only, without inspecting the held-out 2026 outcomes.

### E133 — Does driver-specific grid uncertainty predict a future setback? (B8)

**Strict podium overlap:** no new podium selector in this measurement gate. The complete walk-forward references remain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and shuffled-label podium logit null 0.1749 / 0.1857. No 2026 outcome was evaluated, and a paired podium bootstrap does not apply without candidate picks.

`experiments/b8_grid_variance.py` reused E115's out-of-fold grid forecast: each historical prediction was fitted only on earlier actual grids and pre-weekend features. It measured squared forecast errors only after those races had finished. For each target, the top six by **pre-weekend standings** received a 20-race-half-life global error variance and a driver-specific variance shrunk toward that global estimate by eight pseudo-observations. Driver-name ordering resolved standings ties; the target grid was solely a retrospective label. A Gaussian tail probability from each variance and the *same* forecast-grid mean predicted whether the contender would start outside the top six. This fixed comparison tests whether past volatility adds information beyond the mean forecast, not whether the Gaussian assumption is literally correct.

| future grid setback Brier, standings top six | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| global variance | 0.202007 | 0.205825 |
| past driver-specific variance | 0.201942 | **0.204555** |
| driver minus global, race-paired 95% CI | -0.000065 [-0.002446,+0.002547] | -0.001270 [-0.002656,+0.000180] |

The actual grid-setback rate was 0.306 inner / 0.305 outer among the six preselected contenders. A within-race shuffle of the *grid setback labels* gave mean driver-minus-global Brier differences +0.005114 / +0.000399; the observed difference was better than 99.8% / 99.55% of 2,000 shuffles. This suggests a driver-identity relation conditional on each race's setback count, **but** the absolute Brier improvement is tiny and the race-paired intervals include zero. The grid-label shuffle is a diagnostic null, distinct from the required shuffled-podium-label baseline. `tests/test_grid_variance_leak.py` changes target-grid labels without changing target variances and confirms that an earlier-grid edit does change them.

**Verdict: weak grid-risk information, no podium evidence.** One fixed podium-vote follow-up is justified to check whether this risk signal reaches the third-place boundary; do not tune the shrinkage, half-life, or probability transform on these repeatedly inspected races.

### E134 — Fixed forecast-grid distribution as a podium rank vote (B8)

**Strict podium overlap**, complete development walk-forward, no target-weekend information:

| model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| forecast-grid top-six probability alone | 0.6339 | 0.5476 |
| standings + logit + grid-probability equal rank vote | **0.6393** | 0.5714 |
| shuffled-label logit null | 0.1749 | 0.1857 |

`experiments/b8_grid_distribution.py` used E133's past-only variance for pre-weekend standings top-six drivers and the global past variance for the remaining field, with E115's out-of-fold mean forecast grid. It ranked each driver's Gaussian probability of starting in the top six, then added that ordering as one **fixed equal-rank vote** alongside the unchanged standings and regularized logit. Training rows, model regularization, missing-data treatment, and the no-grid tie-break remained the frozen settings; the probability vote used no target grid or qualifying result. The top-three set changed versus `ens` in 9/61 inner and 15/70 outer races. `tests/test_grid_distribution_leak.py` scrambles target grids without changing the target vote, while an earlier grid edit changes it.

The new vote gained **+0.0109** versus frozen `ens` in the inner period (race-paired bootstrap 95% CI [-0.0164,+0.0437]) and lost **-0.0143** outer ([-0.0381,+0.0095]). Against plain standings it gained +0.0328 inner ([-0.0109,+0.0765]) and lost **-0.0190** outer ([-0.0571,+0.0190]). The outer loss is four fewer correct podium slots than standings over 210 slots; it is below the project's roughly ±5-point detection limit, but there is no positive evidence here. No weight or variance parameter was tuned to the outer score, and no 2026 outcome was inspected.

**Verdict: reject this distribution vote.** E133's small grid-risk signal did not improve strict podium overlap in the competitive outer era. E114's mean-grid vote and E115's forecast-grid feature were already negative there; the fixed uncertainty follow-up does not justify a larger conditional podium simulator on the same data. Keep the frozen predictor unchanged.

### E135 — Register a prospective Kalshi comparator for 2026 São Paulo

**Strict podium overlap: pending.** The frozen pre-weekend 2026 round-20 model podium in `results/prospective_2026_r20.md` remains Antonelli, Verstappen, Hamilton; its outcome has not been scored. No new 2026 result, grid, practice, sprint, or qualifying data was read. The unchanged 2020–25 development references are standings 0.6066 inner / 0.5905 outer, frozen `ens` 0.6284 / 0.5857, and shuffled-label logit null 0.1749 / 0.1857. A paired podium bootstrap cannot be computed before an outcome exists.

The [Kalshi market API](https://docs.kalshi.com/api-reference/market/get-markets) listed the active `KXF1RACEPODIUM-BRAZ26` event on 2026-11-03 with 22 driver contracts, opened at 05:06 UTC. At the availability check, only **4/22** had traded, with **110.06** total contracts and a **0.24 median bid–ask spread**, so the opening snapshot was not treated as a credible market forecast. The [archived event schedule](https://github.com/theOehrly/f1schedule/blob/master/schedule_2026.json) puts first practice at 2026-11-06 15:30 UTC. E132's **unchanged Thursday 00:00 UTC** cutoff falls 39.5 hours earlier. The pre-registered market-only rule, missing-quote handling, and pending outcome are recorded beside the frozen prediction. Neither the current quote nor the future cutoff quote is an input to that prediction.

**Verdict: prospective test registered, result pending.** Capture the last hourly bid/ask midpoint at or before 2026-11-05 00:00 UTC after that time arrives, freeze the quote table, then evaluate only after the official race result. Do not use any later price or choose a model blend from this single race.

### E136 — FastF1 retirement-cause source gate (A4/B4)

**Strict podium overlap:** no new selector was run. The unchanged development references are plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and shuffled-label logit null 0.1749 / 0.1857. There are no candidate podium picks for a paired bootstrap. No 2026 outcome was evaluated.

E101 found no named mechanical failures in the official 2023–25 status field, so the cause-specific reliability idea needs a genuinely independent source. I checked FastF1 3.9.0's installed `Session._load_drivers_results` implementation and its cached race-result rows, with cache offline mode enabled. The implementation joins F1 timing-feed **driver information** to Ergast/Jolpica **race results**, including `Status`; its `_drivers_from_f1_api` maps names, team, number, and other driver metadata but no failure cause. Cached 2023 Bahrain and 2025 Australia both reported `Retired` for every retired driver (3/3 and 6/6), matching `data/results.csv`. The cached 2024 Bahrain race had no retirements and therefore does not test cause specificity.

**Verdict: stop this FastF1 route.** Its race-result `Status` is not an independent cause label and cannot rescue A4/B4 coverage. The sample check corroborates the code path but does not claim every historical row was audited. No target-weekend data, outside documents, new feature, or 2026 label entered a model. A4/B4 require a separate permissible structured cause source before another reliability experiment.

### E137 — Prepare the prospective 2026 São Paulo market capture (C1)

**Strict podium overlap: pending.** No new podium selector, 2026 result, or target-weekend session data was used. The unchanged development references are standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and shuffled-label logit null 0.1749 / 0.1857. The 2025 Kalshi result remains 0.7222 versus same-race standings 0.6852 on 18 available races (E132); this preparation adds no scored races and no paired bootstrap.

`experiments/c1_capture_2026_r20.py` now implements the E135 cutoff exactly: event `KXF1RACEPODIUM-BRAZ26`, last hourly bid/ask midpoint ending at or before **2026-11-05 00:00 UTC**, descending midpoint with driver-name ties, and market unavailable if fewer than three valid quotes. It refuses capture before that cutoff or at/after scheduled FP1 on 2026-11-06 15:30 UTC, and refuses to overwrite a previously saved table. It uses Kalshi's [live market-candlestick endpoint](https://docs.kalshi.com/api-reference/market/get-market-candlesticks) and [event endpoint](https://docs.kalshi.com/api-reference/events/get-event); 2026 live bars expose `close_dollars`/`volume_fp`, so the existing E132 quote parser now also accepts those fields while retaining its historical-field path. One live hourly candle was fetched to verify schema and parseability, but its price was not saved as a forecast or used to revise E122's picks. The capture script was run before cutoff and correctly refused without writing an output file.

`tests.test_kalshi_capture` checks early/late time guards, a bar after cutoff, cutoff-inclusive ranking, and name ordering; `tests.test_kalshi_leak` checks both historical and live field schemas. Both passed, as did `test_no_leak`, `test_preweekend`, `test_profile_leak`, `test_guard`, and `test_metrics`. The scheduled cutoff has not occurred, so **no quote table or market top three is frozen yet**. Run `PYTHONPATH=. .venv/bin/python -m experiments.c1_capture_2026_r20` after the cutoff and before FP1, then commit the saved table and top three before any weekend outcome is read.

**Verdict: reproducible prospective capture ready; accuracy remains unknown.** This changes neither the model nor the historical evidence for 0.750. Missing or sparse cutoff quotes must be reported as unavailable, not replaced by a later price.

**2026-11-04 scheduling update:** A one-time local Codex project automation, `freeze-s-o-paulo-podium-market`, is active for **2026-11-04 19:05 EST = 2026-11-05 00:05 UTC**. Its saved instruction is to run the fixed capture script only after cutoff and before FP1, preserve an existing output, verify bar timestamps and quote coverage, run the Kalshi and standard leak tests, then log and push the frozen result. The automation was created and its local configuration inspected; **it has not run yet**, and no 2026 market top three or race outcome has been observed. A failed or missed run must be reported rather than repaired with post-cutoff prices.

### E138 — Junior-series newcomer evidence: podium incidence and source gate (C5)

**Strict podium overlap:** no new selector was run. The unchanged references remain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and shuffled-label logit null 0.1749 / 0.1857. With no candidate podium predictions there is no paired bootstrap, and no 2026 result was inspected.

Before seeking a cross-series feed, I counted how often such evidence could directly inform an early F1 podium call. In `data/results.csv`, sorted by event date, an apparent newcomer is a driver whose first recorded F1 entry was in 2020–25 after the available 2018–19 history. This yields 17 drivers; `groupby(driver).cumcount()+1` gives entry number. Their first five entries span 78 driver-races and **zero podiums** (29/0 inner, 49/0 outer). Through entry 10 there was one podium, Antonelli's 2025 round 10 (0 inner, 1 outer); through entry 20 there were **four podium slots** (0 inner, 4 outer) among 281 newcomer entries: Piastri twice, Antonelli once, Hadjar once. The denominator is 393 official podium slots across 131 development races, so those four account for **1.0%** of slots. An independent check found exactly 393 rows with `order <= 3` and the same 393 with official `finish <= 3`, with no disagreement. Entry count includes any non-starter present in the result table, so it is not strictly a count of race starts; the 2018 starting boundary is also a history limit.

I did not ingest junior-series results. The [official FIA Formula 2 site terms](https://www.fiaformula2.com/en/latest/article/terms-and-conditions.6XzLrOyLLmDREjs3oMriKp) restrict extraction and copying of its material; a [third-party F2 pipeline](https://github.com/Alarchemn/F2-Data-Pipeline) found in the source check explicitly describes scraping that site. Under the project's rule to ask before restrictive scraping or licence acceptance, that route is excluded for now. Source documentation was read only to assess access, never as a feature.

**Verdict: deprioritize C5 for the 0.750 strict gate.** The incidence count does not prove junior-series skill has no later value, and it is not an upper bound on every possible effect of a newcomer feature. It does show that a direct correction to the first few F1 podium calls has little observed leverage on the full-race metric. A separately licensed, timestamped structured source would still need an incremental test against standings before use.

### E139 — Does a driver adapt after changing F1 teams? (B6 pace gate)

**Strict podium overlap:** no new podium selector was run. The full development references remain plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and shuffled-label logit null 0.1749 / 0.1857. No 2026 outcome was evaluated; there are no candidate podium picks for a paired podium bootstrap.

The E108 matched-lap teammate table had only **two** transfers with at least three measured races in both the first five and next five entries. For a broader gate, `experiments/b6_adaptation_gate.py` used the existing 2018–25 clean-race-pace table. It measured a switcher's percentage-point pace gap minus the same teammate's, averaged over entries 1–5 and 6–10 at the new team. Eligible transfers required at least three measured races in each window, the same teammate in both, and no prior spell at that team lineage. There were 7 eligible 2019, 8 inner 2020–22, and 11 outer 2023–25 transfers. The observed late-minus-early gap averaged -0.089, -0.221, and -0.065 points, respectively (negative means improvement); 4/7, 5/8, and 7/11 improved. This is weak evidence for a common adaptation effect, not a driver-specific estimate.

For each transfer, the candidate forecast for its entries 6–10 was the first-five relative-pace mean plus a global adaptation adjustment. The adjustment was the sum of completed **earlier** transfers' late-minus-early changes divided by their count plus ten zero pseudo-transfers; its shrinkage was fixed before this run. A transfer contributed to training only when its entry-6–10 window ended strictly before the target's first entry in that window. The baseline forecast was the unadjusted first-five mean. A secondary direction comparator used each teammate's **pre-event season-to-date race points** at that first late-window event, with a driver-name tie, not target-weekend grid or result.

| eligible transfer window | inner (8) | outer (11) |
|---|---:|---:|
| unadjusted relative-pace MSE | 0.20772 | 0.46712 |
| adapted relative-pace MSE | 0.19877 | 0.46595 |
| adapted minus unadjusted, paired 95% CI | -0.00895 [-0.03739,+0.01755] | -0.00117 [-0.05971,+0.06729] |
| faster-teammate direction: unadjusted / adapted / prior points | 5/8 / 5/8 / 7/8 | 7/11 / 8/11 / 8/11 |
| adapted minus prior-points direction, paired 95% CI | -0.250 [-0.625,0] | 0.000 [-0.273,+0.273] |

The outer MSE difference reversed by season: adapted-minus-unadjusted was **+0.00803 in 2023** and **-0.00883 in 2025**; no 2024 transfer met the window filter. `tests.test_adaptation_leak` changes a target switcher's entries 6 onward and confirms its pre-window forecast is unchanged, then changes entries 1–5 and confirms it moves. All standard no-leak, pre-weekend, profile-leak, guard, and metric tests passed. The cohort is **retrospectively conditioned on future pace availability and stable teammate membership**, so these are selected-window diagnostics rather than a full-field prospective performance estimate. Several drivers contribute multiple transfers, which the simple paired bootstrap does not cluster. Race-pace gaps also confound adaptation with upgrades, strategy, and teammate form.

**Verdict: no justified podium promotion.** The fixed adjustment has a negligible, uncertain outer MSE gain and ties the simple prior-points comparator on teammate direction. This pilot neither reaches the 0.750 gate nor rules out all driver adaptation; a new independent measurement or genuinely later transfer cohort would be required before another B6 podium model.

### E140 — Broader multi-stint X/Y profile transfer gate (A1/B5)

**Strict podium overlap:** no new selector was run. The unchanged full-window references are plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` 0.6284 / 0.5857, and shuffled-label logit null 0.1749 / 0.1857. No 2026 outcome was evaluated. A paired podium bootstrap does not apply until a candidate produces picks.

Before extracting any new profile, fix five previously untested 2024-to-2025 circuit pairs: **Jeddah, Suzuka, Silverstone, Singapore, Abu Dhabi** (2024 OpenF1 race session keys 9480, 9496, 9558, 9606, 9662). They cover early and late season and varied track types; the list will not be changed based on results. A source-coverage check found 2025 race-pace labels for 18/20, 20/20, 15/20, 20/20, and 20/20 drivers, respectively; this informed feasibility, not a replacement choice. Extract only **2024** race car/location samples with E127's unchanged clean-lap, two-half, X/Y, 50-bin method. Map each 2024 driver to the official results team, subtract the 2024 same-bin field median speed, average the residual over halves/bins and the team's available drivers, and carry that **one** prior-year scalar to the corresponding 2025 team. No 2025 session profile, telemetry, grid, or result is a feature.

On teams with a 2024 profile and observed 2025 race pace, compare the profile's 2025 team-pace order with (a) pre-event 2025 team championship points, (b) the same team's 2024 same-circuit race pace, and (c) a fixed equal-rank standings/profile blend. Report per-circuit and mean Spearman, paired race bootstrap versus standings, a within-circuit shuffled-pace-label null, and exact common-team coverage. The prior profile must beat standings on the new circuits before a podium feature is justified. This is a five-race outer-era measurement replication, **not** a full walk-forward podium or pristine holdout test.

The extraction produced **92 driver-race profiles** (18 Jeddah, 17 Suzuka, 19 Silverstone, 19 Singapore, 19 Abu Dhabi), each with both halves and all 50 bins; the saved source has 9,200 rows, no duplicate profile keys, and 21–59 clean laps per retained driver. All ten teams have at least one profile at every circuit. Early retirements fail the unchanged 16-clean-lap gate and remain missing. The 2025 pace target has 9–10 common teams per circuit.

| 2025 team race-pace Spearman | Jeddah | Suzuka | Silverstone | Singapore | Abu Dhabi | mean |
|---|---:|---:|---:|---:|---:|---:|
| pre-event 2025 team standings | 0.766 | 0.365 | 0.567 | 0.867 | 0.624 | **0.638** |
| 2024 X/Y team speed residual | **0.818** | 0.745 | **0.617** | **0.903** | **0.782** | **0.773** |
| 2024 same-circuit race pace | 0.661 | **0.770** | 0.483 | 0.891 | 0.624 | 0.686 |
| fixed standings + X/Y equal ranks | 0.840 | 0.665 | 0.583 | 0.939 | 0.745 | 0.754 |

The profile-minus-standings mean correlation was **+0.135** (race-paired bootstrap 95% CI [+0.045,+0.249]); the fixed blend was +0.117 ([+0.049,+0.209]). An independently written NumPy/pandas calculation from the saved 2024 CSV reproduced all five profile correlations. It also compared 2024 final team championship points on the same teams: their five-race mean correlation was about 0.696, below the profile's 0.773, though Suzuka favoured the points baseline. The within-circuit shuffled-pace-label null had mean correlation approximately zero and one-sided p=0.0002 for the profile's 0.773. This confirms real pace information but not incremental podium information. At the front, the profile identified **13/15** actual top-three *team-pace* slots versus standings **12/15**; the equal blend also identified 12/15. `tests.test_profile_2025_transfer_leak` scrambles 2025 profile rows on synthetic and saved source data without changing the prior feature, while changing 2024 source rows does change it.

**Verdict: the broader profile passes the pace-order gate, but its P3/P4 gain is weak and 0.750 podium accuracy remains unshown.** Five selected outer races and repeated exploration make the positive interval descriptive, not a discovery claim. Run exactly one fixed podium vote before deciding whether this deserves full-calendar historical extraction; do not tune weights or choose circuits from these results.

### E141 — Fixed X/Y team-profile podium vote (A1/A8)

**Strict podium overlap first**, on E140's five fixed 2025 races only:

| fixed five-race subset | strict overlap | correct slots / 15 |
|---|---:|---:|
| plain standings | 0.7333 | 11 |
| frozen `ens` | 0.7333 | 11 |
| prior-year X/Y team vote alone | 0.4667 | 7 |
| standings + logit + X/Y equal-rank vote | **0.8000** | **12** |
| shuffled-label logit null | 0.0667 | 1 |

`experiments/a1_2025_podium_vote.py` implements the one candidate registered above. It adds the **team profile rank as one equal vote** to unchanged pre-weekend standings and frozen regularized logit votes; both teammates share the profile vote and driver-name ties use `tie=None`. Training for each 2025 race uses only strictly earlier races; the X/Y source is 2024 only. No 2025 target grid, qualifying, sprint, practice, telemetry, race pace, or official result is a feature. The candidate changed two of five top-three sets versus `ens`. At Suzuka it replaced Russell with Piastri and gained one correct slot; at Silverstone it replaced Leclerc with Russell but both sets had two correct drivers. A separate direct intersection of the printed picks with official `data/results.csv` podium rows reproduced **12/15** candidate slots versus **11/15** for `ens`. The race-paired candidate-minus-standings and candidate-minus-`ens` differences were both +0.0667 (bootstrap 95% CI **[0,+0.2000]**). The source-specific 2025-profile scramble and the standard no-leak, pre-weekend, profile-leak, guard, and metric tests passed.

**Verdict: promising but far too small to claim 0.750.** This is one extra correct podium pick on five already inspected outer-era races, with a paired interval touching zero. E140's pace labels informed the decision to run this follow-up, so even its apparent 0.800 is exploratory. The standalone profile vote is weak, and E140 found only one extra top-three *team-pace* slot; the gain may be a favourable boundary case. Keep the frozen model unchanged. The next meaningful gate is the **entire 2024-to-2025 calendar**, with the same profile extraction and equal-rank vote, no circuit selection or retuning. No 2026 outcome was evaluated.

### E142 — Full-calendar 2024-to-2025 X/Y replication (A1/A8)

**Pre-registered before extraction: strict podium overlap pending.** E140–E141's five-circuit subset is insufficient for model promotion. Fix **all 24 circuits shared by the 2024 and 2025 calendars** before extracting any additional profiles; retain every race with available 2024 profiles and 2025 official result, and report missing-source coverage rather than dropping races for a better score. Reuse E127's unchanged lap/X/Y/50-bin method, E140's per-team mean residual, and E141's unchanged standings + logit + profile equal-rank vote. If driver 1 cannot supply the prior-race X/Y reference lap, use the lowest-numbered driver who can; this changes only the 2024 position reference, never the 2025 target. For an event with no 2024 team profile, the profile vote is neutral (all tied); a missing individual team profile receives the field-median speed residual (zero). Compare the same 24 2025 races with plain standings, frozen `ens`, and the shuffled-label null; show paired bootstrap, per-season/source coverage, and top-three changes. This is still outer-era development evidence after many experiments, not an untouched holdout. The 2026 prospective comparison remains separate.

The source pass completed **452 driver-race profiles at all 24 circuits**, 16–20 per circuit, each with two halves and all 50 X/Y bins (45,200 saved rows, no duplicate keys, 16–75 clean laps). Every scored 2025 race has a 2024 source; the 2025 race-pace comparison has ten common teams in 22 races and nine in two. Missing early-retirement driver profiles remain missing under the registered rule; each evaluated team has at least one source driver.

| all 24 races in 2025 | strict overlap | correct slots / 72 |
|---|---:|---:|
| plain pre-event standings | 0.6667 | 48 |
| frozen `ens` | 0.6528 | 47 |
| prior-year X/Y team vote alone | 0.4722 | 34 |
| standings + logit + X/Y equal-rank vote | **0.6944** | **50** |
| shuffled-label logit null | 0.1806 | 13 |

The fixed candidate changed 14/24 podium sets versus `ens` and gained three net slots; it gained two versus plain standings. Its paired race improvement over standings is **+0.0278** (bootstrap 95% CI **[-0.0417,+0.1111]**); over `ens` it is **+0.0417** (95% CI **[-0.0278,+0.1111]**). Direct intersections of all 24 printed top-three sets with official podium rows independently reproduced 50, 48, and 47 correct slots for the candidate, standings, and `ens`. The five E141 pilot races remain 12/15 versus 11/15; the other 19 give **38/57 versus 37/57** for the candidate and standings.

Across all 24 races the profile's mean 2025 team race-pace Spearman is **0.657** and the fixed standings/profile rank blend is 0.706. Pre-event team standings have mean 0.684 over **23 defined races**: round one has zero pre-event team points for all teams, so its Spearman correlation is undefined. On those same 23 races, profile mean is **0.660**, profile-minus-standings **-0.0239** (race-paired 95% CI **[-0.1153,+0.0660]**). The positive five-circuit E140 pace result therefore does not replicate calendar-wide. The `NaN` initially returned by the paired pace bootstrap was a reporting bug from passing round one's undefined standings correlation; the source and podium scores were unaffected. `experiments/a1_2025_fullcalendar.py` now excludes only that undefined pair from the pace bootstrap and prints its sample size. All 24 races stay in the strict podium score.

Only 2024 race car/location/lap samples were used to construct the added feature; the walk-forward 2025 logistic component still trains on strictly earlier races. `tests.test_profile_2025_transfer_leak` checked both saved source files by injecting 2025 rows and changing 2024 rows. The standard no-leak, pre-weekend, profile-leak, guard, and metric tests passed. No 2026 outcome was evaluated.

**Verdict: keep the frozen model.** The candidate has a modest, uncertain 24-race development gain and misses 0.750 by four correct podium slots; the standalone source is much weaker. Repeated inspection of the 2025 outer era and the small five-race pilot preclude a new held-out success claim. A time-forward replication on a different season, or genuinely prospective data, is needed before promoting this feature. Do not tune its weight on these 24 races.

### E143 — Independent-season X/Y pace gate (A1/B5)

**Pre-registered before extraction: strict podium overlap pending; no new podium selector is authorized by this gate.** Fix all **22 circuits shared by 2023 and 2024** (exclude 2024-only Shanghai and Imola) before extracting new data. Apply E142's unchanged multi-stint clean-lap, two-half, X/Y 50-bin extraction to the **2023** race sessions only, subtract each 2023 same-bin field median, and average driver residuals by official 2023 team. Use that prior-year team scalar to forecast **2024** team race-pace order. Compare mean per-race Spearman with pre-event 2024 team championship points, 2023 same-circuit race pace, and the fixed equal-rank standings/profile blend. Report exact circuit/driver/team coverage, race-paired bootstrap, and within-race shuffled-pace-label null. Round one's all-tied standings correlation is undefined and must be omitted only from paired pace comparisons; keep every race with a defined pace target in profile summaries. No 2024 target-weekend session data is a feature. This is a different development season, not a new untouched holdout after E142 was inspected. Only consider a podium replication if the prior profile adds credible team-order information versus standings; do not tune the profile or vote on the 2025 results.

The 2023 source pass saved **410 driver-race profiles at all 22 circuits** (16–20 per circuit), 41,000 rows, no duplicate keys, and both halves/all 50 bins for every retained driver. Clean-lap counts range 19–65; early retirements and short races for individual drivers remain missing. The first run stopped on a transient OpenF1 DNS failure before Abu Dhabi; the resumable script then completed from its saved profiles, with no circuit replacement. For the team comparison, 2023 Alfa Romeo maps to 2024 Sauber and 2023 AlphaTauri to 2024 RB; only their constructor names changed, and 2024 target standings/pace are untouched. The pace comparison has ten common teams in 20 races and nine in two.

| 2024 team race-pace Spearman | mean over 22 races | mean over 21 defined standings pairs | paired difference vs standings, 95% CI |
|---|---:|---:|---:|
| pre-event 2024 team standings | 0.8052 over 21 | 0.8052 | reference |
| 2023 X/Y team speed residual | 0.5184 | 0.5122 | **-0.2930 [-0.4095,-0.1902]** |
| 2023 same-circuit race pace | 0.5723 | 0.5709 | -0.2343 [-0.3178,-0.1505] |
| fixed standings + X/Y equal ranks | 0.7127 | 0.7199 | -0.0853 [-0.1330,-0.0419] |

The profile's within-race shuffled-pace-label null had mean approximately zero and one-sided p=0.0002: the profile contains real team-pace information but much less than current standings in this season. A separate raw-CSV pandas/Spearman calculation reproduced **0.5184** profile mean, **0.8052** standings mean, and **-0.2930** paired difference on the same team/race set. The source-specific cutoff test injected target-year rows into both 2023 and 2024 files without changing prior features, then changed prior-year speed and confirmed a change. Standard no-leak, pre-weekend, profile-leak, guard, and metric tests passed. No 2024 target-weekend session data or 2026 outcome entered the model.

**Verdict: stop this scalar team-profile route.** The 2023→2024 independent-season replication reverses E140's five-circuit positive pace result and strongly trails standings; even its fixed blend loses. No podium vote was run on this gate because the preregistered premise failed. Strict podium overlap stays unmeasured for E143; the unchanged full development references are plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` **0.6284 / 0.5857**, and shuffled-label logit null **0.1749 / 0.1857**. The E142 2025 candidate remains exploratory at 0.6944, not a promoted selector. A different measurement of race-specific team behaviour would be needed to revisit A1/B5.

### E144 — Close first-stop undercut measurement gate (B3)

**Pre-registered before extraction: strict podium overlap pending; no podium selector will be fitted at this gate.** Use only the existing offline 2018–25 FastF1 race-lap cache. For each race, take each driver's first recorded pit-in lap. An opportunity is an earlier pitter running **directly behind** another car at the end of the preceding lap, in positions **P2–P8**, with a timing-line gap of **0–2 seconds**. The car ahead must make its first stop **1–5 laps later**. Both must have known positions one lap after the later stop, and all their available laps from the preceding lap through that post-stop lap must be green; require accurate, non-pit laps at the before/after comparison lines. An undercut conversion is a reversal of their relative positions at that post-stop line. Record the earlier pitter's team, opportunity count, and conversion, while excluding pairs lacking complete timing or stop data. This measures a team/strategy event from a **completed prior race**, not a target-weekend feature.

First report source coverage, a few auditable examples, and the conversion base rate. Proceed to a time-ordered, beta-smoothed prior-team conversion forecast versus a past-global-rate Brier/log-loss comparator only if there are at least **100** eligible opportunities across 2020–25 and at least **five teams with ten or more** opportunities; otherwise stop for insufficient data. A podium model would require an additional independent forecastability result and explicit pre-weekend feature test, not just evidence that undercuts occur. Do not infer a causal pit-crew effect from this observational label: tyre offsets, traffic, and strategy choice remain confounders.

**Pre-extraction timing correction:** A cached 2023 Bahrain pit sequence shows `PitInTime` on lap *p*, `PitOutTime` on *p+1*, and the first accurate, non-pit comparison lap on *p+2*. Therefore interpret “one lap after the later stop” as the first clean lap after pit exit, **later pit-in lap +2**. The initial one-race dry run produced zero candidates under the impossible non-pit requirement at *p+1*; it was not scored or used to select a threshold. All other registered filters remain fixed.

The offline extraction found **74 opportunities in 55 races** across 2018–25, of which **45 converted** by the registered position-reversal label (60.8%). By year, eligible opportunity counts were 6, 11, 4, 14, 4, 9, 13, and 13 for 2018 through 2025. The **2020–25 development period has only 57 opportunities**, and its largest team counts are Ferrari, Mercedes, and McLaren at **nine each**; **zero teams** have ten. One 2018 race (round 14) could not load from the offline cache, as in prior FastF1 extractions. The source CSV has no duplicate race/attacker/defender/first-stop keys. Examples include 2025 round 4 Norris behind Russell by 0.975 seconds at P3 before Norris's lap-10 first stop and Russell's lap-13 stop; Norris remained behind at the registered lap-15 comparison line. Verstappen behind Antonelli by 1.198 seconds at P7 in the same race did reverse order after stops at laps 10 and 12. These examples check the timing and position label, not causal undercut attribution.

`tests.test_undercut_gate` constructed a synthetic conversion and confirmed that a gap above two seconds or a caution in the window removes it. The real source used historical race timing only, with no 2026 row or target-weekend feature; thus no source-to-feature leak path was added. There was no team forecast, no Brier/log-loss comparison, and no paired podium bootstrap because the registered sample thresholds failed before modelling.

**Verdict: stop the close first-stop team-effect route for insufficient observations.** Even though an observed position reversal is common within this selected sample, only 57 development opportunities spread across many teams cannot support a stable team-specific prior or a claim toward 0.750. The strict podium result is unmeasured for E144; full-window references remain plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` **0.6284 / 0.5857**, and shuffled-label logit null **0.1749 / 0.1857**. A broader pair definition would be a new, preregistered measurement with its own validation, not an after-the-fact rescue of this gate.

### E145 — Team-relative tyre-degradation transfer gate (A3)

**Pre-registered before extraction: strict podium overlap pending; no new selector at this measurement gate.** E104 measured driver-minus-teammate stint-degradation slopes and found little repeatability. That table cannot test a team effect because each teammate pair sums to zero. Reuse its existing 2018–25 offline FastF1 race-lap filters: green, accurate, non-pit laps after lap one; lap time below 107% of the driver's median; lap-time residual against the same-lap field median; each driver-stint-compound slope requires at least eight laps and six tyre ages and absolute slope at most one second per lap. Average repeated same-compound stints for a driver by usable-lap count, subtract the **same-race, same-compound field median slope**, average each driver's compound residuals, then average the team's available drivers. Retain one team-race outcome and its measured-driver count. Use the existing historical constructor-lineage mapping from E139 across name changes.

For each 2020–25 target race, predict the team's relative degradation only from **strictly earlier** team-race rows using a 10-race half-life and ten zero pseudo-observations. Compare to a zero team-effect baseline on identical covered team-races by MSE and a race-paired bootstrap; also report forecast/outcome correlation, per-season coverage, and a within-race shuffled-outcome null. Do not select a circuit, compound, or weight from the outcomes. Consider a podium feature only if the candidate beats zero-MSE on both inner 2020–22 and outer 2023–25, with an outer paired interval excluding zero. Otherwise stop. The target race's tyre data is a retrospective label only, never a pre-weekend feature.

The fixed extractor produced **1,701 team-race slope outcomes across 171 races in 2018–25** (1,535 with two measured drivers, 166 with one), without duplicate event/team keys or missing slope values. One 2018 race lacks usable offline lap data. Development coverage was 597 team-races / 60 races inner and 697 / 70 outer; each target forecast used only earlier team-lineage source rows. The relative slope is in seconds of extra lap-time residual per tyre-age lap, after same-compound field-median subtraction. The measured team values range from -0.203 to +0.402 seconds/lap, so a few extreme but still E104-filter-eligible team outcomes remain.

| pre-weekend team-degradation forecast | inner 2020–22 | outer 2023–25 |
|---|---:|---:|
| candidate MSE | 0.0016360 | 0.0007904 |
| zero team-effect MSE | 0.0017098 | 0.0007897 |
| candidate minus zero, race-paired 95% CI | **-0.0000740 [-0.0001292,-0.0000233]** | **+0.0000006 [-0.0000167,+0.0000177]** |
| forecast/outcome Pearson correlation | 0.1904 | 0.0289 |
| within-race shuffled-outcome one-sided p | 0.0002 | 0.2090 |

The candidate improved MSE in each inner year (2020–22) but not consistently in the outer era: candidate-minus-zero MSE was +0.000008 in 2023, +0.000011 in 2024, and -0.000017 in 2025. An independently written per-team/per-event NumPy loop reproduced every forecast to floating-point precision (maximum difference 7×10⁻¹⁸) and the inner/outer team-row MSE means. `tests.test_team_degradation_leak` scrambled target-and-later raw source values on synthetic and saved data without changing a target forecast, then changed an earlier value and verified a response. The source CSV audit found only 2018–25 rows, zero duplicate keys, and the expected one/two-driver coverage. Standard no-leak, pre-weekend, profile-leak, guard, and metric tests passed. No 2026 outcome or target-weekend input was used.

**Verdict: stop before a podium model.** A small team degradation signal exists in the inner era but disappears on the outer development window, failing the registered two-era MSE gate and providing no evidence for a 0.750 strict podium selector. E145 has **no new strict podium score**; the unchanged full-window references remain standings **0.6066 inner / 0.5905 outer**, frozen `ens` **0.6284 / 0.5857**, and shuffled-label logit null **0.1749 / 0.1857**. The observational slope mixes tyre choice, stint windows, traffic, setup, and car behaviour; its inner correlation is not a causal tyre-management estimate.

### E146 — Registered prospective market cohort for 2026 rounds 21–23 (C1; pending)

**Strict podium overlap: pending future race outcomes.** E132's 2025 market-only 18-race result was 0.7222 versus same-race standings 0.6852, with a paired interval touching zero; E137's Brazil 2026 capture remains scheduled separately. Before the remaining 2026 weekends, fix the cohort as **all three remaining races** from the structured Jolpica schedule: Las Vegas round 21 (race 2026-11-22 04:00 UTC), Qatar round 22 (2026-11-29 16:00 UTC), and Abu Dhabi round 23 (2026-12-06 13:00 UTC). Their OpenF1 structured FP1 start times currently are 2026-11-20 00:30, 2026-11-27 13:30, and 2026-12-04 09:30 UTC, respectively. Use the same **Thursday 00:00 UTC** quote cutoff as E132/E137: **November 19, November 26, December 3**. The cutoff precedes FP1 in every case. Do not move a cutoff or omit an unavailable race after observing market or race outcomes.

At each cutoff, query Kalshi's public `KXF1RACEPODIUM` event series. Match **one** 2026 event whose title names that Grand Prix and “Podium Finishers”; if none or multiple match, record the market as unavailable. For each contract open before cutoff, use E137's last hourly bid/ask midpoint ending at or before cutoff. Rank valid midpoints descending with driver-name ties; fewer than three valid contracts means unavailable. Persist the full quote table or an immutable unavailable record before FP1, then compare market-only top three with the frozen standings/`ens` picks after official results arrive. Report strict overlap first, same-race standings and shuffled-label null, quote coverage, paired bootstrap, and the complete race denominator; the tiny prospective sample cannot establish a general 0.750 capability by itself. No target-weekend session data or outcome may enter a prediction.

On 2026-11-04 the live public series listed **Brazil only**. It had no round-21/22/23 event yet, so this registration saves no future quote and makes no performance claim. [Kalshi's event-list API](https://docs.kalshi.com/api-reference/events/get-events) documents `series_ticker`, status, and pagination filters; its [candlestick API](https://docs.kalshi.com/api-reference/market/get-market-candlesticks) specifies that `end_ts` includes bars ending at or before the requested time. Those API documents define extraction semantics; their text is not a model feature.

`experiments/c1_capture_2026_remaining.py` now implements the fixed three-round schedule, paginated event lookup, unique year/title match, hourly pre-cutoff quote ranking, an immutable JSON record for either a full quote table or unavailable status, and guards against capture before cutoff or after FP1. It was run before the round-21 cutoff and correctly refused without writing a file. `tests.test_kalshi_2026_remaining` checks all three time windows, unavailable events, quote ordering, exclusion of a post-cutoff bar, and refusal to overwrite output. Its first mock run exposed a test-only market-path parsing error; a second run exposed an output-directory creation bug in an empty temporary workspace. Both were fixed, and the test passed. The existing Kalshi capture/source tests and standard no-leak, pre-weekend, profile-leak, guard, and metric tests also passed. **No future-round quote is frozen yet**; the 0.750 gate remains unshown.

Three one-time local Codex project automations are active for **2026-11-19, 2026-11-26, and 2026-12-03 at 00:05 UTC**, five minutes after each fixed cutoff. Their saved local recurrence rules and project target were inspected; the Indianapolis 19:05 EST run times were independently converted to those UTC times. Each automation is instructed to preserve existing output, record unavailability rather than use later quotes, run checks, and commit/push before FP1. Scheduling is preparation, not evidence of execution; quote coverage and accuracy remain pending.

### E147 — Registered Manifold podium-market source and 2024 backtest (C1; pending)

**Strict podium overlap: pending.** Manifold's [public API](https://docs.manifold.markets/api) supports read-only market search and timestamped bets without an account, and a 2026 São Paulo market has 23 independent driver-podium answers (`MULTIPLE_CHOICE`, `shouldAnswersSumToOne=false`). Before retrieving historical prices, fix **all 24 official 2024 races** as the coverage denominator. Discover candidate markets from the union of three fixed public searches: `2024 Formula 1`, `2024 Grand Prix podium`, and `2024 F1 podium`, sorted newest with up to 1,000 results each. Keep only market questions naming one 2024 F1 Grand Prix and podium finishers, with independently resolvable driver answers and market/answer creation before that race's cutoff. Use the race-name alias table in code; never select a question by its resolved answer or final probability. The cutoff is **00:00 UTC three calendar days before the official race date**, matching E132's date-based rule; a structured-session check showed it precedes FP1 at all 24 races, including the Saturday Bahrain and Jeddah races.

For each eligible market, reconstruct each driver answer's last pre-cutoff `probAfter` from a **filled, non-cancelled normal bet** with `createdTime` and every fill timestamp at or before cutoff; exclude limit bets or answers created later. Count only answers with such a quote. If several eligible markets cover one race, choose the one with the greatest **pre-cutoff eligible bet amount**, then earliest market creation and lexical market ID. A race is market-available only with at least three mapped, quoted drivers. First report candidate-market and available-race coverage; proceed to scoring only if at least **ten** races are available. Score the market-only top three by descending reconstructed probability, driver-name ties, against official podiums; compare plain pre-event standings, frozen `ens`, and the label-shuffled logit null on the **same races**, with a paired race bootstrap. Also show all 24 race statuses and quote coverage. A 2024 score is repeatedly inspected outer-era development evidence, not a pristine holdout or proof of 0.750. No descriptions, comments, later bets, final market volume/probability, target-weekend sessions, or target race labels may enter the selection or feature.

The initial metadata-only search found at least a dozen 2024 independent-driver podium-question candidates, but no historical `probAfter` values were inspected before this registration. Manifold's API documentation says `GET /v0/bets` can filter by contract and `beforeTime`, and exposes `createdTime`, `probAfter`, `isFilled`, and fills; the implementation must still verify actual response semantics and avoid partially filled or later-filled orders. API data use is limited to this non-commercial research project; no account, purchase, data dump, or licence acceptance was performed.

**Result (2024 outer-era subset; inner 2020–22 unavailable for this source):** The fixed search returned 514 distinct markets. Sixteen independently resolved driver-podium questions were created before their race cutoffs, covering 14 of the 24 official races; all 14 had at least three mapped drivers with valid pre-cutoff filled-bet quotes. The other ten races are marked unavailable in `data/parts/manifold_2024_audit.json`: rounds 1, 9, 10, and 18–24. Quoted-driver counts per available race ranged from 5 to 11. Two eligible questions existed for Japan and Monaco; the registered pre-cutoff eligible-amount rule selected one each. The chosen Monaco market had substantially more eligible amount than other markets, but most covered markets were thin. Each stored quote includes the originating bet ID and timestamp; no final answer probability or post-cutoff fill was used.

| strict podium overlap on the same 14 races | hits / 42 slots | mean |
|---|---:|---:|
| Manifold pre-cutoff top three | 22 | **0.5238** |
| plain standings | 23 | 0.5476 |
| frozen `ens` | 24 | 0.5714 |
| label-shuffled logit null | 13 | 0.3095 |

The race-paired bootstrap gives market minus standings **-0.0238** (95% CI **[-0.1190, +0.0714]**), market minus frozen `ens` **-0.0476** (**[-0.1429, +0.0476]**), and market minus shuffled-label null **+0.2143** (**[+0.0476, +0.3571]**). The market matched all three podium drivers at Saudi Arabia and Emilia Romagna, but none at Austria. This sample is too small and selected by market existence to resolve a few-point difference; its market score is far below 0.750 and does not beat either serious baseline.

The first title parser accepted only “Prix,” missing the Emilia Romagna question phrased “Gran Premio”; it initially reported 13 races and 0.4872. A direct metadata check found the omission. The parser now accepts both Grand Prix and Gran Premio wording and the **14-race result above supersedes the initial calculation**. An independent check against the older `data/cache_P.pkl` reproduced market and standings hits, but that cache's ensemble differed by one Hungarian podium hit from the current frozen `run.py` implementation. E147 therefore recomputes standings, logit, ensemble, and shuffled-label null directly from the current `build_features` and `walk_forward` code on strictly earlier races; the old cache was not used for the reported baseline. The Manifold source-time test injected bets created after cutoff and bets filled after cutoff and verified both leave the quote unchanged, while an earlier bet changes it. Saved quote timestamps all precede their cutoffs. Standard no-leak, pre-weekend, profile-leak, guard, and metric tests passed. The prospective 2026 E137/E146 captures remain separate and pending.

**Verdict: do not add this historical Manifold selector to the frozen model.** It may contain information above a shuffled null, but this fixed 2024 sample gives no evidence of a strict-overlap gain over standings or `ens`, much less 0.750. No 2026 held-out outcome was used.

### E148 — Registered prospective Manifold podium-market cohort for 2026 rounds 20–23 (C1; pending)

**Strict podium overlap: pending future official results.** Before looking at current 2026 Manifold probabilities, fix all four remaining rounds: São Paulo (round 20), Las Vegas (21), Qatar (22), Abu Dhabi (23). Use the same Thursday **00:00 UTC** cutoffs as the existing Kalshi captures: **2026-11-05, 2026-11-19, 2026-11-26, 2026-12-03**, respectively. Each is before structured FP1. Search the union of fixed public terms `2026 Formula 1`, `2026 Grand Prix podium`, and `2026 F1 podium` (newest, up to 1,000 each); require a question naming 2026, the specific race, and podium, independent driver answers, and creation before cutoff. Use the round-19 official roster to map answer names, fixed before these races. Reconstruct each answer's last pre-cutoff filled, non-cancelled normal-bet `probAfter` using the same E147 fill-time exclusions; require three mapped quoted drivers. If several questions qualify, select by pre-cutoff eligible amount, earliest creation, lexical ID. Save one immutable full quote table or explicit unavailable record per race after cutoff and before FP1. Do not inspect or use current displayed answer probabilities, final volumes, descriptions, comments, target-weekend sessions, or outcomes.

Score the fixed cohort after official results arrive, with all four rounds in the denominator and unavailable status reported. Strict podium overlap comes first, then same-race frozen standings and `ens`, shuffled-label null, and paired race bootstrap. A four-race sample alone cannot establish a 0.750 generalization claim. E147's 14-race 2024 negative result is development evidence and is not a reason to modify this prospective selection rule.

`experiments/c1_capture_manifold_2026.py` implements the fixed four-round source lookup, round-19 roster mapping, pre-cutoff normal-bet reconstruction, explicit unavailable status, immutable JSON output, and refusal to run before cutoff or at/after FP1. Running it before the São Paulo cutoff correctly raised `ValueError` without writing a prospective quote. `tests.test_manifold_2026_capture` verified all four time windows, exclusion of a post-cutoff bet, ranking, saved timestamps, and overwrite refusal with a synthetic market. `tests.test_manifold_leak` and the standard no-leak, pre-weekend, profile-leak, guard, and metric tests passed. **No 2026 Manifold quote or result has been inspected or frozen yet; coverage and strict overlap remain pending.**

The four existing one-time local Codex market-capture automations were updated to run this Manifold capture alongside Kalshi at **00:05 UTC** on each race's fixed cutoff date. Their saved active prompts, round numbers, local recurrence rules, and project target were inspected after the update. They preserve existing outputs, report unavailable sources, run both source and standard guards, and commit/push the captures. This is scheduling evidence only; it does not imply a capture has happened or that a future quote is available.

### E149 — Registered probability diagnostic for pre-weekend Kalshi quotes (C1; pending)

**Strict podium overlap first: unchanged E132 market-only score of 0.7222 (39/54) versus same-race standings 0.6852 (37/54) on 18 available 2025 races; no new selector is proposed.** E132 measured top-three overlap but did not compare quoted podium probabilities with the frozen logit's probabilities. Fix the same 18 E132 races and their saved Thursday-cutoff quote table. Match valid quoted contracts to actual 2025 entrants by `round,code`; do not use any later market value. Compute per-driver Brier and clipped log-loss for the raw bid/ask midpoint, the unchanged frozen pre-weekend logit (`C=0.03`, 100-race half-life), and a label-shuffled-logit null, on **identical quoted entrant rows**. Also report per-race macro means, race-paired intervals, quote coverage, market probability sum per race, and full-race strict overlap for market, standings, frozen `ens`, and null. The standings comparator remains the full-field pre-event ranking; its uncalibrated points are not a probability. Use the current `run.py` `build_features` path and strictly earlier training races, no target grid/session input or grid tie-break.

This is a repeatedly inspected 2025 diagnostic with no inner 2020–22 source coverage. Even if market probabilities score well, do not tune a blend or promote it to a 0.750 claim; the registered 2026 prospective market cohort is the next actual test.

**Result (outer-era 2025 subset; inner 2020–22 unavailable):** The 18 pre-registered E132 races contain 359 official entrants; **357** have valid saved cutoff quotes. Colapinto has no quoted contract in rounds 7 and 8. The first calibration-script assertion incorrectly assumed every entrant had a quote and failed; restricting Brier/log-loss to the registered identical quoted-entrant subset fixed that analysis bug without changing any ranking or market value. The market had an extra quoted contract in one race, but it was not an official entrant and was excluded from driver-label scoring. Its raw quoted-entrant probability sums average **3.222** per race (range 2.320–3.565), so these midpoints are not a normalized three-podium distribution.

| strict podium overlap, same 18 complete races | correct slots / 54 | mean |
|---|---:|---:|
| Kalshi market-only | 39 | **0.7222** |
| plain standings | 37 | 0.6852 |
| frozen `ens`, recomputed from current `run.py` path | 36 | 0.6667 |
| label-shuffled logit null | 9 | 0.1667 |

Market minus standings is +0.0370 (race-paired 95% CI [0,+0.0926]); market minus `ens` is +0.0556 ([0,+0.1111]). On the **same 357 quoted entrant rows**, raw-midpoint Brier is **0.06930**, frozen logit **0.07111**, and shuffled-logit null **0.12937**. The market-minus-logit race-mean Brier difference is **-0.00174** (95% CI [-0.00828,+0.00435]). Log-loss is **0.23623** market, **0.23623** frozen logit, and **0.43061** null (driver-row means); market-minus-logit race-mean difference **+0.00018** ([-0.01558,+0.01550]). An independent calculation from saved quote CSV and official result rows reproduced **39** correct market slots, **357** matched quoted entrants, and market Brier **0.06930238**. The unchanged Kalshi source-time leak test, pre-weekend scramble, guard, and metric tests passed. No 2026 race label or target-weekend input was read.

**Verdict: suggestive market information, no calibrated or strict-overlap gain established at the 0.750 gate.** The market's small Brier advantage is uncertain, its log-loss ties the frozen logit, and the strict score remains below 0.750. This is another look at the already inspected 2025 subset; leave the frozen model and the prospective 2026 comparison unchanged.

### E150 — Registered same-circuit team tyre-degradation transfer gate (A3/C3; pending)

**Strict podium overlap: pending measurement gate; no new selector yet.** E145's global team-relative degradation forecast failed in the 2023–25 outer period, but it did not ask whether a tyre-management signature repeats at the same circuit. Reuse the exact 1,701 team-race 2018–25 slopes and lineage mapping from E145; no re-extraction, parameter search, target-weekend input, or new data source. Attach the official circuit ID to each prior race. For each 2020–25 target team-race, compute E145's strictly earlier global forecast `g`. For that team at that circuit, average all strictly earlier slopes with weight `0.5 ** (target_season - source_season - 1)`; combine the weighted sum with **three fixed global-forecast pseudo-observations**: `(sum(w * same_circuit_slope) + 3*g)/(sum(w)+3)`. If no prior same-circuit source exists, this equals `g`. Do not condition on target qualifying, weather, tyres, stint, or measured degradation.

Compare same-circuit forecast with global E145 and zero on the identical covered team-races by MSE and race-paired bootstrap, reporting inner 2020–22 and outer 2023–25 separately, per-year MSE, source coverage, forecast/outcome correlation, and a within-race shuffled-outcome null. Proceed to a podium feature only if the candidate beats **both** global and zero in both periods and the outer candidate-minus-global race-paired 95% interval excludes zero. Otherwise stop at this gate and retain the frozen `ens`. The circuit signature is a historical feature; the target race's slope remains a retrospective label only.

**Result:** The fixed forecast covered 597 team-races in 60 inner races and 697 team-races in 70 outer races. A strictly earlier same-circuit team slope existed for **507/597 inner** and **687/697 outer** target rows; missing cases fell back exactly to E145's global forecast. Per-year circuit/global/zero MSE was 2020 **0.001639/0.001544/0.001625**, 2021 **0.001810/0.001804/0.001914**, 2022 **0.001605/0.001547/0.001582**, 2023 **0.000786/0.000729/0.000721**, 2024 **0.000634/0.000629/0.000617**, and 2025 **0.001005/0.001008/0.001025**. Only 2025 favoured the circuit forecast over global, by a very small amount.

| pre-weekend forecast of target team degradation | inner 2020–22 | outer 2023–25 |
|---|---:|---:|
| same-circuit + global MSE | **0.0016861** | **0.0008091** |
| E145 global MSE | 0.0016360 | 0.0007904 |
| zero MSE | 0.0017098 | 0.0007897 |
| circuit minus global, race-paired 95% CI | **+0.0000509 [+0.0000040,+0.0000987]** | **+0.0000188 [-0.0000157,+0.0000560]** |
| circuit/outcome correlation | 0.155 | 0.100 |
| global/outcome correlation | 0.190 | 0.029 |

The candidate-minus-zero difference was -0.0000232 (race-paired 95% CI [-0.0000968,+0.0000456]) inner and +0.0000194 ([-0.0000203,+0.0000600]) outer. Within-race shuffled-outcome correlation nulls were near zero, with one-sided p=0.0002 inner and 0.0064 outer for the circuit forecast; this detects some signal, not incremental MSE value over E145. An independently written vectorized past-only join reproduced both period sample sizes and circuit/global MSEs to floating-point precision. `tests.test_circuit_degradation_leak` scrambled target-and-later source slopes on synthetic and saved data without changing the target forecast, then changed an earlier same-circuit slope and observed a forecast response. All standard no-leak, pre-weekend, profile-leak, guard, and metric tests passed. Exact circuit IDs may hide layout changes, and the slope still mixes tyres, traffic, setup, and stint choice; no 2026 outcome or target-weekend source was used.

**Verdict: stop before a podium model.** The registered gate fails in both periods: same-circuit history **worsens** MSE versus the global forecast, and the outer candidate-versus-zero MSE is also worse. The outer correlation increase does not rescue the prespecified MSE gate. E150 has **no new strict podium score**; full-window references remain standings **0.6066 inner / 0.5905 outer**, frozen `ens` **0.6284 / 0.5857**, and shuffled-label logit null **0.1749 / 0.1857**. No evidence toward 0.750 comes from this circuit-specific tyre statistic.

### E151 — Structured absolute-compound source gate and prospective-market scope update

**Strict podium overlap:** no new selector or score. The unchanged full-window references remain plain standings **0.6066 inner / 0.5905 outer**, frozen `ens` **0.6284 / 0.5857**, and shuffled-label logit null **0.1749 / 0.1857**. No 2026 outcome was read. E119 and the C3 source recheck had already found that the cached timing feeds label tyres only relative to each event. A further source check inspected [OpenF1's published stint schema](https://openf1.org/docs/), which exposes `SOFT/MEDIUM/HARD`, and the [Pirelli developer portal](https://developer.pirelli.com/), which did not establish a public historical F1 nomination endpoint with pre-weekend timestamps. Search results also pointed to Pirelli/FIA preview documents, which the user's **no outside documents** instruction excludes as model inputs. This is a verification gap, not proof that no structured absolute C1–C5 archive exists. Without provenance and a pre-weekend availability time, C3 cannot supply an admissible 2020–25 feature or a paired podium test.

The user then asked to **steer away from the Kalshi API**. All four existing one-time local 2026 market automations were immediately changed to run **only** the registered E148 Manifold capture at their unchanged Thursday cutoffs, and their saved active prompts and recurrence rules were inspected. No future Kalshi API calls are planned. E132/E149's already saved historical results remain in this log for reproducibility, but E135/E137/E146 Kalshi prospective captures are superseded and will not be run. Manifold's prospective cohort remains pending; the frozen podium model and strict 0.750 gate are unchanged.

**Verdict:** C3 remains source-blocked under the permitted data rules; continue with other structured, past-only avenues. This scope change stops future Kalshi acquisition, not the broader podium-accuracy goal.

### E152 — Registered clean adjacent-car passing conversion gate (B2; pending)

**Strict podium overlap: pending measurement gate, no new selector yet.** B2 asks whether a repeatable driver skill converts close, pace-relevant positions into on-track passes. Reuse the offline 2018–25 FastF1 **race-lap timing cache** and official driver/team identifiers, with no target-weekend input to a forecast. For each completed race, at lap `t` identify a trailing car at positions **P2–P8** and the immediately preceding car at P1–P7, both at the same lap number, with cumulative finish-line time gap **0–2 seconds**. Require both cars' laps `t` and `t+1` to have green `TrackStatus == "1"`, accurate timing, no pit in/out, and non-missing positions/times. A conversion is the trailing car ahead of that same car at the end of `t+1`. Count only one opportunity per trailing car/lap; this deliberately excludes ambiguous pit and caution reversals but cannot prove a pass was driver-caused. Record all eligible opportunities, including failures, with source season/round/lap, both driver codes, gap, and conversion. Use no target-race lap row as an input to its own forecast.

First report race/driver coverage, total opportunities and conversions, per-season counts, and the share of observed reversals under these filters. Proceed to a time-ordered driver-rate forecast only if 2020–25 has at least **10,000 opportunities, 150 conversions, eight drivers with at least ten conversions**, and outer 2023–25 has at least **75 conversions**. If it passes, forecast each target opportunity's conversion probability from that trailing driver's strictly earlier races using a fixed **50-race half-life** and **200 global-rate pseudo-opportunities**; compare to the same strictly earlier 50-race weighted global conversion rate on identical future opportunities. Report Brier/log-loss by inner 2020–22 and outer 2023–25, per-year results, race-paired bootstrap, and a within-race shuffled-driver null. Consider a podium feature only if the driver rate beats global Brier in both periods with an outer race-paired interval excluding zero; otherwise stop. The full-window strict-overlap references remain standings 0.6066/0.5905, frozen `ens` 0.6284/0.5857, and shuffled-label logit null 0.1749/0.1857, but a new strict score does not exist at this gate.

**Measurement result:** The offline extraction produced **18,148** eligible opportunities and **1,030** reversals across 171 races in 2018–25, a 5.68% conversion rate under this narrow definition. The 2020–25 development set has **14,467 opportunities, 838 conversions, 130 races, and 34 trailing drivers**; 16 drivers have at least ten conversions. The outer 2023–25 period has 477 conversions, clearing all four registered feasibility counts. Development season counts (opportunities/conversions) are 2020 **1,816/99**, 2021 **2,147/101**, 2022 **2,061/161**, 2023 **2,758/197**, 2024 **2,879/137**, 2025 **2,806/143**. One 2018 race was unavailable in the offline cache, and 2021 Belgium had no eligible green adjacent-car comparison; neither is backfilled from a later source.

| past-only probability of a clean close-car pass | inner 2020–22, 6,024 opportunities / 60 races | outer 2023–25, 8,443 / 70 |
|---|---:|---:|
| driver-specific Brier | **0.056236** | **0.053034** |
| global Brier | 0.056370 | 0.053334 |
| driver minus global, race-paired 95% CI | -0.000196 [-0.000418,+0.000011] | **-0.000392 [-0.000689,-0.000103]** |
| driver-specific log-loss | 0.225900 | 0.214874 |
| global log-loss | 0.227104 | 0.217470 |
| driver minus global log-loss, race-paired 95% CI | -0.001712 [-0.003659,+0.000110] | **-0.003145 [-0.005306,-0.001072]** |

Driver-minus-global Brier by year was -0.000045 in 2020, +0.000064 in 2021, -0.000420 in 2022, -0.000937 in 2023, +0.000018 in 2024, and +0.000002 in 2025. The outer pooled gain is driven mostly by 2023; the within-race shuffled-driver null had mean Brier difference +0.000303 outer versus observed negative, one-sided p=0.0005, while inner null mean was +0.000079 with p=0.0005. This indicates matched driver identity carries some information under the fixed filters, not a stable or causal passing skill. An independent streaming decay calculation reproduced every past-only driver/global forecast to within 7×10⁻¹⁷. The source has no duplicate race/lap/trailing keys. `tests.test_clean_passing` checked a synthetic pass and excluded gap, pit, and caution cases; its saved-source test scrambled target-and-later conversions without changing a target forecast and changed an earlier label to verify sensitivity.

The standard no-leak, pre-weekend, profile-leak, guard, and metric tests also passed after the new source was saved. All 18,148 extracted opportunities are from completed 2018–25 races; no 2026 target-weekend data was read.

**Verdict: the registered measurement gate passes, narrowly.** Driver history beats global Brier in both periods and the outer paired interval excludes zero. The small and uneven gain justifies **one** fixed pre-weekend podium feature test, not a claim toward 0.750. The event label can still include unrecorded slowdowns, damage, or timing-chart corrections; no target race's pass label may enter its own feature. E152 has no new strict podium score.

### E153 — Registered one-feature passing-history podium test (B2; pending)

**Strict podium overlap: pending.** Use E152's saved source unchanged. Before each target race, compute every entered driver's 50-race-half-life conversion probability from **strictly earlier** clean opportunities, shrunk by 200 pseudo-opportunities toward the strictly earlier global rate. Drivers without past opportunities receive that global rate. Join this one `prior_pass` feature by official season/round/driver code to the unchanged `run.py` pre-weekend feature table; no target-weekend lap, grid, practice, qualifying, sprint, or race label is an input. Fit the same frozen podium logit (`C=0.03`, 100-race training half-life) with `prior_pass` appended once. Compare its standalone ranking and an equal-rank average with plain standings against the frozen logit, frozen `ens`, plain standings, and label-shuffled-logit null, all walk-forward over 2020–25 with `tie=None`.

Report inner 2020–22 and outer 2023–25 strict podium overlap first, same-race paired bootstraps, changed top-three sets, per-season results, and standalone logit Brier/log-loss on identical driver-races. Do not tune shrinkage, weighting, contender filters, or model type on those outcomes. This already inspected outer period is development evidence even if it clears 0.750; a model promotion would still need a separately frozen prospective test. No 2026 outcome may be used in this experiment.

**Result — strict podium overlap first:**

| pre-weekend walk-forward model | inner 2020–22 (61 races) | outer 2023–25 (70 races) |
|---|---:|---:|
| plain standings | 0.6066 | **0.5905** |
| frozen `ens` | 0.6284 | 0.5857 |
| `ens_pass` (one added history feature) | **0.6339** | **0.5810** |
| frozen logit alone | 0.6393 | 0.5571 |
| pass-feature logit alone | 0.6393 | 0.5524 |
| label-shuffled-logit null | 0.1749 | 0.1857 |

The augmented ensemble added **one** correct podium slot inner (116/183 versus 115/183 for frozen `ens`) and lost **one** outer (122/210 versus 123/210); plain standings had 124/210 outer. It changed only **1/61 inner** and **2/70 outer** top-three sets. Outer `ens_pass` minus standings was **-0.0095** (race-paired 95% CI [-0.0429,+0.0238]); minus frozen `ens` was **-0.0048** ([-0.0143,0]). Inner minus standings was +0.0273 ([0,+0.0546]) and minus frozen `ens` +0.0055 ([0,+0.0164]); these are one-slot movements, not reliable evidence of a general lift. By season, augmented ensemble strict overlap was 2020 **0.6863**, 2021 **0.6667**, 2022 **0.5606**, 2023 **0.5303**, 2024 **0.5556**, and 2025 **0.6528**. Frozen `ens` differed only in 2020 (0.6667) and 2024 (0.5694).

The standalone frozen/augmented logit Brier was **0.077192/0.077145 inner** and **0.083136/0.083226 outer**; log-loss was **0.249676/0.249867 inner** and **0.262932/0.263118 outer**. The augmented-minus-frozen race-paired Brier difference was -0.000047 (95% CI [-0.000314,+0.000211]) inner and +0.000086 ([-0.000627,+0.000788]) outer. Log-loss differences were +0.000192 ([-0.000723,+0.001091]) inner and +0.000176 ([-0.001630,+0.001970]) outer. Thus the small pass-conversion probability signal from E152 does not improve podium calibration either.

The new `prior_pass` table agrees **exactly** with E152's independent opportunity-level forecasts on all 18,148 matched historical opportunities; every feature is finite and no 2026 feature row was computed. `tests.test_passing_podium_leak` scrambled target-and-later raw pass outcomes in synthetic and saved source tables without changing the target feature, then changed an earlier outcome and verified a response. The E152 source filter test and standard no-leak, pre-weekend, profile-leak, guard, and metric tests passed. The original feature set, model regularization, standings vote, and pre-weekend `tie=None` path were preserved; no target-weekend input or 2026 label entered this comparison.

**Verdict: do not promote the passing-history feature.** Its outer strict overlap is below both standings and the frozen ensemble, and its probability changes are too small to justify a variant search on these inspected seasons. The 0.750 gate remains unachieved. Per the user's request, pause the research goal after logging and pushing this experiment; do not start E154 until the user resumes it.

### E154 — Reset-year in-season pace blend, early rounds (R1 in HANDOFF.md; registered before running)

**Hypothesis:** after a regulation reset the pecking order is still being discovered, so decayed in-season race pace carries more weight than cumulative standings in the first rounds (E8: 2026 accuracy rose R1-5 .40 → R12+ .63 while 2023–25 stayed flat; E86: testing pace matters most after resets). **Only 2022 is a reset season inside the dev window; 2026 is restricted and will not be inspected.** Hence this is a diagnostic with ~10 races, far below the E60 detection threshold, and cannot promote anything alone.

**Fixed variants (no tuning):** `pace3` = rank of `-(0.7*t3_pace_gap + 0.3*d3_pace_gap)` (missing → worst); `blend` = rank-average(standings, pace3); `ens_pace` = rank-average(standings, logit, pace3). Compared with standings and frozen `ens` on rounds 1–10 and 11+ separately, per season 2020–25. Placebo: the same rule on non-reset seasons (2020, 2021, 2023–25). Pre-weekend path (`tie=None`, guard on), history-only columns, no 2026 rows.
**Gate:** a reset-year rule is credible only if (a) 2022 R1–10 shows `blend` or `ens_pace` > standings by a paired CI excluding zero or by ≥0.05 with the same sign in a majority of placebo-contrast seasons being NOT positive (i.e. the effect is specific to the reset), and (b) the rule is stated before any 2026 look. Otherwise stop R1.

**Result (E154):** a first run scored `pace3` ≈ 0.00 podium everywhere because my script passed the pace gap with the wrong sign (this harness ranks LOWER score as better). The label-sanity pattern (a model scoring ~0 rather than ~0.17) exposed it; fixed in `experiments/r1_reset_blend.py` and rerun. Strict podium overlap on rounds 1–10 (10 races/season), standings / `ens` / `blend` / `ens_pace`: 2020 .667/.700/.700/.733; 2021 .567/.633/.600/.600; **2022 .500/.500/.533/.500**; 2023 .667/.633/.567/.633; 2024 .633/.633/.600/.633; 2025 .667/.667/.700/.700. 2022 R1–10 `blend` minus standings = **+0.033** (CI [0.000,+0.100]); `ens_pace` 0.000 ([-0.100,+0.067]). The non-reset pooled R1–10 placebo gave `blend` -0.007 ([-0.047,+0.033]) and `ens_pace` +0.020 ([-0.020,+0.060]); 2020 and 2021 show the same size of lift as 2022, so it is not specific to the reset. Rounds 11+ show no consistent pattern.
**Verdict: gate failed; stop R1 in dev data.** One extra correct slot in 10 races is inside noise and not reset-specific. This does not test 2026 (restricted), where E8 still shows a rising curve; no 2026 look was taken.

### E155 — Does a team's execution residual persist? (R2 persistence gate, ad hoc diagnostic; stop rule fixed in HANDOFF.md before running: stop if persistence < 0.15)

**Measure:** per team-race, mean over its cars of (true race-pace rank − official finishing order), so positive = finished better than pace rank (strategy, safety-car, start and attrition luck, E93's 39% class). Data: 2019–25 `y_pace`/`order` from the feature table; no podium model was built and no 2026 row was read. This is a past-only persistence test, not a new selector.

| persistence test | corr | n |
|---|---:|---:|
| team-race residual vs mean of its previous 10 races (≥5 required) | **+0.082** (Spearman +0.073) | 1,428 |
| per-season (2019–25) | +0.12, -0.02, +0.15, +0.15, -0.02, +0.09, +0.05 | 159–239 |
| half-season team means, raw | +0.389 | 70 |
| cross-season team means, raw | +0.305 | 55 |
| half-season team means, **strength-adjusted** | **+0.074** | 70 |

The raw half- and cross-season correlations are a **confound**, not skill: the residual correlates +0.58 with the team's mean pace rank (fast cars systematically finish below their clean-lap pace rank, slow cars above it), so any team-strength persistence shows through. Regressing the residual on pace rank in each half leaves +0.074. The race-level 0.082 is below the pre-stated 0.15 stop line.
**Verdict: stop R2.** There is no repeatable team-level execution edge beyond team strength, consistent with E93 ("mostly luck near the front") and E9/E11. No feature was built; no leak test needed.

### E156 — Frozen rounds 1–3 pre-season-testing blend (HANDOFF §6 item 1; registered before running)

**Rule (fixed, no tuning):** for rounds 1–3 only, score = 0.75 × rank(`ens`) + 0.25 × testing strength, testing strength = mean of within-race percentile ranks of the team's mean testing best-lap gap and long-run gap (missing → 0.5). Rounds 4+ use `ens` unchanged. Weight 0.25 comes from E87, not retuned. Testing laps are pre-season sessions (before round 1) and the driver→team map is the entry list, so no target-weekend data enter. Coverage: only seasons with testing files (reported per season; races without testing data fall back to `ens`). Pre-weekend path (`tie=None`, guard on). Compare to standings and `ens` on identical (season, round); paired bootstrap; full-season effect reported as well as rounds 1–3. No 2026 rows. **Gate to adopt as the final candidate:** rounds 1–3 gain over `ens` with the same sign in a majority of covered seasons AND full-season effect not negative; otherwise record as unsupported.

**Result (E156):** testing files cover 2022–25 (2026 excluded); 12 covered races (rounds 1–3 × 4 seasons), 2020–21 uncovered and fall back to `ens`. Rounds 1–3 strict overlap: `ens` **0.611**, blend **0.667**, standings **0.667**. By season (ens/blend/standings): 2022 .444/.556/.556; 2023 .667/.667/.778; 2024 .667/.667/.667; 2025 .667/.778/.667. Blend minus `ens` on covered races **+0.056** (CI [0.000,+0.139]), blend minus standings **0.000** ([-0.083,+0.083]). Full-season: inner `ens` .6284 → blend .6339 (standings .6066); outer `ens` .5857 → blend .5905 = standings .5905; outer blend minus `ens` +0.005 ([0.000,+0.014]), blend minus standings 0.000 ([-0.033,+0.033]).
**Verdict: do not adopt as a gain claim.** The blend repairs `ens`'s weak early rounds (positive in 2 of 4 seasons, never negative) but does so by landing on the standings number, not above it; the gate's "majority of seasons positive" fails (2 of 4) and the sample is 12 races. It costs nothing and is directionally consistent with E86/E87, so it remains the default early-round rule for the final 2026 candidate, but it is worth ~0.5 points full-season and cannot approach 0.750. 2026 untouched.

### E157 — R5 source-feasibility gate: archived pre-weekend expert/tipster podium picks (user lifted the no-outside-documents rule on 2026-11-07)

**Strict podium overlap:** no new selector or score. References unchanged: standings 0.6066 inner / 0.5905 outer, frozen `ens` 0.6284 / 0.5857. No 2026 row read; no model fit. This is a coverage, timestamp and leak-risk audit of candidate sources, performed with web search and page fetches before any extraction code.

**What exists.** Per-race prediction series are published by Sportskeeda ("Race Predictions" / "Top 5 Bold Predictions"), Sportskhabri ("Preview and Predictions"), Racing Post ("betting tips and F1 predictions", with bet365 odds), Betfred, and CBS/SportsLine (model picks), plus occasional Sky Sports/Autosport panels (e.g. Autosport's 2025 triple-header finale). No single outlet covered all 24 races in the probes; Sportskeeda/Sportskhabri slugs differ per race, so URLs are not enumerable by pattern; Sportskeeda blocked direct fetches (HTTP 405), Sportskhabri redirects to a new domain, and archive.org's availability endpoint rate-limited (HTTP 429) on two requests. Many pieces are generic ("Verstappen favourite", "McLaren double podium"), i.e. close to the standings ranking the model already has.

**Timestamp/leak audit (the decisive finding).** The one dated page fetched end to end, Racing Post's 2024 Hungarian GP tips, shows "Published 20 July 2024, last updated 17:28" - **Saturday, after qualifying** - and its text quotes the target weekend's qualifying result (Norris pole, Piastri +0.022 s, Pérez 16th) and free-practice long-run gaps. Under the Thursday 00:00 UTC cutoff that page is inadmissible. Betting-tip content is normally written after practice or qualifying, so most tipster archives would fail the cutoff for most races and would need per-article date verification; archive-capture timestamps are not guaranteed to equal publication time either, and pages re-edited afterwards (as here) are a leak path.

**Not done.** No extraction was run and no prediction was scored; one page is an illustration, not a coverage estimate. A proper audit would need a fixed outlet list, an enumerated URL list for 2023–25, and a published-before-cutoff count per race before any outcome is looked at.

**Verdict: R5(a) is feasible only as a small, leak-audited subset, with weak expected information.** Expected admissible pieces are few, redundant with standings, and require LLM extraction with the memory-contamination controls in HANDOFF §6 item 2c. Do not scrape broadly yet. The higher-information, cleaner R5 sources are **timestamped bookmaker/exchange odds snapshots taken before Thursday 00:00 UTC** (E132 showed market prices as the only pre-weekend source above 0.70) and **dated structured announcements**; both require an account, purchase or dataset the user must supply (Betfair historical data account per E118/E132; the Kalshi API is off-limits per E151).

### Scope update (2026-11-07, user): no betting sites, markets or tipsters
The user ruled out **all betting involvement** (bookmaker/exchange odds, prediction markets, betting-site tips) and asked that the work stay purely analytical/ML. This closes E157's market and tipster options, and the already-scheduled Manifold captures (E148) are no longer a candidate input. No experiment was run; the 0.750 gate and the frozen model are unchanged. Remaining outside-document candidates (X1–X6 in RESEARCH_AVENUES.md: FIA component-usage and penalty notices, car-presentation submissions) are non-betting and structured, and remain the only R5 route within this constraint.

### E158 — FIA-notice gate, part 1: oracle value of knowing grid penalties in advance (registered before running; analytical diagnostic)

**Why:** the FIA documents that exist before the cutoff (previous event's cumulative "PU elements used per driver", prior penalties) can at best forecast *grid penalties* (E157 audit below). If **perfect advance knowledge of each driver's true grid penalty** does not lift pre-weekend strict podium overlap, no FIA-derived PU/penalty feature can, and the route closes without scraping anything. **Oracle, not a model:** `orc_pen = max(grid - qualifying position, 0)` from the target weekend (uses post-hoc information, like E22), added to the frozen history feature set of the logit and of the `ens` standings+logit rank average; 2020–25, inner/outer, paired against `ens`, plus the count of races where any driver ranked top-6 by `ens` carries a penalty ≥3 places. No 2026 rows.

**FIA document audit (E157 follow-up, fia.com decision-document pages for the 2024 season):** every event has an enumerable document list ("Doc N - Title", "Published on dd.mm.yy hh:mm CET"), so exact cutoff enforcement is possible, and the PDFs follow a predictable address pattern (`<year> <GP> Grand Prix - PU elements used per driver up to now.pdf`, `... - New PU elements for this Competition.pdf`). Admissible under the Thursday 00:00 UTC cutoff is only the **previous** event's cumulative PU report and earlier penalties; "Car Presentation Submissions" (Abu Dhabi 2024: Friday 06.12 07:57 CET), "New PU elements for this Competition" (Friday 11:32 CET, after FP1 started) and the provisional/final grids are published during the target weekend and are inadmissible. No scraper was written.

**Result (E158, oracle):** perfect advance knowledge of each driver's true grid penalty (`grid - qualifying position`, 38 of 131 dev races have a top-6-ranked driver penalised ≥3 places):

| strict podium overlap | inner 2020–22 | outer 2023–25 |
|---|---:|---:|
| `ens` | 0.6284 | 0.5857 |
| `ens` with oracle penalty as logit feature | 0.6339 | 0.5857 |
| logit / logit + oracle penalty | 0.6393 / 0.6230 | 0.5571 / 0.5476 |
| `ens` with oracle penalty demotion (1/20 rank per lost place, fixed) | 0.6120 | 0.5714 |

Paired outer `ens_pen` minus `ens` = **0.000** (95% CI [-0.014,+0.014]); inner +0.005 ([-0.011,+0.022]); outer demotion rule -0.014 ([-0.033,+0.005]). So even **perfect knowledge of grid penalties** does not improve pre-weekend podium selection, and the real FIA-derived signal could only forecast penalties with error.
**Verdict: FIA PU/penalty route closed.** No scraping or feature build is justified. Remaining FIA items (car-presentation specification changes, technical directives) are unstructured, published on the weekend, and have no structured historical measure of performance effect; they are not testable as an analytical feature under the cutoff.

### E159 — FINAL pre-weekend 2026 look (single, registered before running; user approved 2026-11-07)

**Candidate, frozen before the run:** `ens` = rank-avg(standings, logit C=0.03 hl=100) on history-only columns, with the E156 rule for rounds 1–3 only (0.75 × rank(`ens`) + 0.25 × testing strength from 2026 pre-season testing, missing → 0.5). Rounds 4+ = `ens` unchanged. Train on all strictly earlier races from 2019 (same as E8). Data: committed `data/results.csv` (2026 rounds 1–19, as in E8; no refresh). Prior 2026 exposure: E6, E8, E86, E93 (and the E8 number 0.544 is known). Reported on all 19 races: strict overlap for candidate, `ens`, standings, shuffled-label null; paired bootstrap candidate vs `ens` and vs standings; secondaries (winner, ≥2 of 3, top-5 cover); and podium Brier/log-loss for the logit vs a standings-rank prior (re-scope target (b)). Post-qualifying 2026 (re-scope target (a)) is **not** re-run: E6's 0.649 vs grid 0.614 (+0.035, CI [0.000,+0.088]) stands. Nothing is tuned after this run.

**Result (E159, final; 19 races, 2026 R1–19; 11 teams had 2026 testing data):**

| 2026 pre-weekend | candidate | `ens` | standings | shuffled null |
|---|---:|---:|---:|---:|
| **strict podium overlap** | **0.561** | 0.544 | 0.526 | 0.228 |
| winner | 0.368 | 0.368 | 0.368 | 0.053 |
| ndcg@3 | 0.596 | 0.579 | 0.559 | 0.194 |
| exact podium set | 0.053 | 0.053 | 0.053 | 0.053 |
| top-5 cover / top-10 / pairwise | .663 / .779 / .773 | .663 / .779 / .771 | .663 / .758 / .755 | .379 / .563 / .583 |

Candidate minus `ens` +0.018 (95% CI [0.000,+0.053]); candidate minus standings +0.035 ([0.000,+0.088]); `ens` minus standings +0.018 ([-0.035,+0.070]). ≥2-of-3 in 63.2% of races; races at ≥0.75: **1 of 19**. Per round: .33 .67 .67 .33 .33 .67 .67 .67 .67 .33 .33 .67 .67 .33 .33 .67 .67 .67 1.0.
Re-scope (b), podium probabilities on 2026: logit Brier **0.08081** / log-loss **0.23882** vs standings-rank prior 0.08331 / 0.35543; race-paired Brier difference -0.0025 (CI [-0.0077,+0.0029]): log-loss clearly better, Brier not resolvable at 19 races.
**Verdict: 75% strict podium overlap is NOT reached (final: 0.561 pre-weekend; 0.649 post-qualifying from E6).** The frozen candidate is the deliverable. No further 2026 looks; no tuning. Research on the 0.750 pre-weekend goal is closed with this entry.

# POST-QUALIFYING PROGRAMME (HANDOFF §11; user confirmed 2026-11-07). Setting = E0's original: grid/qualifying of the target race known, race not started. Pre-weekend 75% is closed (E159). 2026 restricted again until a post-quali candidate is frozen.

### E160 — P1: post-qualifying ensembles (registered before running)
**Fixed variants:** rank-avg(grid, compact_logit), rank-avg(grid, standings, compact_logit), rank-avg(grid, standings), each equal weight, vs `grid` and the frozen E5 `compact_logit` (C=0.3, hl=100, COMPACT features). Dev 2020–25, inner 2020–22 / outer 2023–25, tie-break = grid (post-quali convention). **Gate:** outer gain over `grid` AND over `compact_logit` with a paired CI excluding zero, or ≥+0.03 with the same sign in both splits, to become the P1 base.

**Result (E160):** strict podium overlap, inner 2020–22 / outer 2023–25: `compact_logit` **0.6831 / 0.6905**, `grid` 0.6721 / 0.6857, ens(grid, logit) 0.6721 / 0.6810, ens(grid, standings, logit) 0.6940 / 0.6667, ens(grid, standings) 0.6831 / 0.6667, standings 0.6120 / 0.5905. By season (logit / grid): 2020 .686/.686, 2021 .697/.697, 2022 .667/.636, 2023 .667/.636, 2024 .667/.667, **2025 .736/.750**. Outer paired: ens_gsl − logit **−0.024** [−0.052, 0.000]; ens_gl − grid −0.005 [−0.024,+0.014]; logit − grid +0.005 [−0.038,+0.048].
**Verdict: gate failed; no post-quali ensemble beats the frozen logit.** Adding standings to a grid-aware model *hurts* out of sample (standings is weaker than grid and the logit already contains `d_season_pts`). The E5 `compact_logit` stays the P1 base. Also confirmed: grid alone reaches 0.750 in 2025, 0.636–0.697 elsewhere; the season, not the model, dominates.

### E161 — P2 measurement gate: corrected practice long-run race pace vs qualifying as a predictor of TRUE race pace (registered before running)
**Measurement (fixed):** per FP session (FP2, FP3; FP1 only on sprint weekends), green-flag accurate laps, no in/out laps, within stints of ≥4 consecutive kept laps, first lap of each stint dropped. Per-session ridge (alpha 10) on log lap time with driver indicators, compound indicators, tyre age (per compound, /10) and lap-index-in-stint (fuel/track evolution proxy); the driver coefficient is that driver's corrected long-run pace; sessions averaged per weekend (FP2 weight 2, FP3 weight 1 where both exist). Comparators: driver median of the same long-run laps (E4-style), session best lap, and qualifying gap (`q_gap`). **Gate:** per-race Spearman with TRUE race pace (`y_pace`), averaged over races: the corrected FP pace must beat `q_gap` on the available inner seasons (2020 cached; 2021–22 being downloaded) with the same sign on 2023/2025; otherwise the practice route closes. No podium model is fitted in this entry.

**Result (E161; cached seasons 2020/2023/2025, 16/21/24 races; 2021/22/24 practice sessions downloading):** mean per-race Spearman with TRUE race pace: `q_gap` 0.669 / 0.552 / 0.626; **corrected FP long-run pace 0.288 / 0.029 / 0.143**; long-run median 0.291 / 0.155 / 0.231; FP best lap 0.734 / 0.506 / 0.613. FP2-only long runs: 0.312 / 0.123 / 0.102. **Gate failed decisively:** corrected long-run practice pace is a far worse race-pace measurement than qualifying, in every season, with or without compound/age/fuel-index correction; the unobserved fuel load and programme differences dominate (E4's conclusion, now confirmed with the full E111-style correction). The practice *long-run* route is closed.
Side finding (not pre-registered, reported for completeness): rank-average(`q_gap`, FP best lap) beats `q_gap` alone as a race-pace predictor in all three seasons: 0.710 / 0.613 / 0.638 vs 0.639 / 0.563 / 0.588 (FP2-only best: 0.759 / 0.626 / 0.664). FP best is a second low-fuel single-lap observation that de-noises qualifying (a bad Q lap, traffic, a deleted lap). E4 found no podium gain from FP gaps on 2023–25 with OpenF1 data; E162 re-tests this one feature post-quali with the FastF1 measure.

### E162 — P2 follow-up: FP best-lap gap as one post-qualifying feature (registered before running)
**Fixed:** `fp_best` = weekend average (FP2 weight 2, FP3 weight 1, FP1 on sprint weekends) of each driver's session best-lap gap (%) from `data/parts/fp_pace_*.csv`; missing → race median. Added as one feature to the frozen E5 `compact_logit` (C=0.3, hl=100); also a fixed rank-avg(compact_logit, fp_best). Post-quali walk-forward on seasons with FP coverage (currently 2020, 2023, 2025; rerun when 2021/22/24 arrive). Train rows without `fp_best` get the median. Comparators: `grid`, `compact_logit`. **Gate:** paired gain over `compact_logit` on 2023+2025 with CI excluding zero, same sign in 2020.

**Interim result (E162; 2020/2023/2025, 63 races; rerun pending 2021/22/24 download):** post-quali strict overlap by season, `compact_logit` / `logit_fp` / `grid`: 2020 .686/.686/.686; 2023 .667/.652/.636; 2025 .736/**.764**/.750. Pooled: logit 0.6984, logit+fp 0.7037, grid 0.6931, fp_best alone 0.545, rank-avg(logit, fp) 0.6772. Paired logit_fp − compact_logit **+0.005** [−0.011,+0.021]; 2023+2025 only +0.007 [−0.014,+0.029]. **Gate not met (CI includes zero; sign flips in 2023).** The measurement gain (+0.06 Spearman) does not reach podium selection, as E10's mapping predicts. Held open only for the 2021/22/24 rerun; no tuning.

### E163 — P5: retirement-aware post-qualifying re-selection (registered before running)
**Fixed:** `p_fin` from a logistic on past-only reliability features (`d10_dnf`, `t10_dnf`, `d3_dnf`, `t3_dnf`, `t10_order`, `gridn`), trained on strictly earlier races with the same recency weights; score = −P(podium | compact_logit) × P(finish). Also the fixed variant using P(finish) only on drivers ranked ≤6 by the logit. Comparators `grid`, `compact_logit`; dev 2020–25 inner/outer. Gate: outer paired gain over `compact_logit` with CI excluding zero. (E18's pre-weekend retirement model changed no picks; this is the post-quali version.)

### E164 — P3: sprint-race result as a post-qualifying feature on sprint weekends (registered before running)
**Fixed:** on weekends with a sprint (`s_order` present in the feature table), add `s_order` (sprint finishing position; missing → race median) and `s_gain = s_grid − s_order` to the frozen compact logit; evaluate on sprint weekends only (2021–25) and pooled with all races. Sprint runs before the Grand Prix, so it is admissible post-qualifying. Comparators `grid`, `compact_logit`. Gate: paired gain on sprint races with CI excluding zero, ≥20 sprint races.

**Result (E163):** inner / outer strict overlap: `compact_logit` 0.6831 / 0.6905; logit × P(finish) 0.6831 / 0.6857; top-6-only variant identical. Outer paired −0.005 [−0.014, 0.000]. **Verdict: no gain; multiplying by predicted reliability changes one outer pick, for the worse.** Matches E18 (pre-weekend). Retirements of podium-pace cars are not forecastable from past DNF rates at the precision podium selection needs. Closed.

**Result (E164, 24 sprint weekends 2021–25):** `compact_logit` 0.6389, grid 0.6111, logit + sprint result 0.6111; paired −0.028 [−0.083,+0.028]. All 2021–25 races: 0.6871 / 0.6784 / 0.6813. **Verdict: the sprint result does not help and slightly hurts;** sprint weekends are also harder than average (0.64 vs 0.69). Closed. (P4, grid × circuit overtaking difficulty, was already in COMPACT as `grid_x_corr`/`circ_grid_corr`; not re-run.)

### E165 — Post-qualifying oracle ladder: how much headroom does each true fact add on top of grid + compact logit? (registered before running; diagnostic, uses post-race truth)
**Fixed:** to the frozen `compact_logit` add, one at a time, (a) TRUE target-race pace `y_pace` as a feature, (b) TRUE retirers (demote DNFs to the back), (c) TRUE team finishing order (teams ranked by best car's finish, drivers within team by the logit), (d) a+b together. Dev 2020–25, inner/outer. Purpose: E10/E22 measured these pre-weekend; the post-quali headroom has never been measured, and it bounds P2 (any practice-based pace improvement) and P5 (reliability) in this setting.

**Result (E165, post-qualifying oracle ladder, strict overlap inner / outer):** grid 0.672 / 0.686; `compact_logit` 0.683 / 0.691; + TRUE retirers 0.760 / **0.724** (+3.3 outer); + TRUE race pace 0.716 / **0.738** (true pace alone 0.760 / 0.738, the E10 cap again); + TRUE pace **and** TRUE retirers 0.760 / **0.771**; TRUE team finishing order 0.754 / **0.776**.
**Reading:** post-qualifying, 0.75 is crossed only when *both* race pace and retirements are known exactly; either alone caps at 0.72–0.74. Qualifying measures pace at rho ≈ 0.55–0.67 (E161) and the E10 mapping is steep, so the pace half needs near-perfect race-trim knowledge that practice does not provide (E161); the retirement half has no forecastable component at selection precision (E163, E18). The headroom exists (+8 pts) but sits entirely in information that is not available before the race starts.

### Priority note (2026-11-07, user): the pre-weekend setting is the headline goal; post-qualifying (E160–E165) is secondary.

### E166 — Pre-weekend combined oracle: true race pace AND true retirers together (registered before running; diagnostic, post-race truth)
**Why:** E22 measured pre-weekend oracles one at a time (pace 0.740, retirers +4.0, team order 0.743); E165 showed post-quali crosses 0.75 only with pace and retirers *together* (0.771). The combined pre-weekend ceiling is unmeasured and bounds the whole pre-weekend route. **Fixed:** frozen `ens`; add (a) true pace alone, (b) true retirers alone (demote), (c) both, (d) true team finishing order with within-team order from `ens`; pre-weekend path, dev 2020–25 inner/outer.

**Result (E166, pre-weekend, strict overlap inner / outer):** `ens` 0.628 / 0.586; + TRUE retirers 0.689 / **0.610** (+2.4 outer); + TRUE race pace 0.743 / **0.738**; + TRUE team finishing order 0.743 / **0.743**; + TRUE pace **and** TRUE retirers 0.771 / **0.757**.
**Reading:** 0.75 pre-weekend *is* inside the information space, but only at the very top of it: it needs essentially exact race-day pace for every driver and exact knowledge of who retires. Either alone is below 0.75. The pre-weekend pace forecast sits at rho ≈ 0.72 against a hindsight persistent-form cap of ≈ 0.775 (E13), and the E10 mapping gives 0.66 even at rho 0.96; retirements add at most +2.4 pts even when perfectly known. So the pre-weekend route to 0.75 requires predicting race-specific pace deviations (car×track r = 0.04, E10; weather oracle no gain, E16) plus retirements, neither of which any permitted or now-permitted non-betting source has been shown to forecast.
**E161 addendum (inner seasons, after the 2021/2022 practice download; 2024 only 5 races so far):** per-race Spearman with TRUE race pace, `q_gap` / corrected FP long-run / long-run median / FP best: 2021 0.724 / 0.189 / 0.246 / 0.715; 2022 0.643 / 0.175 / 0.344 / 0.675; 2024 (5) 0.731 / 0.250 / 0.229 / 0.667. Same pattern on inner: the long-run route stays closed; FP best tracks qualifying.

### E167 — Retirement causes from Wikipedia race classifications → cause-specific pre-weekend hazard (registered before running; the one pre-weekend information gap documents can fill; hard bound +2.4 pts from E166)
**Source:** `https://en.wikipedia.org/wiki/<year>_<Grand Prix name>` race-classification table, "Time/Retired" column, 2018–2025; non-betting, retrospective, used only as *past* labels (a driver's cause history before race K). Mapping (fixed): mechanical = {engine, power unit, hydraulics, gearbox, electrical, electronics, brakes, suspension (no contact), oil/water/fuel leak, overheating, turbo, MGU, exhaust, transmission, battery, driveshaft, wheel, puncture-without-contact is NOT mechanical}; collision/accident/spun off/damage = incident; everything else = other. Coverage reported per season versus Jolpica DNF counts. **Test:** rerun E101's cause-specific hazard (exposure-based, 10-race half-life) with these labels; score = frozen logit × P(no mechanical retirement); gates: (1) the hazard's outer Brier on mechanical events must beat a smoothed team baseline, (2) paired outer strict overlap vs `ens`. Expected ≤ +2 pts; promote only if (2) excludes zero.

**Result (E167):** `experiments/wiki_retirements.py` parsed all 172 race classifications 2018–25 (`data/parts/wiki_retirements_2018_2025.csv`); matched to Jolpica rows by normalised surname. Named mechanical retirements per season, now: 2019 25, 2020 27, 2021 27, 2022 39, **2023 26, 2024 17, 2025 16** (Jolpica had 0 for 2023–25); unknown-cause rows fall to 0–14 per season. Rerunning E101's pipeline unchanged with these labels (`experiments/e167_wiki_hazard.py`): outer mechanical events 59 of 1,366 eligible driver-races; cause-specific hazard Brier **0.04177 / log-loss 0.18145 vs generic-DNF hazard 0.04164 / 0.18064** (inner 0.07143 vs 0.07155). Strict podium overlap: logit × cause risk = logit × generic risk = logit = 0.6393 inner / 0.5571 outer, i.e. **identical selections**; standings 0.5905, `ens` 0.5857.
**Verdict: gate (1) fails; route closed.** With the label gap filled, cause-specific mechanical history still carries no information beyond a team's generic DNF rate, and neither changes a single podium pick. This closes the last document-fillable pre-weekend information gap identified in HANDOFF §12. The Wikipedia cause table is kept as a data asset.

**Final result (E162, all six seasons 2020–25, 131 races; FP coverage 94–100% per season):** post-quali strict overlap, `compact_logit` / `logit_fp` / `grid`: 2020 .686/.686/.686; 2021 .697/.697/.697; 2022 .667/.667/.636; 2023 .667/.667/.636; 2024 .667/.667/.667; 2025 .736/**.750**/.750. Pooled 0.687 / **0.690** / 0.679; rank-avg(logit, fp best) 0.636; FP best alone 0.519. Paired `logit_fp` − `compact_logit` **+0.0025 [0.000,+0.008]** (outer +0.0048 [0.000,+0.014]); rank-average variant −0.051 [−0.087,−0.018].
**Verdict: gate not met.** The feature changes one pick in 131 races (2025), is never negative, and is far below the ~4.5-pt detectability threshold; adding practice pace by averaging ranks hurts significantly. Practice pace is closed as a podium input, pre-weekend (E4, E161) and post-qualifying (E162). The `logit_fp` variant is not promoted.

### E168 — Dedicated pre-weekend winner model (registered before running; one experiment, then pause)

**New goal:** maximize exact winner hit rate, accepting lower second/third-place accuracy. Strict podium overlap remains a reported secondary metric. No target-weekend inputs, outside documents, betting sources, or Kalshi API.

**Fixed test:** use the existing pre-weekend feature set and chronological 2019+ training schedule. Compare one logistic classifier trained on `win` (C=0.03, 100-race half-life, matching the frozen podium logit) with plain standings and frozen `ens` = equal rank-average(standings, podium logit). Also train a same-specification null classifier with one winner label randomly assigned per past race (fixed seed, stable across forecasts). Predict one exact winner per race by lowest score, with driver-name tie-break, in inner 2020–22 and outer 2023–25. Report winner counts/rates first, strict podium overlap, per-season winner counts, and race-paired bootstrap intervals versus both references. Do not inspect or score 2026. **Gate:** promote only if winner accuracy beats both standings and `ens` on the outer split with paired 95% intervals excluding zero, and the inner effect is nonnegative. The outer seasons have been repeatedly inspected, so even a passed gate would be provisional. If the gate fails, retain the frozen predictor. Pause research after logging and pushing this batch, as requested.

**Result (E168):** exact winner hits, with strict podium overlap in parentheses:

| model | inner 2020–22, 61 races | outer 2023–25, 70 races |
|---|---:|---:|
| dedicated `win_logit` | **32/61 = 0.525** (0.639) | **32/70 = 0.457** (0.562) |
| plain standings | 29/61 = 0.475 (0.607) | **32/70 = 0.457** (0.590) |
| frozen `ens` | 30/61 = 0.492 (0.628) | **32/70 = 0.457** (0.586) |
| podium logit alone | 32/61 = 0.525 (0.639) | **32/70 = 0.457** (0.557) |
| shuffled-winner null | 5/61 = 0.082 (0.137) | 4/70 = 0.057 (0.110) |

Dedicated-model winner hits by season 2020–25: **11, 8, 13, 18, 8, 6**; standings: **9, 8, 12, 19, 9, 4**. Race-paired `win_logit` minus standings: inner **+0.049** (95% CI [-0.066,+0.164]), outer **0.000** ([-0.086,+0.071]); versus `ens`: inner **+0.033** ([-0.049,+0.115]), outer **0.000** ([-0.057,+0.057]). The outer sample cannot resolve modest gains, but there is no observed gain to promote. The independent top-score/true-winner count check matched the harness exactly. Standard no-leak, pre-weekend scramble, profile-leak, guard, and metric tests passed; no new feature or source was introduced. Only 2020–25 were scored; 2026 was not evaluated.

**Verdict: winner gate failed.** Training directly on winners did not improve outer exact-winner accuracy over standings or the frozen ensemble, while it reduced strict podium overlap. Keep the frozen predictor. Stop after E168 and pause the research goal per the user's instruction; do not start E169 without a resume request.

### E169 — Race-conditional winner likelihood (registered before running)

**Scope:** user resumed research on 2026-11-16. Exact top-1 winner accuracy is primary; strict podium overlap remains secondary. Two competing explanations for E168: (a) its independent binary labels dilute the winner decision across many non-contenders; (b) the available history lacks the information needed to distinguish contenders. Test (a) by fitting one conditional softmax choice per historical race, winner-only, rather than E30's top-three Plackett–Luce likelihood. No new data or features.

**Fixed protocol:** same history-only columns, median imputation/scaling fitted on strictly earlier training rows, 2019+ training, 100-race half-life. Minimize sum of recency-weighted per-race winner negative log-probabilities plus `||beta||² / (2 * 0.03)` (C matches E168; no intercept because it cancels within a race). Test one candidate, `choice_win`, against standings, frozen `ens`, and E168 `win_logit`. Include a conditional model trained on one randomly assigned winner per historical race (fixed seed 169, stable chronological assignment). Train/predict/reveal chronologically for inner 2020–22 and outer 2023–25 only. Report winner counts, per-season counts, strict overlap, winner log-loss for normalized winner probabilities, and paired winner intervals. Gate: positive outer gain versus both standings and `ens` with 95% paired intervals excluding zero, and nonnegative inner effects. No hyperparameter search or 2026 evaluation. Verify objective gradient against finite differences, uniform zero-coefficient probabilities, target-label independence, and independent winner counts before any promotion.

**Result (E169):** winner hits `choice_win` **32/61 = 0.5246 inner / 32/70 = 0.4571 outer**, identical hit/miss sequences to E168's binary winner model (two predicted names differ, both wrong). Standings **29/61 / 32/70**; frozen `ens` **30/61 / 32/70**. Paired winner differences versus standings **+0.0492 [-0.0656,+0.1639] inner / 0.0000 [-0.0857,+0.0714] outer**; versus `ens` **+0.0328 [-0.0492,+0.1148] / 0.0000 [-0.0571,+0.0571]**. Conditional null **6/61 / 1/70**. Candidate strict podium overlap **0.6393 / 0.5714**; null **0.1475 / 0.1190**. Candidate winner log-loss **1.5556 / 1.4622**, versus normalized E168 binary winner probabilities **1.5648 / 1.4539**, normalized podium logit **1.7237 / 1.6613**, conditional null **3.1904 / 3.1810**. Choice-model winner hits by season: **11, 8, 13, 18, 8, 6**.

Finite-difference gradient, zero-coefficient loss, uniform prediction, target-outcome/grid mutation, probability-sum, convergence and independent winner-count assertions passed. Results are saved in `results/e169_per_race.csv` and `results/e169_predictions.csv`; no 2026 evaluation. **Verdict: gate failed.** Race-conditional likelihood does not unlock additional exact winner hits. A post-hoc diagnostic of these saved picks found the choice model selected the actual winning **team** in **37/61 inner / 44/70 outer**, versus exact driver **32/61 / 32/70**; this motivates E170, not a new accuracy claim.

### E170 — Team-win probability × conditional teammate winner (registered before running)

**Hypothesis:** fitting team strength and within-team winner choice separately may improve the 12 outer errors where E169 picked the winning team but the wrong driver. This differs from E28, which forecast team finishing position for podium selection; E170 trains directly on winning-team and winner-among-winning-team labels.

**Fixed:** one binary team-win logistic on existing team history columns (`t3_*`, `t10_*`, `t_season_pts`, `t_circ_res`), aggregated by team and historical race. One driver logistic fitted only to members of historical winning teams, using existing driver history columns (`d3_*`, `d10_*`, `d_season_pts`, `prev_order`, `d_circ_res`) centered within the current team. Both C=0.03, half-life 100, strictly past 2019+ training, median imputation/scaling fitted only on their training rows. Normalize team-win probabilities across teams and driver probabilities within each team; multiply for the joint winner probability. Null: assign one random winner per past race (seed 170) before forming both training labels. Same inner/outer seasons and paired winner gate as E169; compare saved, freshly rebuilt E169 standings/`ens`/winner references on exactly matched races. No feature/source changes to production, target labels or weekend data used only for scoring, no 2026 evaluation. Independently verify winner counts and target-label invariance. No retuning after results.

**Result (E170):** team/driver winner model **31/61 = 0.5082 inner / 30/70 = 0.4286 outer**; standings **29/61 / 32/70**, frozen `ens` **30/61 / 32/70**. Inner paired winner gain versus standings **+0.0328 [-0.0656,+0.1311]**, versus `ens` **+0.0164 [-0.0328,+0.0656]**; outer differences versus either **-0.0286 [-0.1000,+0.0429]**. Candidate winner hits by season **11, 8, 12, 18, 7, 5**; null **5/61 / 1/70**. Strict overlap **0.6230 / 0.5238**, versus standings **0.6066 / 0.5905**, `ens` **0.6284 / 0.5857**, null **0.2623 / 0.0619**. Candidate selected the actual winning team **36/61 / 41/70**, versus E169 **37/61 / 44/70**. Target-label/grid mutation, symmetric-input uniform distribution, normalization and independent winner-count checks passed; no 2026 evaluation. **Verdict: gate failed.** Separating car and driver decisions degraded team selection and exact winner accuracy. Do not promote; do not tune the two stages on outer results.

### E171 — Explicit past winner propensity (registered before running)

**Hypothesis:** the existing `PAST_COLS` averages finish/points/qualifying/pace, but never averages wins. Winners may carry additional information beyond a good average finish. Add four features: driver and team exponentially weighted historical win rates, half-lives 3 and 10 races. Use ONLY strictly prior races; team label is any of its drivers winning that past race. Three pseudo-observations shrink driver rates toward 0.05 and team rates toward 0.10; downweight older regulation eras by the existing 0.3 driver / 0.1 team factors. A missing history takes that prior. No outside data.

**Fixed candidate:** E168's binary winner logistic plus those four columns, C=0.03 / half-life 100. Two untrained diagnostics rank directly by driver win rate (3- or 10-race half-life), not eligible for post-hoc selection or promotion. Null: same augmented logistic with one randomized winner label per training race (E168 seed 168). Inner 2020–22 / outer 2023–25 only, baseline predictions from this batch's rebuilt E169 artifacts on identical races. Gate: beat both standings and `ens` on outer exact winner hit rate with paired 95% intervals excluding zero, nonnegative inner effects; show comparison with E168 too. Report strict podium overlap secondarily. Source-specific target-and-later scramble, earlier-label sensitivity, uniform-prior and no-history checks required before scoring. No 2026 evaluation or tuning.

**Result (E171):** augmented winner logistic **30/61 = 0.4918 inner / 32/70 = 0.4571 outer**. Paired versus standings **+0.0164 [-0.0820,+0.1148] inner / 0.0000 [-0.0714,+0.0714] outer**; versus `ens` **0.0000 [-0.0984,+0.0984] / 0.0000 [-0.0714,+0.0714]**; versus E168 **-0.0328 [-0.0984,+0.0328] / 0.0000 [-0.0571,+0.0571]**. Candidate strict overlap **0.6393 / 0.5667**. Per-season winner hits **11, 6, 13, 19, 8, 5**. Direct 3-race/10-race win-rate diagnostics **33/61 and 31/61 inner / 33/70 each outer**, with overlap **0.5574 and 0.6120 inner / 0.5095 and 0.5571 outer**. These diagnostics were not candidates for selection; one outer extra hit is not established improvement.

The single seeded augmented null hit **13/61 inner / 18/70 outer**, much higher than E168's 5/61 and 4/70; its overlap remained **0.1803 / 0.1762**. This does not establish leakage or a usable winner model, but makes a multi-seed null audit necessary before relying on one random baseline. Source-scramble, earlier-label responsiveness, no-history prior, independent winner counts, and standard no-leak/pre-weekend/profile/guard/metric checks passed. No 2026 evaluation. **Verdict: gate failed; no promotion.** Explicit winner history did not add exact winner hits to the logistic on outer data. E172 will audit the null spread as well as a genuinely different nonlinear winner learner.

### E172 — Nonlinear winner classifiers and seeded-null audit (registered before running)

**Why:** E168–E171 tested linear winner classifiers and two-stage linear factorization. Old GBM experiments optimized podium labels. Test existing fixed GBM settings directly on `win`, without tuning: 250 trees, learning rate .03, 8 leaves, minimum child samples 30, L2=5, subsample .8, feature subsample .7, seed 0, recency half-life 100. Two declared variants: existing base pre-weekend features; base plus E171's four win-history columns. No model is selected on outer performance; report both and make no promotion if any required comparison fails. Include one shuffled-winner GBM on augmented features (seed 168), with a random winner in every strictly earlier training race.

**Null audit:** separately run E171's augmented logistic with random-winner seeds 0–19 (all declared in advance; original seed 168 retained as an additional reference). Report inner/outer distribution of exact winner rates and how many null seeds meet/exceed the observed augmented logistic. This audits sensitivity of the null baseline; it is not another optimized candidate or a formal proof of independence. Expose a seed argument in E168's existing null helper without changing its default, so historical runs are unchanged.

Same walk-forward train/predict/reveal, 2019+ training, inner 2020–22 / outer 2023–25, references from this batch, exact winner first and strict overlap secondary. Same promotion gate as E169, separately for each variant; no target-weekend inputs, new external data, 2026 evaluation, or parameter retuning. Audit target-label invariance and independent winner counts.

**Result (E172):** base GBM winner **33/61 = 0.5410 inner / 29/70 = 0.4143 outer**; augmented GBM **32/61 = 0.5246 / 33/70 = 0.4714**; standings **29/61 / 32/70**, frozen `ens` **30/61 / 32/70**. Base GBM outer paired difference versus standings **-0.0429 [-0.1429,+0.0429]**, versus `ens` **-0.0429 [-0.1143,+0.0286]**. Augmented outer difference versus standings **+0.0143 [-0.0718,+0.1143]**; inner differences versus standings **+0.0492 [-0.0492,+0.1475]**, versus `ens` **+0.0328 [-0.0656,+0.1311]**. Base/augmented strict overlap **0.6066/0.5847 inner / 0.5571/0.5762 outer**. Base winner hits by season **12,8,13,17,7,5**; augmented **12,8,12,19,8,6**. GBM null **6/61 / 2/70**, overlap **0.1585 / 0.0952**. Independent count, target-label/grid mutation, and default-null-seed compatibility checks passed.

Twenty-seed augmented-logit null audit: inner mean winner rate **0.0951**, range **0.0000–0.3607**; outer mean **0.1064**, range **0.0000–0.3429**. None of the 20 seeds met the actual augmented logistic's **0.4918 / 0.4571**. Seed 168's **0.2131 / 0.2571** lies within this broad spread. This is a sensitivity diagnostic, not a formal significance claim or proof of no leakage. Winner-label null scores need not be near 1/20 in one fitted run; continue reporting the distribution rather than relying on a favorable seed. Source timing tests supply the leakage evidence. Saved per-race predictions/metrics and `results/e172_null_audit.csv`; no 2026 evaluation. **Verdict: neither nonlinear variant passes the gate.** One extra outer winner for the augmented GBM is exploratory and does not justify promotion.

### E173 — Faster adaptation of the winner classifier (registered before running)

**Hypothesis:** E168's 100-race training half-life retains too much of earlier dominant-team eras for winner selection in a changing field. Test exactly one shorter half-life, **20 races**, on the existing base pre-weekend features and C=0.03 binary winner logistic. This changes training recency only; feature decay remains 3/10 races. No hyperparameter search. A same-20-race null uses randomized historical winners (seed 168); compare against saved E169 standings, frozen `ens`, and original 100-race winner model. Extend the existing winner helper with an `hl=100` default, preserving historical behavior. Inner 2020–22 / outer 2023–25, strictly earlier 2019+ training, exact winner primary, strict overlap secondary, independent winner-count audit; same paired gate as E169. No 2026 evaluation or retuning after results.

**Result (E173):** shorter-half-life winner model **32/61 = 0.5246 inner / 33/70 = 0.4714 outer**, versus standings **29/61 / 32/70**, `ens` **30/61 / 32/70**, 100-race winner model **32/61 / 32/70**. Paired winner gain versus standings **+0.0492 [-0.0656,+0.1639] inner / +0.0143 [-0.0571,+0.0857] outer**; versus `ens` **+0.0328 [-0.0492,+0.1148] / +0.0143 [-0.0571,+0.0857]**; versus original winner model **0.0000 [0,0] / +0.0143 [-0.0286,+0.0714]**. Candidate overlap **0.6448 / 0.5714**; null winners **9/61 / 4/70**, overlap **0.2186 / 0.1048**. Candidate winner hits by season **11,8,13,18,8,7**. Independent winner counts and default-helper compatibility checks passed; no 2026 evaluation. **Verdict: gate failed.** One extra outer correct winner is not a demonstrated gain; retain frozen predictor.

### E174 — Winner error anatomy and hindsight information ladder (registered before running; diagnostic only)

**Question:** the podium error analysis does not identify the best route to the new winner objective. Audit frozen `ens`'s saved 2020–25 predictions, keeping 2026 excluded. Decompose each wrong winner into wrong team versus correct team/wrong teammate. Measure fixed hindsight selectors: (a) perfect winning-team knowledge, driver within that team ordered by `ens`; (b) preserve `ens`'s chosen team but know its actual better finisher; (c) demote actual retirers in `ens`; (d) actual clean-lap race-pace order; (e) actual race pace plus actual retirers. Report inner/outer exact winner first and strict podium overlap second, pace coverage and actual-winner pace coverage. These selectors use target-race outcomes and are explicitly inadmissible predictors, not proof of a ceiling or deployable results. Keep the guarded pre-weekend walk-forward path unchanged. Compare each oracle with `ens` using race-paired winner differences as descriptive diagnostics. Assert that error categories are disjoint/exhaustive and independently check winner counts. Do not fit or promote a model from this analysis.

**Result (E174, hindsight only):**

| selector | inner exact winner | outer exact winner | strict overlap inner / outer |
|---|---:|---:|---:|
| honest frozen `ens` | 30/61 = 0.4918 | 32/70 = 0.4571 | 0.6284 / 0.5857 |
| **ORACLE** actual winning team, within-team driver from `ens` | 48/61 = 0.7869 | 54/70 = 0.7714 | 0.6885 / 0.5667 |
| **ORACLE** chosen team unchanged, actual better teammate | 37/61 = 0.6066 | 42/70 = 0.6000 | 0.6230 / 0.5857 |
| **ORACLE** actual retirers demoted | 32/61 = 0.5246 | 34/70 = 0.4857 | 0.6885 / 0.6095 |
| **ORACLE** actual clean-lap race pace | 44/61 = 0.7213 | 39/70 = 0.5571 | 0.7432 / 0.7381 |
| **ORACLE** actual pace plus actual retirers | 45/61 = 0.7377 | 39/70 = 0.5571 | 0.7705 / 0.7571 |

Frozen `ens` errors are **24 wrong-team + 7 wrong-teammate = 31 inner**, **28 wrong-team + 10 wrong-teammate = 38 outer**. Thus **73.7% of outer winner errors are wrong-team choices**. Pace is available for **1,134/1,220 inner entrants and 60/61 actual winners**, **1,329/1,398 outer entrants and all 70 actual winners**; missing pace ranked last. All selectors use the same full race set, not an easier complete-case subset. Outer race-paired winner gains over `ens`: winning-team oracle **+0.3143 [0.2143,0.4143]**; teammate oracle **+0.1429 [0.0714,0.2286]**; retirers **+0.0286 [0,0.0714]**; pace and combined pace/retirers each **+0.1000 [-0.0429,+0.2429]**. These are descriptive differences for hindsight rules, not tests of forecast skill.

**Bug caught and fixed:** pandas' default CSV parser altered last-bit scores on reload and changed one tied podium ordering. The `ens`-reproduction assertion failed before diagnostic reporting. Using `float_precision="round_trip"` restores the exact saved scores and all baseline winner/podium metrics; E170's reference-prediction loader uses the same fix. No winner counts or fitted model changed. The final diagnostic asserts disjoint/exhaustive error categories, reproduces frozen metrics, and independently verifies actual winner selection counts.

**Verdict:** prioritize forecasting the winning team and contender execution, with transferable teammate skill secondary. The median clean-lap pace proxy alone is substantially weaker for exact winners than for podium overlap in the outer competitive seasons. This does not prove that the true fastest car cannot be forecast or win; it says this particular hindsight pace measure is insufficient. No oracle is a deployable model, and the 0.7714 winning-team number is not an achieved pre-weekend result. Winner research remains active per user instruction; no model promoted and no 2026 evaluation.

**Batch verification (E169–E174):** `tests.test_winner_artifacts` independently recomputes exact-winner and strict-podium hits from every saved model ranking in E169–E173, preserving floating-point score ties on CSV reload; every per-race result matches. E170 was regenerated after the loader fix with unchanged reported aggregate results. Choice-gradient/target-label, two-stage target-label, real-source win-history scramble, GBM target-label, standard no-leak/pre-weekend/profile/guard/metric checks, syntax compilation, and `git diff --check` passed. The prior untracked Wikipedia extract was left untouched and was not used. All reported scores are development evidence after repeated looks; no small gain is claimed as established success.

### E175 — Portable teammate pace for exact winner selection (registered before running)

**Previous turn:** progress: E169–E174 produced verified negative model results and a winner-specific error decomposition; no running process to wait on. **Hypothesis:** E116's past-only matched teammate pace graph carried real contender probability information for podiums. It may discriminate the winner better than ordinary driver averages. Use the existing source and unchanged `feature_table` (10-race decay, ridge anchor 50); no re-extraction or tuning.

**Two fixed candidates:** (1) E168's C=.03, 100-race-half-life binary win logistic plus `prior_portable_gap`; (2) frozen `ens` chooses the team as before, but its winner is replaced by that chosen team's driver with the lowest observed prior graph gap (missing graph history → retain original winner; driver-name tie-break). Candidate (2) changes only which driver is promoted to P1 and preserves remaining order. No outer-driven choice between candidates. Same inner 2020–22 / outer 2023–25 chronological 2019+ training and pre-weekend guard; freshly rebuild standings, podium logit and `ens`, and compare E168 winner baseline. Include a one-winner-per-race label-shuffle null for the augmented logistic (seed 168, with the broad null sensitivity already documented in E172). Winner first, strict overlap secondary; per-season counts, paired intervals against standings/`ens` and winner baseline. Candidate gate unchanged: positive outer gain over both standings and `ens` with paired intervals excluding zero, nonnegative inner effects. No 2026 evaluation, outside documents, market inputs, or Kalshi calls. Check graph target/future-source independence, winner-swap target-label invariance, missing-history behavior and independently reconstruct saved metrics.

**Result (E175):** augmented winner logistic **31/61 = 0.5082 inner / 33/70 = 0.4714 outer**; graph winner swap **32/61 = 0.5246 / 33/70 = 0.4714**; standings **29/61 / 32/70**, frozen `ens` **30/61 / 32/70**, E168 winner **32/61 / 32/70**. Both candidates' outer winner hits by season **19,8,6**; augmented inner **11,7,13**, swap inner **11,9,12**. Augmented inner paired gain vs standings **+0.0328 [-0.0656,+0.1311]**, vs `ens` **+0.0164 [-0.0656,+0.0984]**; outer vs standings **+0.0143 [-0.0714,+0.1000]**, vs `ens` **+0.0143 [-0.0429,+0.0714]**. Swap inner vs standings **+0.0492 [-0.0492,+0.1475]**, vs `ens` **+0.0328 [0,+0.0820]**; outer vs standings **+0.0143 [-0.0714,+0.1000]**, vs `ens` **+0.0143 [-0.0286,+0.0714]**. Augmented/swap strict overlap **0.6339/0.6284 inner / 0.5762/0.5857 outer**. Null winners **4/61 / 4/70**, overlap **0.1311 / 0.1095**. Graph coverage **98.77% / 98.86%** of entrant rows. Source-scramble and swap target-label/missing-history/tie tests passed. **Verdict: both gates fail.** The portable signal yields one extra outer winner, unresolved after repeated development looks. No promotion or 2026 evaluation.

### E176 — Pre-season testing for early-round exact winners (registered before running)

**Hypothesis:** testing pace was the only measured new early-season team-order signal (E86). The frozen E156 blend optimized podium overlap; assess fixed winner-specific uses without retuning the testing weight. Use only the existing pre-season best/long-run summaries in `testing_2022_2023.csv` and `testing_2024_2025.csv` (exclude 2026 and other similarly named mileage/medium files). Mean by season/driver code, then current-entry-list team mean; testing strength is the average within-race percentile of team best-gap and long-run-gap (missing → .5). Sources are the existing pre-season sessions, not the target weekend.

**Two fixed candidates, rounds 1–3 only:** (1) `.75 * rank(E168 winner logistic) + .25 * testing strength`, weight inherited from E156, not tuned on winner outcomes; (2) choose the lowest-testing-strength team, within-team order from the same winner logistic. Rounds 4+, or seasons with no testing observations, use the unmodified winner logistic. Same C=.03/hl100, earlier-2019+ walk-forward, inner 2020–22 / outer 2023–25. Report both candidates on all races, covered rounds 1–3 and individual covered seasons, exact winner first/strict overlap second. Rebuild standings/`ens` references, and a shuffled-winner-logit null (seed 168) with the same .25 testing blend. Candidate gate: beat standings and `ens` on full outer exact-winner accuracy with paired 95% intervals excluding zero, nonnegative inner effects, and improvement over the base winner logistic in a majority of covered seasons; otherwise no promotion. No outer-driven model selection or 2026 evaluation. Test target-weekend-label and future-season-source independence, sensitivity to actual pre-season inputs, and unmodified fallback after round 3.

**Result (E176):** both testing candidates **32/61 = 0.5246 inner / 32/70 = 0.4571 outer**, exactly the same winner hit/miss sequences as the base winner logistic. Standings **29/61 / 32/70**, frozen `ens` **30/61 / 32/70**. Full outer paired differences versus standings **0 [-0.0857,+0.0714]**, versus `ens` **0 [-0.0571,+0.0571]**, versus base winner model **0 [0,0]**. Full inner differences vs standings **+0.0492 [-0.0656,+0.1639]**, vs `ens` **+0.0328 [-0.0492,+0.1148]**. Covered early races: inner 2022 **0/3** winners for both candidates/base, standings/`ens` **1/3**; outer 2023–25 **3/9** for both/base, standings/`ens` **4/9**. Covered per-season candidate winner hits **0,1,2,0** for 2022–25; zero seasons improve over base. Blend/team-first strict overlap **0.6448/0.6448 inner / 0.5619/0.5571 outer**; on covered races **0.6667/0.6667 inner / 0.6296/0.5926 outer**. Null winners **5/61 / 5/70**, overlap **0.1366 / 0.1190**. Target-label/future-season-source, actual-preseason-input responsiveness, and exact score fallback after round 3 passed. **Verdict: both gates fail.** Testing improves no exact winner choices in this fixed formulation. Only 12 covered races; do not infer absence of smaller testing information or tune another weight on these outcomes. No promotion or 2026 evaluation.

### E177 — Team best-car history for winner selection (registered before running)

**Hypothesis:** the existing team history averages both drivers, which can understate a team's ability to win when its drivers differ sharply. E170's team classifier also used these means. Add best-car summaries from earlier races: minimum official finish, minimum qualifying gap and minimum clean-lap race-pace gap within each historical team/race. Average these team-race minima at fixed 3- and 10-race half-lives (six columns), with existing older-era team weight .1. Every source race must have `source.k < target.k`; missing observations omitted from that indicator's denominator, no history → missing/imputer. Sources are unchanged official results, historical qualifying and existing FastF1 pace; target-weekend values must not enter.

**One candidate:** base E168 winner logistic + six best-car history columns, C=.03 / training half-life 100. Compare freshly rebuilt standings, frozen `ens`, E168 winner logistic; same shuffled-winner null (seed 168) on augmented features. Same 2020–22 / 2023–25 walk-forward and paired winner gate as E175, plus report the winning-team hit count to see whether the feature addresses E174's main errors. No new external data, tuning or 2026 evaluation. Verify within-race min rather than mean on synthetic data, source target/future scramble, earlier-input responsiveness, and independent reconstruction of saved winner/podium rankings.

**Result (E177):** augmented best-car winner model **32/61 = 0.5246 inner / 31/70 = 0.4429 outer**, versus standings **29/61 / 32/70**, frozen `ens` **30/61 / 32/70**, E168 winner **32/61 / 32/70**. Inner paired differences versus standings **+0.0492 [-0.0656,+0.1639]**, versus `ens` **+0.0328 [-0.0492,+0.1148]**, versus winner baseline **0 [0,0]**; outer versus standings **-0.0143 [-0.0857,+0.0571]**, versus `ens` **-0.0143 [-0.0714,+0.0429]**, versus winner baseline **-0.0143 [-0.0429,0]**. Candidate exact winner hits by season **11,8,13,18,8,5**. Strict overlap **0.6393 / 0.5571**, null **0.1257 / 0.1143**; null exact winners **3/61 / 4/70**. Candidate winning-team hits **37/61 / 43/70**, compared with `ens` **37/61 / 42/70** and winner baseline **37/61 / 44/70**. The extra team hit versus `ens` did not improve exact winner choice, and the direct winner baseline lost one correct team and winner. Synthetic-min, real-source target/future scramble and past-input responsiveness checks passed; no 2026 evaluation. **Verdict: gate failed.** No promotion or further best-car-feature tuning on these inspected outer races.

**E175–E177 verification/versioning:** the independent saved-ranking audit reproduces every per-race winner/podium metric for these experiments; the no-leak, pre-weekend, profile, guard and metric checks passed. E175/E176 code, results and E177 registration were committed/pushed as `fabda3c`; E177 feature implementation and cutoff tests were separately checkpointed as `a9cd0a8`, before its outcome log. E177 artifacts and result are a separate follow-up commit. Research remains active; the unrelated untracked Wikipedia extract was not used or changed.

### E178 — Winner-specific regularization, selected only on inner races (registered before running)

**Previous turn:** progress, with E175–E177 results and code committed; no live experiment remains. **Known:** C=.03 was inherited from the podium model and used for all linear winner experiments. **Assumption to test:** stronger/looser regularization could better fit the rarer win label. Competing explanation is that the available features lack discriminating information, so a different C will not improve a winner forecast. This is one bounded calibration check, not an outer learner sweep.

**Frozen grid and selection:** C in **[.003, .03, .3, 3]**, base E168 history-only features and 100-race training half-life; median/scaler fitted strictly on earlier 2019+ rows. Walk-forward **inner 2020–22 only**, choose highest exact-winner count, tie → smallest C (strongest regularization). Save every inner variant's predictions/metrics and a locked selection JSON before any outer fit. Inner selected score is optimized development performance, not unbiased validation. The selector must reject outer rows; a test changes synthetic outer labels while inner selection stays identical.

If C=.03 wins, stop without a new outer run: it is the already evaluated baseline and no new candidate was found. Otherwise run **only that single locked C** on outer 2023–25, compare standings, frozen `ens`, E168 C=.03 winner and a matched selected-C shuffled-winner null (seed 168). Report winner first, strict overlap second, per-season counts and race-paired winner intervals. Promote only on positive outer gain versus standings and `ens` with intervals excluding zero and nonnegative inner effects; repeated outer inspection means even a passed gate remains provisional. No 2026 evaluation, new features/sources, or outer-driven retuning. Extend the winner helper with `C=.03` default so historical callers remain unchanged. Verify explicit default equivalence and independently reconstruct saved rankings; commit the registration/implementation before outcomes, then commit results separately.

**Result (E178):** all four C settings scored **32/61 = 0.5246 exact inner winners**. Inner strict overlap in grid order .003/.03/.3/3: **0.6503 / 0.6393 / 0.6120 / 0.5683**; these are optimized development comparisons. The registered strongest-regularization tie-break selected **C=.003**, saved in `results/e178_selection.json` before fitting outer. Inner winner counts by season for selected C **11,7,14**, versus base **11,8,13**; no aggregate winner improvement anywhere on the grid.

The single locked outer candidate scored **32/70 = 0.4571 winners**, standings/`ens`/base winner each **32/70**. Every paired winner difference was **0.0000**, CI **[-0.0714,+0.0714]** against all three references. Selected outer winner counts **18,8,6** (2023–25). Strict overlap selected **0.5714**, standings **0.5905**, `ens` **0.5857**, base **0.5619**. Selected-C null **6/70 = 0.0857** winners / **0.0952** overlap. The selector rejects outer rows; synthetic outer-label changes do not change an inner-only selection, tie selection and explicit default-C compatibility tests passed. No features or sources changed and no 2026 evaluation. **Verdict: gate failed.** Winner-specific regularization did not improve exact winner accuracy; do not tune further C values on these outer results. Registration/implementation checkpoint was `926e746`; results are committed separately.

### Goal clarification (user, 2026-11-16)

The user answered the success-threshold question with **75% exact winners across all races**. This refines the active winner-first objective; it does not permit target-weekend data, changing the denominator, counting the winner somewhere in a longer list, or substituting driver-wise binary accuracy. Strict podium overlap remains secondary. Current best observed winner variants remain 33/70 = 47.1% outer with unresolved gains versus standings; 75% is not achieved.

### E179 — Past laps-led share for exact winner selection (registered before running)

**Hypothesis:** laps spent leading can preserve evidence of winning form that a retirement, pit sequence or late race event erased from the official finish. E42 tried past led share for podium labels; the new winner objective has not tested it. Reuse only existing `start_2018_2021.csv`, `start_2022_2024.csv`, and the <=2025 rows of `start_2025_2026.csv`, joining by season/round/driver code. No extraction, outside documents or new API calls.

**Fixed transform:** four columns: driver and team past laps-led share averaged at half-lives 3 and 10 races. Team-race share = sum of its drivers' observed shares (all missing → missing); one team exposure per race. Use only strictly earlier source races, prior-era weights .3 driver / .1 team as in base features, missing indicator values omitted from its denominator, no history → missing. Include exactly these four columns in the base C=.03 / training-half-life-100 winner logistic; do not combine other post-hoc positive variants. Compare freshly rebuilt standings, `ens` and base winner; same augmented-feature random-winner null (seed 168). Same inner 2020–22 / outer 2023–25 walk-forward. Exact winner first, strict overlap secondary, per-season hits and race-paired differences; gate requires 75% winner rate plus positive paired gains versus standings/`ens` with intervals excluding zero and nonnegative inner effects before claiming success. A smaller observed gain is research evidence only. No 2026 evaluation or tuning. Verify target/future raw-source scramble, past responsiveness, team aggregation and missing-source behavior before scoring; independently reconstruct saved rankings. Commit registration/implementation and outcomes separately.

**Result (E179):** past-leading winner model **31/61 = 0.5082 inner / 34/70 = 0.4857 outer**, versus standings **29/61 / 32/70**, frozen `ens` **30/61 / 32/70**, base winner **32/61 / 32/70**. Inner paired winner differences vs standings **+0.0328 [-0.0492,+0.1148]**, vs `ens` **+0.0164 [-0.0984,+0.1311]**, vs base **-0.0164 [-0.0984,+0.0656]**. Outer differences vs standings **+0.0286 [-0.0286,+0.0857]**, vs `ens` **+0.0286 [-0.0429,+0.1000]**, vs base **+0.0286 [-0.0429,+0.1000]**. Candidate winner hits by season **11,7,13,19,8,7**. Strict overlap **0.6448 / 0.5571**, null **0.2077 / 0.1857**. Seed-168 null winners **16/61 / 19/70**, illustrating E172's warning that one shuffled classifier can still strongly prefer dominant drivers by chance; do not treat it as a fixed 5% comparator. Driver history coverage **99.34% / 99.36%**, team **99.51% / 99.86%**.

Synthetic aggregation/no-history, real-source target/future scrambling and past-input responsiveness passed, along with standard no-leak/pre-weekend/profile/guard/metric checks. Independent saved-ranking audit matches all per-race winner/podium metrics. The extractor normalizes led laps by the race-wide maximum lap count, so summing teammate shares has a common denominator. **Verdict: 75% target not reached; no demonstrated gain or promotion.** This is the highest observed development winner count so far, but it is an exploratory two-race improvement after many looks, with a negative inner comparison against the base winner model. Do not tune this source or blend weights on outer outcomes. No 2026 model evaluation or results refresh.

### E180 — Primary-method literature audit for the 75% pre-weekend winner objective (source audit, no model score)

After E178/E179, a broad exploratory method search found qualifying-centered studies, driver/constructor Bayesian research and several incompatible accuracy claims. Fix the next inspection to three primary research pages: van Kesteren/Bergkamp's 2023 driver/constructor analysis (DOI 10.1515/jqas-2022-0021), the 2026 dynamic driver/constructor state-space paper (arXiv 2608.04629; studies 2014–21), and the qualifying-predictive-power paper (arXiv 2507.10966). Check prediction cutoff, target/metric, chronological evaluation, and any input or assumption genuinely missing from the repo. Reading methods is not ingesting outside text as features; no article, market, or betting input will enter a forecast. No accounts, licences, purchases or restrictive scraping.

**Exposure note:** the initial unrestricted search incidentally returned some 2026 news/result snippets. These were not opened or used in modelling; no 2026 result refresh, prediction scoring or tuning was performed. These seasons were already repeatedly inspected, so do not describe 2026 as pristine. Subsequent method searches/opens will use specific research pages and domain filters to avoid further incidental race-outcome exposure. Report the audit as exploratory research coverage, not exhaustive proof that 75% is impossible. Save findings and admissible follow-up candidates in a new root `WINNER_METHOD_AUDIT.md` before choosing another model experiment.

**Result (E180):** audited all three primary methods against cutoff, metric and evaluation protocol; findings and ranked avenues are in `WINNER_METHOD_AUDIT.md`. None demonstrates 75% exact winners across all races before the weekend: the 2023 study is explanatory with unknown seasonal effects, the dynamic model uses same-race observations and future-informed smoothing, and the qualifying study measures same-weekend association rather than our target. A causal joint qualifying/ranking latent filter is a distinguishable modelling hypothesis, not an established gain; constructor continuity and winner-relevant pace measurement are lower-priority audits. No new forecast score, model promotion, data ingestion or 2026 evaluation. Verified the smoothing and seasonal-effect caveats directly in the primary discussion sections, and checked claimed novelty against E30/E80/E99/E169. Target remains unmet; goal active.

### E181 — Causal joint qualifying/ranking latent filter (registered before scoring)

**Question:** can shared driver/team states jointly updated by completed qualifying times and official full-field race ranks improve exact winners, unlike separately tested pace filters and feature-based rank models? Competing explanation: changing race-specific strength is not recoverable from historical observations. E180's papers do not settle this. This is a new fixed approximation, not a reproduction of their smoothed Bayesian analyses.

**Contract and scope:** input is existing structured official results, 2019–22 only, sorted by race; output is a prior win probability per entrant before its weekend. No current/future observations, global scales, future identity vocabulary, grid, practice, sprint or telemetry. Register identities only as their entry lists arrive. Unknown driver/team starts at zero with variance .25/1.0. Between races add diagonal process variance .0025/.04 for drivers/teams. At a regulation-era transition, halve team means and their cross-covariances and add team variance 1; driver states carry. All these constants are fixed before outcomes, not selected on inner or outer results.

At each completed event: qualifying observation = negative within-event standardized minimum of available Q1/Q2/Q3 times (skip missing; fewer than two or zero spread → no qualifying update). Center the driver+team design on observed qualifiers; Gaussian observation variance 1. Jointly update with a full-field Plackett–Luce likelihood on official order, including retirees. This models observed finishing outcomes rather than treating retirees as absent competitors; no separate reliability claim. Use Gaussian prior + convex likelihood MAP, analytic gradient, then full Hessian inverse for a Laplace posterior. Predict before updating via 2,048 seeded Gaussian state draws and mean softmax, retaining shared-team uncertainty. No same-race grid in any equation. This approximation is not full posterior sampling. Bounded budget: one model, one jointly shuffled-observation null (permute qualifying/rank assignments together within each completed event, seed 181), ≤300 optimization iterations per update; abort on failure, no silent fallback.

**Inner gate before any outer scoring:** evaluate all 61 races in 2020–22; exact winner primary, strict overlap secondary, winning-team count and categorical winner log-loss. Compare standings, frozen `ens`, E168 winner; paired bootstrap and matched shuffled null. Proceed to a separately registered outer run only if ≥35/61 winners (at least 3 more than E168), winning-team count ≥ E168, winner log-loss below E168, and more winners than null. Otherwise close this fixed formulation without tuning or inspecting outer scores. Even passing is exploratory after extensive prior work, not a success claim for the 75% all-race target.

**Verification before results:** finite-difference gradient/Hessian; missing/constant qualifying, empty history, probability normalization, positive covariance and earlier-source responsiveness. Target-and-future qualifying/rank/retirement changes must leave that target prediction exactly unchanged on real source data. Entry identity/race-order validation fail closed. Extend independent saved-ranking audit. Standard no-leak/pre-weekend/profile/guard/metrics tests must pass. No new dependency, outside text feature, market or 2026 access. Checkpoint implementation before measuring outcomes.

**Result (E181):** joint filter **30/61 = 0.4918 inner exact winners**, versus standings **29/61 = 0.4754**, frozen `ens` **30/61 = 0.4918**, E168 winner **32/61 = 0.5246**. Strict overlap **0.5956**, versus standings **0.6066**, `ens` **0.6284**, winner baseline **0.6393**. Joint winner hits by season **11,6,13**; winning-team hits **37/61**, equal to all three registered references. Categorical winner log-loss **2.0775**, worse than E168 **1.5648** (normalized probabilities). Jointly shuffled-observation null **2/61 winners**, **6/61 teams**, overlap **0.1366**, log-loss **3.1143**. Paired exact-winner differences: vs standings **+0.0164 [−0.0984,+0.1311]**, vs `ens` **0 [−0.0984,+0.0984]**, vs E168 **−0.0328 [−0.0984,+0.0328]**.

**Verdict: inner gate failed; no outer run.** The fixed joint continuous/rank observation filter does not improve the available winner decision and has poorer probability quality. This closes this specific formulation, not every Bayesian model. Do not tune process variances, likelihood scales or sampling choices after seeing these results. The 75% all-race objective remains unmet; no promotion, new source extraction, market access or 2026 model evaluation. Implementation checkpoint `9da7ada` was committed/pushed before scoring; source feature history preserves the same 2018 observations as prior baselines while state updates start in 2019.

**Verification:** analytic likelihood gradient and Hessian match independent central differences; missing/constant qualifying and first-event/empty-source cases pass. Real-source separate qualifying and official-rank target/future mutations leave target probabilities exactly unchanged and change later forecasts; row permutations leave all forecasts unchanged. The first synthetic prior test incorrectly assumed uniform entrant probabilities despite asymmetric team covariance; corrected the fixture to symmetric team assignments before scoring. Standard no-leak/pre-weekend/profile/guard/metric checks passed (existing pandas dtype warnings). Independent saved-ranking reconstruction reproduces all winner/podium metrics; separate reconstruction verifies 30 winner/37 team hits, normalized probabilities and 2.0775 log-loss. Saved artifacts `results/e181_inner_*` contain all 61 inner races, no outer races. Unrelated untracked Wikipedia extract untouched.

### E182 — Constructor-history continuity source audit (no forecast score)

**Observed:** `f1/data.py` preserves source constructor IDs and `f1/features.py` groups historical team features directly by `team`; there is no lineage normalization in that path. E139's separate `experiments/b6_adaptation_gate.py` already defines a fixed lineage map (`toro_rosso`/`alphatauri`→`rb`, `force_india`/`racing_point`→`aston_martin`, `renault`→`alpine`, `alfa`→`sauber`). That map is used for switch detection, not by the main feature builder. Source audit used only stored results through 2025, with no prediction evaluation or 2026 rows.

First occurrences of new raw names have zero raw-name history but these numbers of earlier events under the existing lineage map: **2019 R1 alfa 21; racing_point 21; 2020 R1 alphatauri 42; 2021 R1 alpine 59; aston_martin 59; 2024 R1 rb 125**. A returning raw name can also omit the intervening alias years (`sauber` in 2024); it is not captured by a first-ever-name check. Counts use strictly earlier races, never future observations.

**Verdict:** a discontinuity is verified, but beneficial continuity is an assumption, not a proven data bug or winner improvement. Do not alter production features automatically. Next eligible test is one fixed inner-only E168 winner model rebuilt with this already-existing lineage mapping, preserving original current-team identities for scoring and an unchanged raw-name baseline. Use the same 35/61 exact-winner, nondecreasing winning-team and better winner-log-loss inner gate before any outer run. Scramble target/future sources and verify history invariance. No process-variance or blend tuning, and no reopening the failed E181 filter. No forecast accuracy claim from this audit.

### E183 — Fixed constructor continuity in winner history (registered before scoring)

**Hypothesis:** preserving prior observations across the raw-name splits verified in E182 improves winner forecasting; alternative is that continuity concerns mostly non-winning teams and cannot improve contender decisions. Use E139's existing fixed `LINEAGE` dictionary unchanged. Build raw and mapped copies with existing `build_features`; merge only mapped history columns selected by the existing pre-weekend list's `t3_`, `t10_`, `t_season_`, `t_circ_` prefixes. Preserve original current entry team identities, all driver/context features, labels and raw baseline columns. No production feature change, new source, external narrative or lineage choice from outcomes.

**Contract:** input is official result/pace rows through 2022, output raw feature table plus `lineage_*` historical team columns keyed one-to-one by season/round/driver. Historical aggregation uses only strictly earlier events, including 2018 feature history; train strictly earlier 2019+ events. Duplicate entrant keys fail validation, unknown team names remain unchanged, first history stays missing except mapped predecessor history, era downweighting remains existing builder behavior. No empty race evaluation or concurrent mutation; file writes happen only after the single run finishes.

**Fixed candidate:** same E168 binary winner logistic C=.03/hl100; replace team columns rather than append duplicate measurements. Same matched shuffled-winner null (seed168); raw standings, frozen `ens` and E168 unchanged. All 61 inner 2020–22 races only, exact winners, strict podium overlap, winning-team count, normalized categorical winner log-loss, per-season hits and race-paired intervals. Gate: ≥35/61 winners, winning-team hits ≥ raw E168, log-loss below raw E168, and more winners than null. Fail → no outer run, no map/weight tuning; pass → separately register one locked outer comparison. 75% all-race goal unchanged. Verify constructor history at a known rename, current identity preservation, unknown-ID behavior, raw-column equivalence, source target/future mutation and past-source responsiveness, and independent saved ranking/probability reconstruction. Standard leak/guard/metric tests required. Commit setup before scoring.

**Result (E183):** lineage winner **32/61 = 0.5246 inner exact winners**, identical hit/miss sequence to raw E168. Standings **29/61**, frozen `ens` **30/61**; winning-team counts lineage/raw both **37/61**. Strict overlap **0.6393**, identical aggregate to raw E168 (standings **0.6066**, `ens` **0.6284**). Lineage seasonal winners **11,8,13**. Normalized winner log-loss **1.5679** versus raw E168 **1.5648**, slightly worse. Matched shuffled null **3/61 winners**, **8/61 winning teams**, overlap **0.1311**, log-loss **3.1477**. Paired winner differences vs standings **+0.0492 [−0.0656,+0.1639]**, vs `ens` **+0.0328 [−0.0492,+0.1148]**, vs raw E168 **0 [0,0]**.

**Verdict: inner gate failed; no outer run.** Continuity restores historical observations but changes no inner winner hit/miss; no production change or promotion. No mapping/weight tuning. Implementation was checkpointed as `d3093c0` before scoring. Synthetic rename/missing-history/unknown-ID and invalid-key checks passed; original source identity and every raw feature remain exact. Real target/future source/pace mutation leaves target historical features unchanged, earlier changes affect them. Standard leak/pre-weekend/profile/guard/metric tests passed. Independent saved ranking audit matches winner and podium metrics; raw standings/logit/winner/ens prediction scores exactly match E181's saved references; separate probability reconstruction verifies log-loss and all 61 winners. Artifacts `results/e183_inner_*`, no 2026 scoring; unrelated untracked extract untouched. Goal active, 75% unmet.

### E184 — Prior sprint points in season totals (registered before scoring)

**Audit finding:** `f1/data.py` stores official `s_points`, while `f1/features.py` season totals sum only race `points/25`. Existing published references therefore implement a race-points prior with historical-form ties, not exact official championship standings. Preserve frozen references and label this limitation; do not silently change previous results.

**One fixed winner candidate, inner only:** reconstruct driver/team season points from strictly earlier events' `points + s_points.fillna(0)`, normalize by season leader exactly as existing features do, leave all other features and hyperparameters unchanged (E168 C=.03/hl100). No same-weekend sprint input. Also report the untrained sprint-inclusive points selector with the existing d10-order tie rule, explicitly not official championship countback. This isolates missing sprint points, not a new standings tie-break optimization. First race/no season history → missing learned features and existing selector fallback. All 61 inner races 2020–22, earlier-2019+ training, 2018 feature history. Matched shuffled-winner classifier seed168, raw standings/ens/E168, paired bootstrap, exact winners/strict overlap/team hits/winner log-loss and seasonal counts. Same inner gate (≥35/61 winners, team count ≥E168, improved log-loss, beat null) before any outer run. Do not tune based on outer or retrospectively relabel old standings as official.

**Verification:** hand-computed synthetic prior totals with missing sprint values and new-season reset; independent season cumulative sums; target/future sprint/race points changes must leave target features unchanged; earlier sprint changes must affect them. New columns explicitly historical and guarded; unchanged raw baseline scores independently reproduce saved inner references. No new source extraction, outside input, production change or 2026 model scoring. Commit implementation before outcomes.

**Result (E184):** sprint-inclusive winner model **32/61 = 0.5246 inner exact winners**, versus frozen race-points selector **29/61 = 0.4754**, `ens` **30/61 = 0.4918**, raw E168 **32/61 = 0.5246**. The sprint-inclusive untrained points selector remains **29/61**. Neither candidate changes a single predicted winner name or winner hit/miss relative to its corresponding raw baseline. Winner-model/team hits **37/61**, selector/team **37/61**; candidate/model strict overlap **0.6393**, selector **0.6066**. Candidate winner log-loss **1.564452** vs raw E168 **1.564779**, a tiny probabilistic change without better winner choices. Null **5/61 winners**, **9/61 teams**, overlap **0.1311**, log-loss **3.132334**. Seasonal model winner hits **11,8,13**; selector **9,8,12**. Model paired winner differences vs race-points baseline **+0.0492 [−0.0656,+0.1639]**, vs `ens` **+0.0328 [−0.0492,+0.1148]**, vs raw E168 **0 [0,0]**. Selector difference vs raw points **0 [0,0]**; vs `ens` **−0.0164 [−0.0984,+0.0656]**; vs E168 **−0.0492 [−0.1639,+0.0656]**.

**Verdict: winner gate failed; no outer run or promotion.** Prior sprint points are a real omission in the frozen baseline description but do not improve inner winner choices in this fixed formulation. README/HANDOFF now clarify that old “standings” figures use race points and form-based ties; no old results were replaced and this test does not establish exact official countback rankings. Implementation checkpoint `f9dcf31` preceded scoring. Synthetic totals/reset/missing-sprint/duplicate-key checks and independent real-source separate-column sums passed; target/future race+sprint point mutation leaves target features unchanged, earlier sprint changes affect them. Standard leak/pre-weekend/profile/guard/metric checks passed. Independent saved-ranking audit reproduces all winner/podium metrics; every raw baseline score equals the prior saved reference exactly, and separate normalized-probability/team/winner reconstruction matches printed results. Artifacts `results/e184_inner_*`; no new source extraction, production change or 2026 model evaluation. Unrelated untracked Wikipedia extract untouched. Goal remains active, 75% unmet.

### E185 — Winner alignment and temporal transfer of existing pace measurements (registered before scoring)

**Diagnostic question:** E174's median pace hindsight rule is weak for exact winners. Do the already extracted corrections materially change winner alignment, or does their signal fail between races? Audit all nine existing measurements without picking a best one after scoring: E94 median/p25/fuel-corrected/first-stint/best-stint/tyre-corrected, E102 clean-air matched, E111 joint corrected, E119 fresh-tyre corrected. This is a measurement audit, not nine new optimized classifiers. Read only stored 2018–22 rows; no extraction, new source, telemetry access or outer/2026 scoring.

**Fixed rules for each measure:** (1) explicitly inadmissible `HINDSIGHT_*` rank of the target race's measurement, used only to describe alignment; (2) pre-weekend `PRIOR_*` rank of the 10-race-half-life weighted mean of strictly earlier driver-code observations, same decay for every measure; (3) matched null for (2), source values permuted among entrants within each completed source race with seed185. No future scaling, fitted weights, era/window search, filtering races by measurement availability, or post-hoc selection. Missing measurement/history ranks last; deterministic driver-name ties. Use all 61 inner races 2020–22, including source-missing races; report actual-winner source coverage. Compare unchanged inner race-points selector, `ens`, E168 winner and pure median history, paired winner intervals. Exact winners first, strict overlap and winning-team count secondary. Hindsight selectors are not forecasts, not proven bounds and cannot establish target success. No claim that physical speed is itself sufficient to determine winners.

**Decision rule:** no source may reopen a forecasting route unless its hindsight rule gets ≥46/61 exact winners (at least 75%) and its fixed prior rule gets ≥35/61, exceeds the median prior rule with paired 95% winner interval excluding zero, and beats its matched null. This is a deliberately explicit measurement/transfer gate, not a theorem that every useful feature must pass it. A failure closes only this direct pace-ranking route; it does not establish information-theoretic impossibility or prohibit a different mechanism. Do not change corrections based on outer results or add a classifier when these rules fail. Commit setup before outcomes. Verify prior-source target/future independence, earlier responsiveness, manual decay/missing cases, shuffle assignment integrity, independent selected-driver/overlap reconstruction and all-race denominator. Standard leak/guard/metric checks required for any historical-feature helper.

**Result (E185), all 61 inner races:** counts below are exact winners; strict overlap is shown for the admissible prior rule. All hindsight counts use target-race truth and are inadmissible as forecasts.

| fixed measurement | hindsight winners | prior winners | prior winning team | prior strict overlap | shuffled-prior winners |
|---|---:|---:|---:|---:|---:|
| median | 44 | 31 | 36 | 0.6175 | 0 |
| p25 | 35 | 31 | 36 | 0.5956 | 0 |
| fuel-corrected | 41 | 32 | 37 | 0.6284 | 0 |
| first stint | 31 | 34 | 39 | 0.6393 | 4 |
| best stint | 29 | 32 | 37 | 0.5792 | 2 |
| tyre-corrected | 40 | 32 | 37 | 0.6066 | 0 |
| clean-air matched | 35 | 35 | 40 | 0.6557 | 2 |
| joint corrections | 40 | 32 | 37 | 0.6557 | 1 |
| fresh/used joint | 37 | 33 | 38 | 0.6557 | 0 |

Raw race-points prior **29/61**, `ens` **30/61**, E168 winner **32/61** (same unchanged references). Clean-air matched prior **35/61 = 0.5738**, median prior **31/61 = 0.5082**. Matched winner differences vs median **+0.0656 [0.0164,0.1311]**, vs race-points prior **+0.0984 [−0.0164,+0.2131]**, vs `ens` **+0.0820 [0,+0.1639]**, vs E168 **+0.0492 [0,+0.1148]**. Matched prior yearly hits **11,10,14**, median **11,7,13**. This is a nominal inner-only pace-transfer signal after nine comparisons and extensive earlier development; not multiplicity-adjusted confirmatory evidence, not an established gain against the actual winner baseline. No source selected for an outer run.

Hindsight median has the highest measured winner alignment, **44/61 = 0.7213**, below the pre-registered 46 threshold. Hindsight actual-winner measurement coverage is **60/61** for median/p25/fuel/best/matched/joint/fresh, **52/61** first-stint and **59/61** tyre-corrected; missing races remain in the denominator. Corrections can improve temporal transfer while reducing alignment with winners in the observed race. That mismatch is further evidence against treating these hindsight rules as proven forecasting bounds.

**Verdict: all registered direct pace-ranking gates fail; no outer run.** The matched signal merits honest recording, but the 75% all-race goal remains unachieved, and the gate cannot be relaxed after viewing outcomes. No correction/decay tuning or new classifier from these failed rules. Different execution mechanisms would need their own prior hypothesis and measurement evidence, not a renamed pace variant. Implementation checkpoint `30dbe38` preceded scoring.

**Verification:** manual decay, missing-history, deterministic null assignment integrity, real target/future-source invariance and earlier-source responsiveness passed, as did standard no-leak/pre-weekend/profile/guard/metric checks. Independent artifact test reconstructs every saved winner, strict-overlap and winning-team hit from selected drivers and official labels; verifies all 61 races for each of 30 rules and all hindsight availability/selection counts. A separate scalar summation of historical observations reproduces median/matched prior top-three selections on two inner races. Frozen baseline metrics match E181 artifacts. Source audit/forecast selection is inner-only; no extraction, outside/market inputs, production change or 2026 scoring. Saved `results/e185_inner_measurement_audit.csv`; unrelated untracked extract untouched. Goal active.

### E186 — Driver-specific winner conversion persistence (registered before scoring; retrospective measurement only)

**Distinct hypothesis:** E155 measured team-average finish-rank minus pace-rank residuals and found little persistence. Test driver-level *win-probability* residuals among the three fastest measured cars, conditioning on their pace-rank opportunities. This is not another corrected pace classifier after E185's failed gate. It tests a different mechanism before any admissible winner feature/model is considered. A persistent residual could reflect driver execution **or systematic pace-proxy bias**, not identifiable pure skill.

**Data/contract:** existing official results and canonical `racepace_ff1.csv`, 2018–22 only; all 61 inner races in 2020–22 scored, no covered-only forecast denominator. During a completed event, measured pace ranks (missing last, driver-name ties) define buckets 1,2,3,4–6,7–10,11+ including missing. Expected per-entrant win rates are `(past bucket wins + 1)/(past bucket entrant count + 2)`, normalized across the current field; counts use only earlier races. This expected rule explicitly knows the current race's completed pace and is **HINDSIGHT**, not pre-weekend. Observe residual `actual win − expected probability`. Update driver residual history only on top-three measured pace opportunities, with fixed 10-event half-life and shrinkage denominator 20+weighted opportunities. Missing pace cannot be an opportunity. Expected-rate bucket counts include all entrants/outcomes.

Before each event's labels are revealed, retrieve each driver's earlier residual mean (none→0); conditional corrected probability = clip(base probability + prior residual, 1e−6,1), normalized across the field. Retrospective baseline and corrected rules share the same known target pace, allowing only earlier identity-specific conversion history to differ. One deterministic null (seed186) permutes residual assignments among eligible drivers within each completed race; no null/candidate can use the current residual. Measure eligible-entry Pearson correlation between earlier residual mean and current residual (with race-cluster bootstrap CI), plus all-race exact winners, strict overlap, winning team and categorical log-loss for all retrospective rules. Compare to unchanged saved race-points/ens/E168 references for scale only; none of these HINDSIGHT scores constitutes a deployable gain.

**Gate:** only consider a separately registered pre-weekend feature if residual correlation ≥.15 (E155's existing material-persistence threshold), conditional corrected rule gets ≥46/61 exact winners, has a positive race-paired winner gain over the shared-pace baseline with 95% interval excluding zero, improves winner log-loss with a paired interval excluding zero, and beats the matched null. Otherwise close this fixed conversion formulation, no decay/shrinkage/bucket tuning and no outer scoring. No source selection among E185 variants. This is a measurement gate, not a proof that every potentially useful execution signal must pass it.

**Verification before outcomes:** manual probability normalization and zero-history correction equality; synthetic shrinkage/decay; eligible/missing handling; prior residual histories invariant to target/future source pace and result mutations while earlier changes affect them. Current diagnostic probabilities intentionally depend on current pace and must never be passed as pre-weekend features. Keep them explicitly `HINDSIGHT_` and out of production/harness feature APIs. Independent saved-driver/overlap and log-loss reconstruction; standard guard/metric checks. Commit setup before outcomes. No extraction, outer/2026 evaluation, outside text or market input.

**Result (E186), retrospective only:** eligible-entry residual correlation **0.3494**, nominal race-cluster bootstrap 95% interval **[0.1968,0.5021]**, **180** measured top-three opportunities across the full **61** inner races. This is descriptive persistence, potentially pace-proxy/driver-strength bias rather than causal execution skill; the interval does not model all serial driver dependencies. All-race hindsight baseline, driver correction and null history each get **44/61 = 0.7213 exact winners**, **48/61 winning teams**, seasonal winner hits **12,17,15**. Driver correction changes one predicted winner name, both original and replacement wrong; hit/miss sequences are identical. Strict overlap base/corrected/null **0.7432 / 0.7322 / 0.7377**. Winner log-loss base **1.060146**, corrected **1.064291**, null **1.284859**. Corrected-minus-base winner difference **0 [0,0]**; base-minus-corrected log-loss (positive would help) **−0.004145 [−0.096645,+0.069693]**.

**Verdict: measurement gate failed; no pre-weekend feature or outer run.** A persistent residual is not enough: the fixed correction improves neither exact winners nor average log-loss relative to the shared-pace retrospective baseline. Neither 72.1% figure is pre-weekend accuracy: both rules know actual target-race pace. This closes the registered conversion rule without tuning shrinkage, bins, decay or clipping. No claim that all execution information is absent or that these figures prove a ceiling. The actual 75% pre-weekend all-race objective remains unmet and active.

**Verification:** implementation checkpoint `18150a1` preceded scoring. Synthetic decay/shrinkage and clipped-probability normalization pass; the zero-history comparison uses 1e−15 tolerance for harmless normalization rounding, while causal-history comparisons remain exact. Real target/future pace+label mutations leave prior histories exactly unchanged; earlier observations change later histories. Separate label-only mutations leave current diagnostic probabilities unchanged, whereas current pace mutation changes them, confirming the explicit hindsight boundary. Missing pace creates no conversion opportunity; row permutations preserve results. Standard no-leak/pre-weekend/profile/guard/metric checks pass. Independent saved-artifact test matches official full entrant counts/labels, reconstructs every probability, selected winner/top-three/overlap/team/log-loss, and derives correlation by centered dot products. Artifacts `results/e186_inner_*`; no new source, production change, market input or 2026 model scoring. Unrelated untracked extract untouched.

### E187 — Championship-context source audit and calendar-leak repair (no forecast scoring)

**Finding before any winner experiment:** E69's `experiments/s3_info.py` computes `nrounds = groupby(season).round.max()` over the full feature table, then derives `rounds_left`, `title_live`, and `lateseason`. This depends on the final observed season calendar, not an archived schedule known before the target weekend. Appending/deleting a later event can change earlier features. It cannot be certified as pre-weekend merely because an announced calendar sometimes matches final race count. E69 and E73b (all blocks) used these fields; the frozen predictor and E168–E186 winner experiments do not. Do not rerun the legacy module: its shared table can load restricted 2026 rows.

**Surgical repair:** retain only E69's three past-points context quantities (driver/team gap to leader, leader's margin over second-place points) in an import-safe experimental helper. Remove full-season count, rounds-left/title-pressure/season-fraction fields from the legacy builder and its E69 block. Do not invent a replacement calendar or claim exact title eligibility without a timestamped schedule. No production feature/model change; old E69/E73b scores remain historical records with this new validity qualification, not fresh corrected results. E70/E71/E72/E73 individual blocks do not depend on those calendar fields.

**Contract:** helper accepts an existing feature table containing chronological k and prior driver/team points; returns a copy with exactly three context columns, preserving missing history. No result/calendar metadata is read. Future race rows cannot change earlier context. Tests must demonstrate the old full-season-max dependency on a synthetic appended future event, verify repaired helper invariance on that fixture and real feature tables, and check missing/tied history behavior. Source-current/future result/qualifying/pace mutation must not change target context. Explicit pre-weekend guard and standard leak/metric checks. No new accuracy, outer scoring, source ingestion, or 2026 model evaluation.

**Duplicate audit:** E72's within-team points share is derived from existing driver/team history and E170 already modeled within-team winner choice with centered driver histories. Merely rerunning that ratio as a direct-winner variant is not a materially new data avenue. No winner experiment is authorized by this audit alone; require a distinct, evidenced mechanism rather than re-labelling old context.

**Result (E187):** reproduced the calendar dependency by appending a later same-season event: earlier `rounds_left` changes. Removed final-calendar-derived columns from the legacy E69 and all-block specifications; the remaining three context fields are prior-points gap, team gap and the current field's second-place points margin. New pure helper `experiments/e187_context_audit.py` allows mutation tests without importing/running the legacy all-season experiment. E69/E69_ET and the E73b all-block historical scores are timing-invalid evidence; the E69 summary row is explicitly flagged rather than silently replaced. E73's standalone stage blend does not use the final calendar. No models rerun, no new accuracy result, no changed frozen/production predictor. The static legacy script passes syntax compilation; no claim that its unrun end-to-end experiment was validated.

**Context scope verdict:** E72's teammate-share and form interaction are transformations of already available historical points/form, not observations of team orders or a newly recovered data source. They have not been isolated in a win-only fit, so this audit does not falsely claim an exact winner test already exists. However, another post-hoc feature variant is not justified as discovery of new team-role information. No winner experiment is started from it. Keep the 75% objective and full race denominator.

**Verification:** synthetic future-event append reproduces the original leak and leaves repaired context exact; missing history and tied leaders behave as specified. Real source target/future result/pace mutations and future calendar edits leave target context unchanged, earlier points affect it. Existing raw feature columns are preserved. Standard no-leak/pre-weekend/profile/guard/metric tests and syntax compilation passed. No new source, outside text, market access, outer or 2026 model scoring; unrelated extract untouched. Next inspect E99's already measured team-strength fusion specifically at the winning-team decision before proposing another model; its full-order correlation gain does not establish top-team/winner accuracy. Audit existing formulas and artifacts first to avoid duplication and accidental outer selection.

### E188 — Repair fusion scaling and test the winning-team decision (registered before scoring)

**Source audit finding:** E99's `experiments/u_items.py` divides centered team indicators by `t[c].std()` over all stored seasons, including evaluation/future events. That makes historical filter states depend on future values through measurement units and relative weights. The published E99/E100 fusion numbers cannot be certified as causal pre-weekend evidence; preserve but flag them. This affects the experimental fusion path, not the frozen predictor. The same file also uses full-table medians in E98 missing-value interactions; note that separate legacy timing issue without treating old E98 negatives as certified evidence.

**Repair/contract:** normalize each of six original indicators by its centered team-race standard deviation in **2018 only**, frozen before forecasting 2019+. Reject missing/zero calibration scales. Warm filter state on 2018 observations using this known calibration, return forecasts only from 2019; no claim that intermediate 2018 states were pre-event forecasts. Six indicators unchanged: canonical race pace, fuel-corrected race pace, Q1 gap, FP long-run gap, negative race points, official order, team means centered on each historical race's field. Missing observations skipped. Use only stored through-2022 files for this audit. Raw constructor IDs retained; no new lineage tuning.

**Fixed filter:** same original m=0/P=3 initial state, q=.03 each event, season variance addition .4, predict before updating. Equal-six observation variance r=2 (the original untuned comparator); reference single race-pace indicator r=1. No 4^6 noise grid or process search. Matched null permutes complete centered team observation vectors among teams within each completed event (seed188), using the same frozen calibration. Winner selector chooses lowest forecast-strength team, then that team's best-ranked E168 driver using saved, unchanged pre-weekend probabilities; promote only that driver to P1 and preserve other baseline order. This is a fixed winning-team probe, not a new flexible learner. No calibrated win-probability claim.

All 61 inner races 2020–22, actual winner first, strict overlap/team hits, yearly counts and paired race intervals versus frozen race-points/ens/E168 and single-indicator selector. Gate for any separately registered outer run: ≥35/61 exact winners, ≥40/61 winning teams, positive paired winner improvement over single indicator and E168 with 95% intervals excluding zero, and beat matched null. Otherwise close this formulation without retuning. No new source, outer/2026 scoring, market or outside-text input. Reproduce the old future-scaling dependency synthetically; prove frozen scaling/filter forecasts unchanged under target/future indicator mutations and responsive to earlier inputs. Independently reconstruct saved selections, all-race metrics and original baseline scores. Standard leak/guard/metric tests required. Checkpoint before outcomes.

**Result (E188), all 61 inner races:** repaired equal-six filter **28/61 = 0.4590 exact winners**, **34/61 winning teams**, strict overlap **0.6393**. Single race-pace filter **25/61 = 0.4098 winners**, **31/61 teams**, overlap **0.6393**. Matched team-vector permutation null **3/61 winners**, **4/61 teams**, overlap **0.4918**; this is a partial team-signal null retaining the baseline's other driver ordering, not a fully randomized podium model. Unchanged race-points/ens/E168 winners **29/61 / 30/61 / 32/61**, each **37/61 teams**. Equal-six yearly winners **11,6,11**, single **11,6,8**. Equal-six paired winner differences: vs race-points **−0.0164 [−0.1148,+0.0820]**, vs `ens` **−0.0328 [−0.1311,+0.0656]**, vs E168 **−0.0656 [−0.1475,0]**, vs single **+0.0492 [−0.0164,+0.1311]**. None establishes an improvement; the registered inner gate fails. No outer run, filter/noise tuning or production change. No calibrated probability/log-loss claim for these rank selectors.

The fixed pre-2019 calibration scales are pace **1.123053**, fuel pace **1.067835**, Q1 gap **0.868035**, FP long-run **0.850547**, negative points **6.218788**, order **4.847433**. These units are frozen from 2018 only and do not use later measurement distributions. The historical E99/E100 gain claims remain timing-invalid rather than replaced by an incomparable winner score. E98's separate global-median issue is flagged explicitly for repair; the combined legacy script was syntax-checked but not executed, so this does not certify all of its other experiments.

**Verification:** checkpoint `1c0a805` committed/pushed before scoring. Synthetic future values reproduce the old global-standardization dependency; fixed 2018 scales, single-indicator update math, zero-scale rejection and raw-input/real-source cutoff tests passed. Target/future observations leave target forecasts exact; earlier observations change them. Standard leak/pre-weekend/profile/guard/metric checks pass. Independent artifact test verifies all 61 races per six rules, original official labels, chosen team and within-team winner from saved pre-weekend baseline rankings, every winner/overlap/team hit and unchanged reference metrics. Artifacts `results/e188_inner_*` include only 2019 warmup forecasts and 2020–22 evaluation; no 2026 scoring, new extraction, outside or market input. Unrelated untracked extract untouched. The 75% objective remains unmet and active.

### E189 — Remove future-distribution imputation from static form interactions (leak repair, no forecast scoring)

E188 identified E98's `X[c].fillna(X[c].median())` when constructing altitude/form and temperature/form interactions over the full feature table. Added a regression test that executes the actual interaction AST block without importing/running the legacy all-season script. **Reproduced before fixing:** appending one future row changes an earlier missing `alt_x_t10_order` from **2 to 501**, a direct future-distribution dependency.

**Repair/contract:** preserve missing form values through the two interactions. Existing logistic/extra-trees pipelines impute on strictly earlier training rows, while LightGBM handles missingness natively; no new global imputer or model is introduced. Known-value interactions and the fixed 24-degree missing-temperature convention remain the same. Tests verify the actual legacy loop is unchanged by future appended values, retains missingness, preserves raw columns, and computes the intended zero interactions on observed forms. Syntax compilation passes. The old E98 scores remain timing-invalid rather than silently replaced; no model rerun, new accuracy result or production change. Run standard leak/guard/metric checks before committing. This is research-integrity work, not progress claimed toward 75% accuracy.

**Next avenue audit:** E111/E119 estimated a common traffic nuisance coefficient, and E102 selected clean-air laps; neither explicitly measured a car/driver-specific response to *start-of-lap* traffic using disjoint within-stint lap pairs. The prior log itself notes that end-of-lap timing and same-lap-number positioning can misclassify lapped traffic. A new traffic-response measurement would first need a timing-valid observable, enough paired samples and independent repeatability, before any winner model. It may reflect DRS, traffic composition or driver behavior, not identifiable aerodynamic skill. Audit cache/source feasibility and pre-register a bounded measurement pilot, rather than claiming this is already a useful predictor.

**E189 verification completed:** actual-loop regression test and syntax compilation pass; standard no-leak/pre-weekend/profile/guard/metric checks pass. Frozen predictor unchanged. No all-season legacy script or model scoring was run.

### E190 — Cache-only start-of-lap traffic source feasibility (registered before extraction; no winner score)

**Distinct observable:** E111/E119 used a common traffic nuisance term and same-lap-number/end-of-lap gap estimates; this pilot asks whether cached timings support driver/car-specific close-following contrasts from traffic exposure already present at lap start. Fixed pre-inner sample: **2019 Spain R5, Hungary R12, Italy R14**, chosen for circuit variety, not outcomes. FastF1 cache only, offline mode enabled after cache initialization. No HTTP, credentials/accounts, new telemetry, sessions outside these three, outer or 2026 data. Do not call FastF1 results labels for evaluation.

**Timing definition:** FastF1's official timing explanation defines `Time` as lap completion session time and `LapStartTime` as the prior lap's completion time; they are derived timestamps with imperfect precision ([source](https://github.com/theoehrly/fast-f1/blob/main/docs/data_reference/time_explanation.rst)). For a clean green/accurate non-pit lap, start exposure = another driver's clean non-pit lap-line crossing in the **preceding 1.5 seconds** before its LapStartTime. Exclude the same driver, use absolute session time rather than matching lap numbers, and do not use the current lap's completion gap as exposure. This is a timing proxy, not continuously observed traffic or causal aero loss; pit-line geometry, DRS and gap evolution remain limitations.

**Paired-sample definition:** consecutive clean driver laps within the same stint/compound, tyre life increases by one, end-lap position unchanged, exposure switches close↔clear. Retain disjoint pairs greedily in lap-number order so no lap contributes twice. First-lap/pit/neutralized/inaccurate laps excluded; lap time <1.07×driver clean median. No effect-size, persistence or winner model estimated in this feasibility step. Record required schema availability, laps, usable drivers, pair counts by driver, both switch directions and zero-pair entrants. Cache failure is recorded, not retried online.

**Feasibility gate:** all three sessions available with required fields; ≥60 disjoint pairs overall and ≥20 in at least two sessions; at least half of timing-table drivers have ≥2 disjoint pairs in at least two sessions. Otherwise stop this fixed pilot without threshold changes or larger extraction. Passing permits only a separately pre-registered independent-repeatability/temporal-transfer measurement, not a winner feature. Tests: synthetic start-time exposure independent of current lap completion and lap-number equality, self exclusion, pair disjointness, stint/tyre/position controls. Independent saved-pair count reconstruction. Checkpoint before cache access. No accuracy or 75% claim from source availability.

**Result (E190): feasibility gate passed, no accuracy measured.** Cache-only sources available for all three fixed sessions with every required field. Spain: **1,276 raw / 1,062 clean laps**, **122 disjoint pairs**, **20/20 drivers** with ≥2 pairs; Hungary: **1,358 / 1,268 laps**, **134 pairs**, **19/20 drivers**; Italy: **991 / 821 laps**, **88 pairs**, **18/20 drivers**. Total **344 pairs**, **202 entering / 142 leaving**. Zero-pair entrants remain in per-session driver denominators. No effect size, driver/car repeatability, race winner or podium result was estimated. Passing only establishes enough timing samples for a next measurement gate; it is not progress claimed as higher winner accuracy.

Official FastF1 timing semantics were checked through Context7 against primary source documentation. Cache offline mode plus a requests HTTP-adapter send block prevented uncached HTTP. Source failures are caught only around session loading; measurement errors propagate rather than being misreported as missing cache. Tests passed for prior lap-start exposure, different lap-number cars, self/pit exclusion, current-completion-time independence, stint/compound/tyre/position controls and disjoint pairs. Independent saved-artifact test reconstructs all session/driver/direction counts, the 344-pair total and the exact gate. An initial synthetic pit-time fixture used datetime missingness rather than timedeltas; corrected/tested in `6bf93f2` before cache access (setup `0161170`). No account/credential action, new telemetry, outer or 2026 scoring. Historical complete-event cleaning uses retrospective quality selection; this is not a live-at-lap-start classifier. Artifacts `results/e190_*`; unrelated untracked extract untouched. Goal active, 75% unmet.

### E191 — Traffic response repeatability and temporal transfer (registered; not yet extracted/scored)

E190 allows a measurement test, not a winner model. Freeze **all 21 scheduled completed 2019 races in existing official results**, cache-only, no additions/substitutions if cache missing. Same E190 timing/exposure/disjoint pairing rules; no threshold search. Source counts justified a pooled team/car measurement rather than sparse per-driver claims. Use official historical code→team mapping from existing results, not FastF1 finish labels.

For each pair compute percent-time contrast `(time2−time1)/mean(time1,time2)*100`, signed positive when entering and negative when leaving traffic. Team-race response is **equal-weight mean of entering and leaving signed contrasts**, so a constant per-lap time trend cancels algebraically despite unequal counts. Require ≥8 disjoint team pairs, at least two of each direction. Compute independent halves from alternate chronologically ordered disjoint pairs, each half ≥4 pairs with both directions represented; no shared lap across halves. Report coverage including missing/zero-pair teams and races. Center each measured response by that race's mean across teams only if ≥6 teams measurable. This is an observational traffic/DRS/driver/strategy response, not identified aerodynamic skill; no such causal claim.

**Measurement gate:** ≥12/21 races with ≥6 valid centered team responses; ≥60 valid independent-half team-race comparisons; split-half Pearson correlation ≥.30 with race-cluster 95% bootstrap lower bound >0. Temporal test predicts a team's centered response from strictly earlier valid 2019 responses only, half-life10 and five pseudo-races at zero; require ≥3 prior observations for a scored measurement pair, ≥60 such team-race rows. Compare MSE against zero (the centered-field baseline), requiring positive paired MSE gain with race-cluster bootstrap lower bound >0 and temporal correlation ≥.15. A within-race team-identity-permutation null (seed191) is reported. All fixed gates must pass before a separately registered full-denominator 2020–22 winner feature is considered. Missing source teams are explicitly excluded only from the measurement-pair diagnostic and counted in coverage; no selective race accuracy claim. If any gate fails, stop without loosening pair counts, changing exposure cutoff, normalization or pooling level. Test constant-trend cancellation, disjoint halves, prior-source cutoff and independent artifact reconstruction before reporting. No 2026, outer model score, new account or API access.

**E191 pre-measurement clarification (2026-11-16):** mathematical review found that direction balancing exactly cancels a common drift in *percentage-point pair contrasts*, not every constant drift in absolute seconds when pair denominators differ. For example +1 second normalized by 90 versus 100 seconds leaves a nonzero balanced remainder. Retain the registered normalized formula and all gates; narrow the cancellation claim and retain fuel/tyre/strategy confounding as a limitation. No measured effects have been inspected. The permutation null shuffles centered response values only among measured teams within an event, retaining each team's missingness/opportunity pattern and the same temporal evaluation rows. Full and half responses are each race-centered only when at least six valid team values are available for that response. Missing sessions/teams remain explicit coverage failures, not substituted races.
