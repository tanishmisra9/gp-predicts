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

## ROUND 6 — the three untried data avenues, pre-weekend only (user: "try 1-3, prioritize no-info pre-weekend")
| # | Experiment | Result | Verdict |
|---|---|---|---|
| E95 | Power-unit supplier x ALTITUDE (Open-Meteo elevation; >600 m = Mexico 2234 m, Interlagos 779, Madring 683, Red Bull Ring 681, Las Vegas 649): supplier's pace deviation at high vs low tracks, cross-year persistence | corr **-0.58** (n=27 supplier-seasons) | no positive persistence: engines do not carry an altitude edge year to year |
| E96 | Same at TEAM level | corr **-0.28** (n=73) | none |
| E97 | 2026-style energy-deployment signature from the fastest quali lap (full-throttle acceleration 200-260 and 280-330 km/h, taper, clipping share), persistence / between-team share / validity vs race pace | persistence: 2022-25 cars .11-.20, **2026 cars acc_mid .42**, taper .20; between-team share 2026 acc_mid .17; corr with race pace -0.15..+0.06 in both regimes | a stable car/PU characteristic that does NOT tell you who is fast |
| E98 | Static circuit descriptors as features: altitude, high-altitude flag, historical race-day temperature/rain, historical tyre hardness, + interactions with team/driver form | podium outer: logit .557 (base .557), lgbm .548 (.533), ET-pace .562 (.557) | no gain |
| E99 | Reliability-weighted FUSION of up to 6 weak team-strength indicators (race pace, fuel-corrected pace, Q1 gap, FP long-run, points, finishing position) in a multi-observation Kalman filter; forecast of TEAM finishing order (Spearman vs actual best-car order) | single indicator inner .7546 / outer .7274; fused (noise tuned on inner) inner .7773 / outer .7311; **fused equal weights outer .7403** | small real gain (+1.3 pts) on the biggest unknown (team order) |
| E100 | Fused strength -> PODIUM: alone / as logit feature / as third member of the ensemble | outer .529 / .557 / .548 (ens .586, standings .590) | the +1.3-pt team-order gain does not reach the podium |
Observation: team-order forecast quality is bounded by ~0.775 even with hindsight (E13); fusion gets 0.740, i.e. most of the available gain. Every pre-weekend avenue now saturates at team-order ~0.74 and podium ~0.58-0.61.

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
