# gp-predicts — F1 race predictor (walk-forward, season-cycle evaluated)

**Current objective (2026-11-16): reach **75% exact pre-weekend winner accuracy across all races**, accepting lower second/third-place accuracy. Research is active.** Strict podium overlap remains a secondary metric. E168–E179 tested direct winner labels, race-conditional likelihood, team/driver factorization, explicit win history, boosted trees, faster training recency, portable teammate pace, pre-season testing, best-car team history, inner-selected regularization and past laps-led share. No candidate passed the paired improvement gate: outer 2023–25 standings and frozen `ens` hit **32/70 winners (45.7%)**; the highest observed development variant hit **34/70 (48.6%)**, an unresolved two-race difference on repeatedly inspected development data. The production predictor is unchanged; no new 2026 evaluation was run. Run the experiment modules individually, e.g. `.venv/bin/python -m experiments.e169_choice_winner`; E170–E174 use E169's saved baseline artifacts, and E172 also uses E171's metrics.

**Baseline clarification (E184):** historical tables labelled “standings” use prior race points, omit sprint points, and break ties with historical finishing form. They are frozen reproducible points priors, not exact official championship rankings. E184 tested a sprint-inclusive version on 61 inner races: winner-model picks stayed 32/61 and the points-selector picks stayed 29/61. No outer run; previous numbers are unchanged.

**Roster timing assumption (E251):** historical backtests use each race’s eventual entrant identities and team assignments, assuming they were known before the cutoff. A previous-race-roster sensitivity check on all61inner2020–22races left every winner hit/miss unchanged, despite11within-season roster changes. Announcement times and season-opening lineups remain unverified; this is not a blanket certification of roster availability.

**Original 75% podium target: not reached; final evaluation logged in E159.** Previously evaluated 2026 (19 races): **0.561** strict podium overlap, vs championship standings 0.526 and a shuffled-label null 0.228. Post-qualifying: 0.649 vs grid 0.614. These 2026 races have been inspected repeatedly and are restricted during further development. Hindsight pace and team-order selectors are diagnostic measurements, not proven ceilings; E166's combined true pace/retirers rule reaches 0.757 outer overlap. `HANDOFF.md` records current direction; `EXPERIMENTS.md` contains full results, including negatives.

| setting | held-out 2026 podium accuracy | baseline | dev 2023-25 |
|---|---|---|---|
| **A. Pre-weekend** (no practice/quali/grid of the target race) — final model `ens` + pre-season-testing rule for rounds 1–3 | **0.561** (vs `ens` alone 0.544) | standings 0.526 | standings 0.590 / `ens` 0.586 |
| B. Post-qualifying (grid known) | 0.649 (95% CI 0.53–0.75) | grid 0.614 | 0.67 / 0.67 / 0.74 |
| *Oracle: true race pace of the target race known (post-race info)* | *0.79 (2022-25: 0.746)* | | |
| *Oracle: true team finishing order known* | | | *0.743* |
| *Oracle: true grid penalties known in advance (E158)* | | | *no gain* |

Podium accuracy = mean over races of |predicted top-3 ∩ actual top-3| / 3 (defined before any model was run; E0, E7).

## Run it
```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python predict.py               # NEXT race from earlier races only (no weekend data needed)
.venv/bin/python predict.py 2026 19       # as-of replay of a finished race (weekend hidden), actual podium shown after
.venv/bin/python predict.py --post        # post-qualifying variant, once qualifying is on Jolpica
.venv/bin/python run.py --pre             # pre-weekend walk-forward 2023-2026 -> results/preweekend_summary.md
.venv/bin/python run.py                   # post-qualifying walk-forward        -> results/summary.md
.venv/bin/python -m tests.test_preweekend && .venv/bin/python -m tests.test_no_leak
```
`data/results.csv`, `data/racepace_ff1.csv`, `data/laps.csv` are committed so everything reruns offline. To refresh: `python -m f1.data`, and for
FastF1 extractors (need FastF1>=3.9 and an F1 account): `F1_MODULE=f1.<ff1|sessions|telemetry|strategy|pits> ./scripts_run_ff1.sh <.env with F1_USERNAME/F1_PASSWORD> <first> <last>`, then `python -m f1.ff1 merge` for race pace. Experiments: `python experiments/e9_profile.py`, `e10_similarity.py`, `e11_blocks.py`.

E268's optional CatBoost experiment uses `requirements-research.txt`. Its fixed ordered categorical model scored **33/61 inner winners**, tied the existing numeric GBM, and failed its improvement gate. It is not used by the production predictor; the saved artifacts avoid rerunning an already scored experiment.

## Data (and why it is split this way)
- **Jolpica-F1 (Ergast successor)**: official results, grid, qualifying, sprints, status, 2018-2026. Source of truth for labels.
- **FastF1**: race laps -> clean-lap race pace per driver, 2018-2026 (validated: corr 0.993 vs. OpenF1 on 1,718 driver-races). I tried FastF1 for
  results too and rejected it: its grid/points/status come from Jolpica internally (NaN for many races when rate-limited) and its timing-based
  finishing order ignores post-race penalties (e.g. 2019 Canada), so it is a worse label source.
- **OpenF1** (`data/laps.csv`): practice/quali lap gaps, used only by the post-quali experiments (they did not help, E4).

## Round 2 (E12-E21): exhaustive audit-driven search - result: the ceiling is closed from three sides
- **Yardstick fixed** (tie-break leak found and removed; null test: shuffled labels -> 0.18 podium, random 0.17). Dev widened to 131 races (2020-25).
- **Hindsight headroom** for "persistent form" forecasting is only +1.5 pts of team-order correlation (0.760 legal vs 0.775 with future races) - E13.
- **Information-stage curve** (E17): history-only 0.595 -> +practice 0.585 -> +grid/qualifying 0.687 -> +this weekend's quali telemetry 0.679. **Even with everything known pre-race, ~0.69.**
- Weather (true rain as oracle), Monte Carlo race simulation + retirement model, stacking, nested tuning, dedicated teammate model, longer history, state-space models: none beat plain championship standings out of sample.
- Telemetry corner/engine metrics from one quali lap are mostly noise (week-to-week persistence 0.04): E20 explains why they never helped.
- Looser questions the same predictions answer: >=2 of 3 podium drivers right 70%; real podium covered by top-5 picks 79%; per-driver yes/no accuracy 88% (always-"no" gets 85%). These are NOT the pre-registered metric.

## Round 3 (E22-E57): 36 more experiments, reasoned backward from the result
Start from the real podiums and ask what information was missing. Oracle ladder (E22): knowing retirers +4.0 pts, true teammate order +5.0, **true team order +13.7 (-> 0.743)**.
Even a perfect team order only reaches 0.743 (E56). Then 30+ models/labels/features/algorithms built around that (Plackett-Luce, Elo, team-then-driver, diversity correction, pace labels,
trend/start-skill/laps-led/Q1/sprint/circuit-continuity features, SVM/kNN/NB/RF/GBM/monotone-GBM, selective prediction): **none beat plain championship standings out of sample (outer 0.590)**.
A harness default once leaked target-weekend columns into two models (bogus 0.71); fixed at the root and guarded by a test (`tests/test_guard.py`). Full tables: EXPERIMENTS.md.

## Round 4 (E58-E92): 35 more experiments (audit routes + web research + new data)
Regime effect: learned models beat standings by 4-8 pts in dominant-team eras (2020-21) and lose 3-4 pts in competitive eras (2022-25) (E58/E65); rolling windows don't change it. Power analysis: smallest detectable gain ~4.5 pts at 131 races (E60).
Brier/log-loss shows the learned model carries real probabilistic information over a standings-rank prior (E64). Prediction markets (Polymarket F1 driver-podium markets exist for 20 races) are unusable: resolved markets have no price history and quotes are untraded (E61/E62).
New information (championship context, rookies, calendar, within-team share, telemetry v2, GP with track kernel, regime-adaptive ensembles, team-shock Monte Carlo): no gain. **Pre-season testing pace is the one genuine signal**: team-order correlation for rounds 1-3 rises 0.752 -> 0.813, and testing clearly beats last-year pace after the 2026 reset (E86). Full tables: EXPERIMENTS.md.

## What was tried for the no-weekend-data goal (E9-E11, all from PAST sessions only)
Corner-by-corner telemetry profiles per car and driver (approach/braking, apex, exit/throttle pickup, following straight), an engine proxy
(full-throttle acceleration + straight-end speed, plus power-unit-pooled reliability), driver style (teammate-relative, portable across teams), circuit
character from earlier visits (layout, typical 1-/2-stop strategy, safety-car share, DNF/collision rate, podium chaos), pit-stop and strategy-execution
skill, and car x track "fit" features that update through the season. Result: **none improved on the base features** (pace-order correlation 0.72 either way).
Reason, measured (E10): a team's circuit-specific deviation repeats year to year only at r=+0.04, and track-similarity weighting makes team forecasts *worse* (0.774 -> 0.766).

## Method
- Walk-forward: for each race, train only on strictly earlier races, predict, then roll forward. Dev = 2023-25 (2022 for some tests); 2026 = held out,
  looked at once per setting and logged.
- Features (`f1/features.py`): exponentially decayed driver/team form (half-lives 3 and 10 races) over finish, qualifying, points, DNF, grid-to-finish
  gain, race pace; in-season standings share; circuit-specific team/driver residuals; regulation-era down-weighting.
- Pre-weekend model: rank-average of (a) championship-standings order and (b) a regularised logistic podium model. Post-quali model: 16-feature logistic.
- Leak control: `tests/test_no_leak.py` (scrambles race K+ outcomes) and `tests/test_preweekend.py` (wipes *all* target-weekend data) assert features are
  bit-identical; both were mutation-checked (they fail when a leak is injected).

## Season-cycle findings (pre-weekend, pooled 2023-26)
- Podium accuracy by stage: R1-5 0.60, R6-11 0.58, R12+ 0.56; slope -0.002/round (p=0.64). **No across-season improvement in normal seasons.**
- **2026 (new regulations) does improve**: R1-5 0.40 → R6-11 0.56 → R12+ 0.63 (n=19, noisy). In-season data matters when the pecking order is reshuffled.
- Single-season standard error is about ±6 pts (19-24 races), so season-to-season swings (0.53-0.67) are largely noise.

## Limitations
- 2026 sample is 19 races. Entry list for an upcoming race is assumed equal to the last race's line-up.
- No weather forecast, tyre allocation, strategy or reliability-by-part data. Race pace is a proxy (fuel/tyre/traffic effects not removed).
- The oracle bound uses clean-lap median pace, an imperfect pace measure, so the true ceiling is somewhat uncertain — but not 20 points higher.

## Rounds 5–8 (E93–E159): error attribution, new data avenues, outside documents, final look
- **Error attribution (E93, 89 races):** 61% of podium errors are race-pace-order mistakes, 39% execution/luck (strategy, safety car, start, retirement). Podiums reached without podium pace: median grid 4 → finish 3.
- **Every structured data avenue** in `RESEARCH_AVENUES.md` was gated and closed: clean-air/joint pace corrections, tyre degradation, cause-specific mechanical hazards (Jolpica names zero mechanical failures 2023–25), multi-stint X/Y telemetry profiles (repeatable identity signal, no forecast value), passing conversion, strategy exposure, forecast-grid distributions, portable teammate pace, pre-season testing (the one real early-round signal: team-order Spearman 0.75 → 0.81; adopted for rounds 1–3), reset-year pace blends (E154), team execution residual (E155: persistence 0.07 after strength adjustment).
- **Outside documents (user lifted the restriction, then excluded all betting/market sources):** FIA decision documents are enumerable with exact publish times, but almost everything useful is published during the target weekend; an oracle with *perfect* grid-penalty knowledge changes pre-weekend podium selection by exactly 0.000 (E158), so the PU/penalty route is closed without scraping. Market prices (E132/E147/E149) are recorded as history only and are not used.
- **Yardstick notes:** per-driver "on the podium?" accuracy is 0.88 vs 0.85 for always-no; most public "80–90% podium accuracy" claims are this metric. Smallest detectable gain at 131 races ≈ 4.5 points (E60). Two leaks were caught and fixed by the audit habit (E30/E31 harness default; E132 cutoff timing).
- **Probabilities (re-scope target):** the learned logit beats a standings-rank prior on podium log-loss (2026: 0.239 vs 0.355) and Brier on dev (E64); this is real but is a different goal from 75% overlap.
