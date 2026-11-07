# HANDOFF — direction for the agent taking over gp-predicts

Written 2026-11-07 after reading `EXPERIMENTS.md` (E0–E153), `RESEARCH_AVENUES.md`, and `README.md`.
The goal is unchanged: **≥ 0.750 strict podium overlap**, pre-weekend, walk-forward. It has not been reached.
This file tells you where the evidence stands, what not to repeat, and where the remaining headroom is.
Read it before touching code. Read `RESEARCH_AVENUES.md` next for the per-avenue detail.

## 1. The task, exactly

- **Metric:** `mean over races of |predicted top-3 ∩ actual top-3| / 3`. Pre-registered in E0. Do not loosen it; looser variants (≥2 of 3, top-5 coverage, per-driver binary) are reported as secondaries only (E88).
- **Setting:** predict before the target weekend. No practice, qualifying, sprint, grid, telemetry or labels from that weekend. Cutoff is Thursday 00:00 UTC of race week (earlier if the weekend starts earlier).
- **Evaluation windows:** inner = 2020–22 (61 races, tune here), outer = 2023–25 (70 races, honest comparison). Both have been inspected many times; call new outer results development evidence, not a holdout.
- **2026 is restricted.** It was looked at in E6, E8, E86, E93. Do not inspect 2026 labels or residuals during development. One final look only, logged.
- **Data rule from the user:** *no outside documents.* No FIA reports, Pirelli previews, articles, interviews, LLM-extracted text. Structured historical data (Jolpica, FastF1, OpenF1, Open-Meteo) only. Kalshi API is off-limits as of E151. If you believe 75% needs outside documents, say so to the user; do not quietly ingest them.

## 2. Where the numbers stand

| pre-weekend, strict podium overlap | inner 2020–22 | outer 2023–25 | 2026 (19 races, frozen E8 look) |
|---|---:|---:|---:|
| plain championship standings | 0.607 | **0.590** | 0.526 |
| frozen `ens` (rank-avg standings + logit) | 0.628 | 0.586 | 0.544 |
| frozen logit | 0.639 | 0.557 | 0.561 |
| label-shuffled null | 0.175 | 0.186 | — |

Post-qualifying (grid known) tops out at ~0.65–0.69 (E6, E17). The gap to 0.75 is ~0.16 pre-weekend, i.e. roughly one more correct podium pick every two races.

Smallest detectable gain at 131 races is ~4.5 points (E60). Anything smaller is indistinguishable from noise and must not be promoted.

## 3. Ceilings measured so far (read these before proposing anything)

These are measured oracles, not proofs. Each is the result of a specific measurement and decision rule.

| oracle (uses post-race truth) | podium overlap | ref |
|---|---:|---|
| true race pace of every driver in the target race | 0.746 (2022–25), 0.789 (2026) | E10 |
| true team finishing order | 0.743 | E22, E56 |
| true retirers known | +4.0 pts over ens | E22 |
| true teammate order | +5.0 pts | E22 |
| everything knowable before lights-out (grid, quali, practice, telemetry) | ~0.69 | E17 |
| hindsight persistent form (team season mean incl. future races) | +1.5 pts team-order corr | E13 |
| true weather | 0 pts | E16 |

**Implication:** 75% is above every single-source pre-race oracle built so far. Reaching it requires predicting part of what has so far behaved as luck: team order on the day (strategy, SC, first lap, reliability) accounts for ~13.7 pts of headroom alone (E22). The 2026 oracle (0.789) is the one place where the pace ceiling clears 75%, so 2026-regime behaviour is where a real shot exists, if anywhere.

## 4. Closed routes. Do not re-run these.

Every item below tied or lost to plain standings on outer 2023–25. Re-running them with a different learner is variant search, not research.

**Models / learners:** logit, ridge, LightGBM (regress, podium clf, lambdarank, monotone), extra-trees, RF, SVM, kNN, NB, AdaBoost, HistGB, bagged logit, stacking (98 subsets), nested tuning, Hedge ensembles, Elo, Plackett-Luce, Bradley-Terry, Kalman (1- and 2-layer, multi-observation fusion), Gaussian process (time and time×track kernel, the latter is *worse*), Monte Carlo race sim with retirement model and team shocks, team-then-driver, diversity penalty, selective prediction/abstention. (E1–E5, E18–E21, E28–E33, E47–E55, E74–E80, E99–E100.) **Finding:** flexibility hurts; inner–outer gap grows with model complexity (E58). Learned models win in dominant eras (2020–21) and lose in competitive eras (2022–25). Regime detection overfit (E77).

**Labels:** pace-based podium label, log-position ridge, log-points, retired down-weighting, soft labels, multinomial bins. (E34–E38, E78.)

**Features from history:** history windows/decay (E3), trend, Q1 gaps, lap-1 start skill, laps led, sprint form, decayed standings, circuit continuity (E39–E46), championship context, rookie/experience, calendar, within-team share, season-stage blends (E69–E73), race-trim strength (E81), sprints as training (E82), static circuit descriptors incl. altitude and PU×altitude (E95–E98), fuel/tyre-corrected pace variants (E94), clean-air matched pace and joint traffic/compound/age correction (E102, E111–E112, E119), tyre-degradation slopes driver- and team-level (E104, E145, E150), short practice running as attrition warning (E120), pit-stop tail and first-stop timing persistence (E121, E125, E144), forecast-grid distribution (E114–E115, E133–E134), portable teammate pace and adaptation (E116–E117, E139), passing conversion history (E107, E152–E153), cause-specific mechanical hazard (E101; blocked: Jolpica names zero mechanical failures 2023–25).

**Telemetry:** single-fastest-lap corner profiles (E9, E11, E20: persistence 0.04), three-fastest-lap v2 (E83–E85), energy-deployment signature (E97), race speed-trap residual (E126), multi-stint X/Y profiles (E106, E113, E127–E131, E140–E143). The X/Y profile is the one telemetry signal that *is* repeatable year to year (driver-shape corr 0.26) and beat standings on five preselected 2024→2025 circuits (team pace Spearman 0.773 vs 0.638), but collapsed on the full 2025 calendar (0.694 vs 0.667 podium, CI crosses zero) and on 2023→2024 (0.518 vs 0.805). Closed as a scalar. Reopen only with a shape-based forecast on a locked, larger, time-ordered sample.

**Premise checks that came back negative:** "team X is good at track Y" repeats year to year at r = +0.04 (E10); track-similarity weighting makes forecasts worse (E10, E74); wet races are not more upset-prone (E16); race difficulty is not predictable from pre-race descriptors (E27, E89, AUC 0.53); the model does not know when it is right (E57).

**Markets:** Polymarket has no usable history (E61–E62). Kalshi scored 0.722 on 18 races of 2025 vs standings 0.685 (E132, E149) but the user has stopped Kalshi use (E151). Manifold 2024 backtest lost to standings (E147). Manifold 2026 prospective captures for rounds 20–23 are scheduled locally (E148); they are a comparator, not a model input.

## 5. What actually carried information (small, real)

1. **Pre-season testing pace** improves early-round team order: Spearman 0.752 → 0.813 with testing + last-year pace; much stronger in the 2026 regulation reset (0.855–0.882). Podium 0.648 → 0.685 on 18 early races at blend weight 0.25 (E86–E87, underpowered). Affects ~3 of 23 races per season.
2. **Brier/log-loss** shows the learned logit beats a standings-rank prior on probabilities even though top-3 selection ties (E64). If you must optimize, optimize Brier, not overlap.
3. **Multi-observation Kalman fusion** of six weak team-strength indicators lifts team-order Spearman 0.727 → 0.740 (E99) but does not reach the podium (E100).
4. **Multi-stint X/Y profiles** are the only telemetry with cross-year persistence (see §4). Signal exists; forecast value unproven.
5. **Clean passing history** passes its Brier gate narrowly (E152) but adds nothing to podium (E153).

## 6. Where the remaining headroom is (revised 2026-11-07 after E154–E155)

**Closed this session:**
- **R1 (reset-year pace blend) — E154, gate failed.** On 2022 R1–10 the standings+pace blend gained +0.033 (one slot in ten races, CI touches zero), and 2020/2021 showed the same lift, so nothing is reset-specific. Caveat on the test itself: `t3/d3_pace_gap` are already inputs to the frozen logit, so the blend was partly redundant with `ens`; and the hypothesis is really about 2026, which was not looked at. The only clean R1 test is the single permitted 2026 look (see §9).
- **R2 (team execution residual) — E155, stopped.** Race-level persistence 0.08 (< 0.15 stop line). The raw half-season (0.39) and cross-season (0.31) correlations are team strength leaking through (residual correlates +0.58 with pace rank); strength-adjusted persistence is 0.07. E93's "39% luck" class stays luck.
- **R3 (shape-based X/Y forecast) — not run, deprioritized.** E143 already has the profile trailing standings by 0.29 team-pace Spearman on 2023→2024; ~200 team-season rows against a 50-bin shape is a fitting exercise, not a measurement. Run it only if the user explicitly wants the telemetry route exhausted.

**Diagnosis after 155 experiments.** Power is not the problem: a 16-point gain would be detectable at 131 races. The problem is that no permitted pre-weekend source carries information that championship standings do not already summarize. Every structured-data route in `RESEARCH_AVENUES.md` (A1–A8, B1–B9, C1–C3) now has a logged negative or blocked result. Continuing to sweep features and learners on 2020–25 is variant search and should stop.

**What remains, in order. Items 2–4 need the user's decision before any work.**

**1. Freeze the one promotable positive signal (permitted, bounded, do this regardless).**
Pre-season testing pace improves early-round team order (E86) with a suggestive podium lift (E87, n=18). Build the fixed rule *now*: for rounds 1–3 only, `ens_r13 = rank-avg(standings, logit, testing_strength)` with weight 0.25 on testing (E87's value; do not retune); rounds 4+ unchanged `ens`. Score 2023–25 full calendar with coverage reporting (testing cache exists 2023–25 only, E124; inner cannot validate, say so). Expected full-season effect ≤ +0.5 pt. This is the candidate for the final 2026 look. It will not reach 75%; it is the honest best model.

**2. R5 — lift the "no outside documents" rule (recommended, user decision).**
Rationale: team order on the day is the largest unknown (13.7 pts, E22) and the only pre-weekend source that has ever scored above 0.70 here is a market price (E132, 0.722 on 18 races). Markets aggregate exactly the documents the rule excludes (upgrade news, reliability rumours, penalties, driver fitness). If the user agrees, the first batch is **not** LLM text mining. It is, in order of leak-safety:
   a. **Pre-weekend expert pecking orders with publication timestamps** (e.g. archived power rankings / podium predictions published ≤ Thursday 00:00 UTC). Treat each as a ranking vote. Gate: 2023–25 strict overlap vs standings on identical races, paired CI. Leak control: use archived-page capture timestamps, never live pages; exclude any source whose archive date is after the race.
   b. **Structured announcements with dates**: confirmed grid penalties / PU changes announced before cutoff, driver substitutions, team upgrade declarations (FIA car-presentation submissions are published Thursday and are structured lists; check the timestamp against cutoff per round).
   c. **LLM extraction of text** only last, and only with blinded prompts (no driver names in the question, outcomes masked) plus a shuffled-date control, because a pretrained model remembers 2023–25 results.
   Success at this stage is a ≥5-pt paired gain over standings on 2023–25. Then the one 2026 look.

**3. R4 — re-scope to post-qualifying (user decision; does not reach 75% either).**
Grid-known accuracy is 0.65–0.69 and E17's everything-known oracle is ~0.69, so 75% is above that ceiling too. The only untested target-weekend input is the **sprint race result** on sprint weekends (~6/yr): a second observation of race-trim pace. Worth one bounded test if R4 is chosen; expect a few points on those races only.

**4. Re-scope the target (user decision).**
Options the evidence supports: (i) a probabilistic target (podium Brier / log-loss, where the learned model already beats standings, E64); (ii) a 2026-only claim, where the pace oracle is 0.789 and E8's in-season curve was rising, acknowledging a 19-race sample cannot establish 75%; (iii) ">=2 of 3 drivers" (0.70, E88) stated explicitly as a different metric. Do not present any of these as meeting the pre-registered goal.

## 7. How to run an experiment here (contract, do not skip)

1. **Pre-register** one hypothesis, ≤3 variants, the gate that stops it, and the exact cutoff, as a new `### E154 — ...` entry in `EXPERIMENTS.md` *before* running. Number continues from E153.
2. **Tune on inner only.** Compare on outer. Report both, plus plain standings and frozen `ens` on identical `(season, round)` sets, with `W.paired_boot` CIs and the label-shuffle null (`experiments/a_yardstick.py` is the reference).
3. **Pre-weekend path:** `walk_forward(..., tie=None)` with `W.assert_pre_weekend()` active. Never pass grid/quali/y_pace/y_q columns. The guard raises if you do (E30/E31 incident).
4. **Leak test every new source.** Write a `tests/test_<source>_leak.py` that scrambles/deletes target-weekend-and-later rows and asserts features are bit-identical, then mutates an earlier row and asserts a change. Then run all guards:
   ```sh
   PYTHONPATH=. .venv/bin/python -m tests.test_no_leak
   PYTHONPATH=. .venv/bin/python -m tests.test_preweekend
   PYTHONPATH=. .venv/bin/python -m tests.test_profile_leak
   PYTHONPATH=. .venv/bin/python -m tests.test_guard
   PYTHONPATH=. .venv/bin/python -m tests.test_metrics
   ```
5. **Promote nothing** that does not beat standings on outer by a paired CI excluding zero. Correlation gains alone do not count.
6. **Log the verdict** (strict overlap first, then secondaries, bugs found, what stops the route) and commit + push the batch with a one-line imperative message, matching the existing history.
7. **Audit habit:** any number that looks too good (>0.70 on dev) is a leak until proven otherwise. It has happened once (E30/E31) and nearly once (E132 Friday cutoff).

## 8. Repo map

- `run.py` — walk-forward harness (`--pre` for pre-weekend). `predict.py` — next-race prediction.
- `f1/features.py` — feature table and `walk_forward`; `f1/data.py` Jolpica; `f1/ff1.py` FastF1 race pace; `f1/kalman.py`, `f1/market.py`, `f1/energy.py`, `f1/laps.py`.
- `experiments/` — one script per experiment, prefixed by avenue (`a1_`, `b3_`, `c1_`, `r1_`, `s1_`…). `experiments/common.py` shared helpers.
- `tests/` — leak, guard, metric tests listed above plus per-source leak tests.
- `data/results.csv`, `data/racepace_ff1.csv`, `data/laps.csv` committed; `data/parts/*.csv` are extracted feature sources; `data/fastf1_cache` is the offline FastF1 cache (3.6 GB, partial 2026).
- `results/` — generated summaries. `RESEARCH_AVENUES.md` — per-avenue status with E-number updates.
- Env: `.venv` Python 3.12, `requirements.txt` pinned. FastF1 ≥ 3.9 needs F1 credentials via `scripts_run_ff1.sh`; Mexico 2026 race laps may still be unloadable (E94 note).

## 9. Status at handoff (updated 2026-11-07)

Research is **paused pending the user's decision on §6 items 2–4**. Last logged experiment: **E155**; start at E156. Frozen deliverable is still `ens` (E8). Two commits this session (`60e5163`, `cf2bd2e`) are local and unpushed.

**Final 2026 look protocol (one look, not yet taken for any new candidate):** freeze the §6-item-1 rule and write its E-entry *before* running `run.py --pre` on 2026; report strict overlap for candidate, `ens`, standings, and null on all 19+ rounds; log that 2026 was previously exposed in E6/E8/E86/E93. Do not take the look until the user has chosen a direction, so that the look is spent on the best candidate available.

Manifold prospective captures for 2026 rounds 20–23 run locally at Thursday 00:05 UTC (2026-11-05 already fired; 11-19, 11-26, 12-03 pending). Score per E148 as a comparator only; check `data/parts/` for the round-20 JSON.
