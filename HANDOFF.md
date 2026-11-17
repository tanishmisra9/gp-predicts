# HANDOFF — direction for the agent taking over gp-predicts

**Current brief, updated 2026-11-16:** the user changed the primary objective to **high exact winner accuracy**, accepting lower second/third-place accuracy, and explicitly resumed research after E168. The goal is **active**, not paused. Pre-weekend restrictions, walk-forward evaluation, betting exclusions and Kalshi prohibition remain. Do not interpret the old podium programme's closure as a request to stop winner research. The user clarified on 2026-11-16 that success means **75% exact winner accuracy across all races**. Report all-race exact top-1 winner rate, never substitute winner-in-top-2 coverage, selective race coverage or per-driver binary accuracy. Report strict podium overlap secondarily.

**Baseline clarification (E184):** the frozen selector labelled `standings` sums prior race points only, omits sprint points, and uses d10 finishing form for ties. Do not call it exact official championship standings. Preserve it for historical comparisons; E184 tested sprint-inclusive totals on inner data: model 32/61 and points selector 29/61, no predicted winner names changed from their raw references. Gate failed, no outer run or official countback claim.

**Winner results:** E168–E179 tested binary win logistic, winner-only conditional choice, team-win × conditional driver choice, explicit past win rates, fixed boosted trees, shorter training half-life, portable teammate pace, pre-season testing for early-round winners, and best-car team history. Outer standings and frozen `ens` **32/70 = 45.7%**; highest observed development candidate **34/70 = 48.6%** (past leading share, E179), with paired intervals including zero; the user target is 75%. E175's portable variants each hit 33/70; E176 testing rules each 32/70; E177 best-car history 31/70. E178 selected C=.003 using inner winner counts only, then tied outer baselines at 32/70; the entire inner C grid tied at 32/61. E179 past-leading share hit 31/61 inner and 34/70 outer, so its two-race outer gain is exploratory and inner loses one to the winner baseline. No promotion or new 2026 evaluation. E172 audits 20 randomized-winner seeds because one null run can be misleading. Source-scramble, model target-label mutations, independent saved-ranking counts and standard guards passed. Per-race artifacts are in `results/e169_*` through `results/e179_*`; reload prediction scores with `float_precision="round_trip"` to preserve ties exactly. User explicitly requested iterative commits; checkpoint each completed experiment rather than accumulating a large uncommitted batch.

**Winner-specific direction (E174, hindsight diagnostic):** frozen `ens`'s 38 outer winner misses split into **28 wrong team / 10 wrong teammate**. Perfect winning-team knowledge with the existing driver choice gives 54/70 = 77.1%; perfect teammate choice within the currently chosen team gives 42/70 = 60.0%; actual retirers alone 34/70 = 48.6%; actual median race pace alone 39/70 = 55.7%. These use target-race truth and are not deployable forecasts or proven bounds. Prioritize evidence that predicts winning-team/contender execution. Portable teammate pace and early-round testing are now tested for the winner objective (E175/E176), and best-car history failed E177; do not repeat those formulations or tune them on outer results. Winner-specific regularization is now closed (E178): the fixed inner-only grid yielded no winner improvement and its locked choice tied outer. Do not extend the C search. Past laps-led share is now tested for wins (E179): no demonstrated paired gain and far below 75%. Do not tune lead-history variants on the inspected outer races. E180 completed a three-paper primary-method audit; see `WINNER_METHOD_AUDIT.md`. None establishes 75% exact-winner pre-weekend accuracy. E181 tested the causal joint continuous-qualifying/rank-likelihood driver–constructor Laplace filter: 30/61 inner winners versus E168 32/61, winning-team count tied at 37/61, winner log-loss worse (2.0775 vs 1.5648). Its inner gate failed; no outer scoring. The fixed formulation is closed, with no tuning of process variances or likelihood scales on these results. E182 source audit verified raw constructor IDs split main feature histories; E139 has a lineage map but the main builder does not use it. New alpine/aston_martin names each omit 59 earlier events at 2021 R1. This is a modelling assumption to test, not an automatic production fix. E183 tested the fixed lineage mapping: 32/61 inner winners, identical hit/miss sequence to raw E168, equal team count 37/61, slightly worse winner log-loss (1.5679 vs 1.5648). Inner gate failed; no outer run, no promotion or lineage tuning. E184 has now audited the omitted prior sprint points; including them changed no inner winner names. No tuning that formulation. E185 audited nine existing pace measurements on all 61 inner races. Median hindsight gets 44/61 winners; none reaches the registered 46/61 alignment gate. Clean-air matched past-only pace gets 35/61 versus median 31/61 and E168 32/61, nominal paired gain over median [0.0164,0.1311], but no multiplicity-adjusted or paired established gain over the winner baseline. All direct pace-ranking gates failed; no outer scoring or classifier follow-up from these rules. Do not tune corrections/decays or relax the gate post-hoc. E186 tested driver-specific win conversion residuals among measured front-three cars, distinct from E155 team-average rank execution. Residual correlation .3494 on 180 opportunities is descriptive, but the fixed correction changes no winner hits (44/61 for all hindsight rules), worsens mean log-loss (1.0643 vs 1.0601), and fails its measurement gate. All scores know actual target pace and are not pre-weekend forecasts. No feature, outer run or residual-parameter tuning. E187 context audit found a final-calendar timing leak in legacy E69 and E73b all-block variants (season final max round determines earlier rounds-left/title-pressure/late-season fields). Removed these columns; old affected scores are explicitly timing-invalid, no rerun. Frozen model unaffected. E72 share fields are existing-data transformations, not new team-order observations; no new winner fit justified by the audit. Next inspect E99 fused team strength at the winning-team decision: its small full-order correlation gain does not prove a top-team gain. Audit formulas/artifacts before any registered inner-only measurement, without reusing inspected outer scores to select a variant. Do not weaken the 75% objective or use selective races. No reopening E181. Full-sample smoothing, target qualifying/grid and finishers-only evaluation are inadmissible. Constructor continuity and winner-relevant pace measurement are lower-priority audits; do not reopen existing negative variants without a materially different measurement. No new forecast scoring occurred in E180. Its initial broad search exposed incidental 2026 snippets without modelling use, logged explicitly; 2026 remains restricted and not pristine.

The original assessment below was written 2026-11-07 after reading `EXPERIMENTS.md` (E0–E153), `RESEARCH_AVENUES.md`, and `README.md`. Its **≥ 0.750 strict podium overlap** objective is historical and was not reached; the current winner brief above supersedes its metric priority and stop instructions.
This file tells you where the evidence stands, what not to repeat, and where the remaining headroom is.
Read it before touching code. Read `RESEARCH_AVENUES.md` next for the per-avenue detail.

## 1. The task, exactly

- **Metric:** `mean over races of |predicted top-3 ∩ actual top-3| / 3`. Pre-registered in E0. Do not loosen it; looser variants (≥2 of 3, top-5 coverage, per-driver binary) are reported as secondaries only (E88).
- **Setting:** predict before the target weekend. No practice, qualifying, sprint, grid, telemetry or labels from that weekend. Cutoff is Thursday 00:00 UTC of race week (earlier if the weekend starts earlier).
- **Evaluation windows:** inner = 2020–22 (61 races, tune here), outer = 2023–25 (70 races, honest comparison). Both have been inspected many times; call new outer results development evidence, not a holdout.
- **2026 is restricted.** It was looked at in E6, E8, E86, E93. Do not inspect 2026 labels or residuals during development. One final look only, logged.
- **Data rule from the user:** *no outside documents.* No FIA reports, Pirelli previews, articles, interviews, LLM-extracted text. Structured historical data (Jolpica, FastF1, OpenF1, Open-Meteo) only. Kalshi API is off-limits as of E151. If you believe 75% needs outside documents, say so to the user; do not quietly ingest them.

## 1b. Data-source constraints (user, 2026-11-07; supersedes anything below that conflicts)

- **No betting or prediction-market involvement of any kind.** No Kalshi, Polymarket, Manifold, Betfair, bookmaker odds, tipster/betting-site picks (Racing Post, Betfred, Oddschecker, etc.), or any feature derived from them. The work must be purely analytical / ML on racing data. The market comparators in E132/E147/E148/E149 are historical records only; do not extend them, and do not score the scheduled Manifold captures as a model input.
- Outside non-betting documents are allowed (the earlier "no outside documents" rule was lifted), but the result must still be an analytical model, not an aggregation of someone else's picks.
- Because of this, §6 item 2 R5(a) (tipster picks) is closed and the "market price" evidence of 0.722 (E132) is not a route to 75%.

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

**2. R5 — outside-documents rule LIFTED by the user on 2026-11-07; R5 chosen. Status: source gate logged as E157 (tipster archives are mostly post-qualifying, so they fail the Thursday cutoff; see E157 before building anything).**
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

## 9. Status at handoff (updated 2026-11-07, after E159)

**The single 2026 look has been SPENT (E159): candidate 0.561, `ens` 0.544, standings 0.526. Do not take another. The 0.750 pre-weekend goal is closed; remaining work is reporting, not research.**

_Older text:_

Research is **paused pending the user's decision on §6 items 2–4**. Last logged experiment: **E155**; start at E156. Frozen deliverable is still `ens` (E8). Two commits this session (`60e5163`, `cf2bd2e`) are local and unpushed.

**Final 2026 look protocol (one look, not yet taken for any new candidate):** freeze the §6-item-1 rule and write its E-entry *before* running `run.py --pre` on 2026; report strict overlap for candidate, `ens`, standings, and null on all 19+ rounds; log that 2026 was previously exposed in E6/E8/E86/E93. Do not take the look until the user has chosen a direction, so that the look is spent on the best candidate available.

Manifold prospective captures for 2026 rounds 20–23 run locally at Thursday 00:05 UTC (2026-11-05 already fired; 11-19, 11-26, 12-03 pending). Score per E148 as a comparator only; check `data/parts/` for the round-20 JSON.

## 10. Assessment (2026-11-07, after E158): requirements, preferences, the field, and direction

### 10.1 What the requirements actually are, traced through the session
| requirement | where set | effect on feasibility |
|---|---|---|
| ≥0.750 strict podium overlap (mean over races) | E0, pre-registered | fixed; never loosened |
| **post-qualifying** setting (grid known) | E0–E6, the original | model 0.649 (2026) / 0.69 (dev) vs grid 0.614; everything-known oracle ~0.69 (E17) |
| **pre-weekend** setting (no target-weekend data) | **E7, user request**; became the headline | model 0.544 (2026) / 0.59 (dev); pace oracle 0.746; team-order oracle 0.743 |
| no outside documents | user, pre-E101 | lifted 2026-11-07 |
| no betting / markets / tipsters, purely analytical ML | user, 2026-11-07 | closes the only pre-weekend source ever above 0.70 (E132) |
| honesty over score: pre-registration, inner/outer, leak tests, no metric loosening | user + log throughout | non-negotiable; also the project's real strength |

The E7 switch to pre-weekend is the decisive one: it moved the target from ~6 points above the best model to ~16 points above it and **above every oracle we can build**. That was a reasonable research question ("does accuracy accumulate across a season?"); it was answered (no, except weakly in reset years, E8/E154). The 75% target was not re-examined when the setting changed.

### 10.2 What other implementations achieve (web review, 2026-11-07; self-reported, not verified)
- Headline "78–90% podium accuracy" repos are **per-driver binary** accuracy, where predicting "no podium" for everyone scores ~85% (one repo says so itself). Our equivalent is 0.882 vs 0.850 always-no (E88). Not comparable to strict overlap.
- Repos that report an honest per-race number land at **62–68% top-3 overlap, post-qualifying** (e.g. "62.2% average overlap 2023–25", "68% podium hit rate", "68.8% average correct podium positions"). Our post-quali 0.65–0.69 is in the same band, and our grid baseline 0.61–0.67 explains most of it, as their own grid baselines do ("front three on the grid gets about 2 of 3 right").
- The one claim above our ceiling (9/24 exact podium sets in a season) comes without a time split or leak test; our exact-set rate is 0.26 post-quali. Treat as unverified.
- Academic work (theses, preprints) reports Brier/log-loss/AUC and concludes race-time factors explain most of the variance, which is our E93/E22 finding.
- **Nobody credibly reports ≥75% strict podium overlap pre-weekend.** Nobody else reports oracle ceilings, inner/outer splits, or mutation-checked leak tests; that methodology is the differentiated output of this repo.

### 10.3 Direction
1. **Stop searching for 75% pre-weekend with analytical methods.** 158 experiments, every permitted route gated and closed, and the measured ceilings sit below the target. Further sweeps would be variant search; the log already says so repeatedly.
2. **Spend the single 2026 look on the frozen candidate** (`ens` + E156 testing rule for rounds 1–3) and record it as the final pre-weekend number. One run of `run.py --pre` with the rule; no tuning afterwards.
3. **Report the project's actual contribution**, which is stronger than any repo found: (a) the baseline the field under-reports (standings pre-weekend, grid post-quali), (b) the information-stage curve (E17), (c) the oracle ceilings (E10/E22/E56/E158), (d) error attribution (E93), (e) the leak incidents caught (E30/E31, E132 cutoff). Update README with the post-E100 rounds; this is also the LinkedIn story already drafted.
4. **If a target must be met, restate it as one the evidence supports and label it as a re-scope, not 75%:** post-qualifying strict overlap with a significant margin over the grid baseline (currently +3.5 pts, CI touching zero: not yet met), or podium Brier vs the standings prior (already met, E64).
5. **Do not** reopen markets, tipsters, FIA documents, telemetry scalars, or learner sweeps; all have logged negative or blocked results with reasons.

## 11. Direction to still reach 75% (2026-11-07; user: "I'm not giving up")

### 11.1 Why pre-weekend is stuck, in one chain (read before choosing a route)
Error budget pre-weekend (E93): ~41 points of misses = ~25 pts pace-order + ~16 pts race-day events. A **perfect** pace forecast removes the 25 and lands at 0.746 (E10); that is the whole pace route, and it is just under the target. The 16 race-day points (strategy, SC, starts, retirements, teammate order) have **no** pre-weekend predictor in any test (E16 weather oracle, E18 retirement model, E155 execution persistence 0.07). The E10 mapping is also steep: a pace forecast at rho 0.96 gives only 0.66, so incremental pace gains are nearly worthless until perfect. **Conclusion: 75% pre-weekend is above the information content of pre-weekend data.** Keep `ens` + testing rule as that setting's deliverable and do not spend more effort there.

### 11.2 The fact that reopens the goal
**In 2025 the grid baseline alone scored 0.750 on the pre-registered metric, post-qualifying** (E6: dev 2025 model .736 / grid .750; 2023 .667/.636; 2024 .667/.667). So the target *is* reachable on the exact metric in a low-chaos season once qualifying is known. Post-qualifying is the project's **original** setting (E0–E6), the one the field evaluates on, and relaxing E7's self-imposed pre-weekend restriction is not loosening the metric. The 2026 oracle with true pace is 0.789. **Route: go back to post-qualifying and build the best possible post-quali model.** The post-quali model has not been touched since E5 (frozen compact logit); every Round 2–8 learning was applied to pre-weekend only.

Honest odds: a typical season likely lands 0.70–0.73; 0.75 needs a season like 2025 or a real gain from practice race pace. Say so in any write-up.

### 11.3 Post-qualifying programme (pre-register each as E160+; inner 2020–22 / outer 2023–25; 2026 restricted again until a candidate is frozen)
**P1. Post-quali ensemble (cheap, do first).** Pre-weekend `ens` beat the lone logit by +3 pts outer. Post-quali never got an ensemble: test rank-avg(grid, standings, post-quali logit) and rank-avg(grid, logit), fixed equal weights, vs grid and vs the E5 compact logit. One day of work; expected +1–3 pts.

**P2. Properly measured practice race pace (the one untried input with a mechanism).** `data/laps.csv` holds only crude scalar FP gaps (best/top5/long-run median, 2023–26); E4/E17's "practice adds nothing" was tested on those. The E102/E111 machinery (green-flag, no in/out laps, same-lap field matching, compound and tyre-age correction, clean-air gap) was built for race laps and **never applied to FP1–FP3 of the target weekend**. Build it on the FastF1 cache for FP2/FP3 long runs (and FP1 on sprint weekends), fuel-corrected by stint position, compound-matched. **Gate (inner, before any podium model):** corrected FP race-pace rank must correlate with TRUE race pace better than the quali gap does (quali rho is the bar; if FP cannot beat quali as a race-pace measurement, stop). Then add as one feature to P1. Mechanism: quali is low-fuel single-lap; race pace is high-fuel degradation-limited; the gap between them is exactly the "podium without podium pace / lost pace in race" error classes.

**P3. Sprint result as a second race-trim observation.** On sprint weekends (~6/yr) the Saturday sprint is a real race on the same track with the same cars. Add sprint finish/pace rank as a feature for those races only; evaluate on sprint races separately and pooled. Few races, so this is a bonus, not a pillar.

**P4. Grid-position-aware structure.** Grid matters beyond pace (track position; overtaking difficulty by circuit). Test one fixed interaction: grid × circuit overtaking difficulty (from past races' positions-gained variance, past-only). E5 had `circ_grid_corr`; the interaction with grid was never in a post-quali model after E5.

**P5. Retirement-aware selection post-quali.** E18's retirement model was calibrated (14.7% vs 15.0%) but tested pre-weekend only. Post-quali, multiply P(podium) by P(finish) and re-select; grid penalties and known PU changes (FIA Friday docs are admissible post-quali) feed P(finish). Cheap once P1 exists.

**P6. Freeze, then prospective.** Freeze the best post-quali candidate on inner/outer, take one 2026 look, then run it **prospectively** on 2026 R20–23 and all of 2027 (second year of the regulation cycle; 2023, the analogue, was a dominant-team season). Prospective races are the only genuinely unseen test left; the 75% claim should be made on those, with the full denominator.

### 11.4 Stop rules
- P2 fails its measurement gate → the practice route is closed for good; P1+P4+P5 are what remain and will not reach 75% in a typical season. Report that honestly.
- Nothing in 11.3 is promoted on <5 pts paired gain at 131 races, same as before.
- Pre-weekend 75% stays closed regardless of P1–P6 results.

### 11.5 Programme status (2026-11-07, end of session)
| item | entry | outer / result | status |
|---|---|---|---|
| P1 post-quali ensembles | E160 | ens(grid,standings,logit) 0.667 vs logit 0.691 | failed; `compact_logit` stays the base |
| P2 corrected FP long-run pace | E161 | Spearman w/ true pace 0.03–0.29 vs quali 0.55–0.67 | failed decisively |
| P2b FP best lap as feature | E162 | six seasons: +0.0025 [0.000,+0.008] (one pick changed in 131 races); rank-average −0.051 [−0.087,−0.018] | gate not met; closed |
| P3 sprint result | E164 | −0.028 on 24 sprint races | failed |
| P4 grid × overtaking | — | already in COMPACT (`grid_x_corr`) | n/a |
| P5 retirement-aware | E163 | −0.005 outer | failed |
| oracle ladder post-quali | E165 | pace 0.738, DNF 0.724, **pace+DNF 0.771**, team order 0.776 | 75% needs BOTH perfect pace and perfect retirements |

**Where this leaves 75% post-qualifying:** the base is 0.69 outer; grid alone reached 0.75 in 2025. Every planned input has been tested; none moved the base. E165 shows the +8 pts of headroom to 0.77 require exact race pace *and* exact retirements, two quantities that E161/E163 show are not forecastable at the needed precision. Remaining honest moves: (1) finish E162 on all six seasons when the download completes; (2) P6, freeze `compact_logit` (+`fp_best` only if E162 clears its gate) and evaluate prospectively on 2026 R20–23 and 2027; a 2025-like low-chaos season is the realistic path to a 0.75 reading, and that is season luck, not model skill, and should be labelled as such.

## 12. Priority correction (user, 2026-11-07): PRE-WEEKEND is the headline. Post-quali (§11) is secondary.

**Pre-weekend ceiling, now fully measured (E166):** `ens` 0.586 outer → +true retirers 0.610 → +true pace 0.738 → +both **0.757** → true team order 0.743. 75% is inside the pre-weekend information space only with near-exact race-day pace *and* exact retirements.

**What "pursue pre-weekend" can honestly mean, in order:**
1. **Prospective evaluation of the frozen pre-weekend model** (`ens` + E156 testing rule) on 2026 R20–23 and all of 2027, full denominator, no tuning. This is the only unseen test left and a low-chaos season is the realistic way a 0.75 reading appears (standings alone reached 0.667 pre-weekend in 2025). Label any such reading as season variance unless the paired gain over standings is significant.
2. ~~Retirement causes from non-betting documents~~ **DONE, E167: labels recovered from Wikipedia (59 outer mechanical events), cause-specific hazard ties the generic one and changes no pick. Closed.** Original text: (Wikipedia race-result tables / FIA classification list retirement reasons that Jolpica lost for 2023–25). This makes E101's cause-specific hazard testable on the outer window. Hard bound: +2.4 pts even with perfect retirer knowledge (E166), so it cannot reach 75% alone; it is the one pre-weekend information gap that documents can actually fill.
3. **Pace side: no open route.** Hindsight persistent form caps team-order corr at 0.775 vs 0.760 achieved (E13); race-specific deviations do not persist (E10 r = 0.04), weather is worthless even when known (E16), telemetry profiles have identity but no forecast value (E127–E143), practice pace is noise (E161). Any new pre-weekend pace proposal must first show, on inner data, a per-race pace Spearman above 0.80 against true race pace; nothing has exceeded ~0.72.
4. Do not reopen: markets/tipsters (user rule), FIA PU/penalty docs (E158), learner/feature sweeps (E1–E92), reset-year blends (E154), execution residuals (E155).
