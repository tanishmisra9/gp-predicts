# Research avenues after E100

**Current winner research:** 75% exact pre-weekend winners across all races, with podium overlap secondary. E190 established cache-only start-of-lap traffic measurement feasibility (344 disjoint pairs on three fixed 2019 sessions). E191 completed the registered test: enough samples, but split-half correlation .2654 and temporal correlation −.1155 failed the gates; forecast MSE was worse than zero. No winner model or accuracy claim followed, and this fixed formulation is closed without tuning. The original podium-focused backlog below is historical.

**Audit correction (E188, 2026-11-16):** references below to E99/E100 fusion gains are historical and timing-invalid because indicators were scaled with full-sample standard deviations. A repaired 2018-calibrated inner winning-team probe reached 28/61 winners versus E168 32/61 and failed its gate. Do not use the old gain as evidence for a new route.

Written 2026-11-02 after reading all of `EXPERIMENTS.md` (E0–E100) and `README.md`.

**2026-11-16 correction (E188):** E99/E100 fusion figures cited below used full-sample indicator standardization. They are historical timing-invalid evidence, not an established causal forecasting gain. E188 registers a corrected fixed-2018 calibration; no automatic transfer of the old numbers.
This is a research backlog, not a new experiment result. No model was trained,
no results were refreshed, and no additional 2026 evaluation was performed.

## Goal and current evidence

**Primary target: at least 0.750 strict podium overlap**, averaged over races:
`mean(|predicted top 3 ∩ actual top 3| / 3)`. Predictions must be made before
the target weekend, using no practice, qualifying, grid, sprint, race labels,
or telemetry from that weekend. A single race has overlap 0, 1/3, 2/3, or 1;
the 75% target concerns the average, not an attainable single-race score.

The logged pre-weekend reference is approximately 0.58–0.61. E12 reports
`ens` 0.606 versus standings 0.598 over 131 races; the outer 2023–25 references
are 0.586 and 0.590. The frozen E8 2026 result is 0.544 over 19 races.
These are different evaluation windows, not interchangeable estimates.
At 0.60, reaching 0.75 requires about 0.45 additional correct podium picks
per race. No avenue below has demonstrated that gain.

E93 attributes 61% of errors to measured pace-order mistakes and 39% to
execution/attrition. Those are retrospective classifications, not evidence
that either group is fully predictable. E10's 0.746 pace oracle and E22/E56's
0.743 team-order oracle are results of particular measurements and decision
rules, **not proven upper bounds on all pre-weekend predictors**. Likewise,
E13's hindsight persistent-form comparison is not a universal statistical
bound. They justify skepticism, not a declaration of impossibility.

## User constraint: no outside documents

The user clarified: **“No outside documents.”** This plan interprets that
conservatively: do not use FIA reports, upgrade submissions, technical articles,
team/Pirelli previews, interviews, or LLM-extracted text as model inputs.
Those suggested avenues remain recorded below as excluded, not recommended.

Eligible work below uses the project's existing historical structured data
and permitted historical session feeds. Source existence does not establish
that every desired field or historical lap is available. Additional datasets
are separated as conditional candidates; no purchase, account creation,
licence acceptance, or restrictive scraping is authorized by this plan.

For future experiments, pre-register an exact cutoff before acquisition.
A conservative proposal is Wednesday 00:00 UTC of race week, moved earlier
if the official weekend begins earlier. This is a proposal, not an existing
project convention. The prediction input must be frozen before that cutoff;
earlier-weekend measurements remain eligible only from completed prior events.

## Why these directions, rather than another model sweep?

Three competing explanations remain plausible:

1. **The existing inputs miss useful physical signal.** Better measurement
   should improve independent-lap repeatability and next-race performance,
   before needing a more flexible podium model.
2. **Useful signal is already measured but poorly assigned to drivers or
   outcomes.** Matched teammate comparisons or a better attrition decomposition
   should improve relevant residuals after conditioning on standings/form.
3. **Much of the remaining variation is unpredictable with permitted data.**
   Cleaner measurements may improve correlations without changing podium
   selections, as E99/E100 already demonstrated.

The cheapest discriminator is a small, past-only measurement pilot, followed
by an ablation against the frozen baseline. The working assumption is that
many-lap measurement contains information absent from three fastest laps.
The tripwire is failure to improve both measurement reliability and held-out
forecast residuals; that should stop a large telemetry rebuild.

## First batch: highest priority eligible avenues

Priority is a judgment about information novelty and cost, not a forecast
of a particular accuracy gain. Every numbered item is untested **in the exact
form proposed here** unless explicitly identified as a follow-up.

### A1. Many-lap race-trim telemetry, aligned by position

- **Hypothesis/data:** Build corner-level car signatures from many clean laps
  in prior races, using raw car and X/Y position channels. Account for compound,
  tyre age, stint phase, traffic, DRS, and caution periods before pooling laps.
  Register circuit geometry using prior visits; do not obtain a target-weekend
  reference lap. Layout changes require an explicit missing-data fallback.
- **What is new:** E9/E20 used a fastest qualifying lap; E83–E85 used three
  fastest laps. Neither tested a many-lap race-trim profile. Position alignment
  is a measurement hypothesis, not a guarantee of accuracy.
- **First test / stop:** On a fixed inner-period sample, split laps within
  driver-stints and test independent-half repeatability, sample-size curves,
  and teammate agreement. Then test next-race team-pace residuals versus simple
  pace/form. Stop if added laps only reproduce overall lap speed or if the
  incremental forecast fails. Check raw sampling/interpolation sensitivity
  before interpreting acceleration or braking features.

**E126 update:** A many-lap, green-flag race SpeedST residual from 171 cached 2018–25 races was tested as one past-only team feature. It gave no strict podium or Brier gain, and its next-race pace-order correlation was slightly negative. Stop that scalar speed-trap route. A full multi-stint, X/Y-aligned corner profile remains a distinct, higher-cost measurement hypothesis; E113 only used five early-stint laps per driver.

**E127–E129 update:** The multi-stint X/Y-aligned profile now shows measurable 2023-to-2024 driver-shape persistence across Bahrain, Austria, and Monza: mean correlation 0.263 across 12 continuing drivers from six teams, with the true mapping first among 64 teammate swaps. The added midfield teams are weaker than the original front six. This is a reason to test incremental **future pace/P3–P4 information**, not to claim a podium lift; no predictor has used the profile yet. See `EXPERIMENTS.md` for the fixed extraction, permutation diagnostics, and the 2020–22 coverage limit.

**E130 forecast update:** On those 12 drivers at the three 2024 circuits, the prior-year profile's mean speed residual forecast race-pace order better than prior-year same-circuit race pace but worse than pre-weekend standings in every race. A fixed standings/profile rank average also lost to standings. Stop that scalar forecast; a shape-based forecast still needs a larger locked sample before it can support any podium claim.

**E131 shape update:** A fixed equal-distance lap-time projection (`-Δspeed / field_speed²`) emphasized slow bins without fitted weights; it also lost to standings in every available 2024 race and slightly trailed E130's mean-speed scalar. Do not search more profile summaries on the same three inspected races. Broader time-ordered coverage is required before an A1 podium test.

**E140–E142 broader-profile update (2026-11-03):** On five preselected new 2024-to-2025 circuits, the prior-year **team-aggregated** multi-stint X/Y speed residual beat pre-event 2025 team standings for future team pace order (mean Spearman 0.773 vs 0.638). A fixed equal-rank podium vote scored **0.800 strict overlap on those five races versus 0.733 for standings**, only one additional correct slot. The full 24-race 2025 replication fell to **0.694 versus 0.667 for standings** (50 versus 48 correct slots; paired interval crosses zero); the profile alone scored 0.472. Across the 23 races where pre-event team-standings pace correlation is defined, the profile trailed standings by 0.024 mean Spearman. Keep the frozen model; do not tune the vote on this inspected season. A separate time-forward cohort is needed for promotion.

**E143 independent-season update (2026-11-04):** On all 22 shared 2023→2024 circuits, the same prior-year team X/Y scalar scored 0.518 mean future team-pace Spearman versus 0.805 for pre-event standings on 21 defined pairs; its paired difference was -0.293 with a 95% interval entirely below zero. The fixed equal-rank blend also lost by 0.085. Stop the scalar route; the E142 podium lift does not justify promotion. No E143 podium vote was run after the pace gate failed.

### A2. Jointly correct pace for traffic, tyre age, and stint selection

- **Hypothesis/data:** Estimate comparable pace from prior lap timings, gaps
  to cars ahead, compound, tyre age, lap number, and caution flags. Match cars
  over common race windows; treat unknown fuel as uncertainty, not an observed
  variable. Retain how much clean evidence each estimate has.
- **What is new:** E94 tested several scalar pace summaries and fuel/tyre
  corrections separately. This proposal tests their joint interaction with
  traffic and which laps survive filtering, not another median/quantile sweep.
- **First test / stop:** Compare matched-window estimates with independent
  later stints and forecast the next event's pace. Test sensitivity to exclusion
  thresholds. Drop it if it only improves same-race fit, or reproduces E94's
  small correlation gain without improving podium probabilities/selections.

**E102/E111–E112 status:** The clean-air matched-lap pilot and the joint traffic/compound/tyre-age correction were completed on 2018–25 historical races. The joint measure improved next-race pace correlation only slightly over matched pace (outer +0.0074, paired interval crosses zero). Its equal-rank podium blend and one-feature logit blend both **tied standings at 0.5905 outer strict overlap**; the added logit feature did not improve outer Brier or log-loss. Do not repeat the same timing-only correction or tune its weights on these inspected races. Reopen A2 only for a genuinely new, validated measurement of traffic or stint state with independent future-race transfer.

### A3. Contextual driver tyre management

- **Hypothesis/data:** Estimate driver-minus-teammate stint degradation and
  late-stint pace, conditioned on compound, tyre age, track load, and traffic.
  Pool across all available historical drivers, with strong shrinkage and
  uncertainty for drivers with few comparable stints.
- **What is new:** E9/E11 included aggregate degradation/style features;
  E94 corrected pace for wear. Neither establishes whether repeated, matched
  stint contrasts identify a portable driver advantage.
- **First test / stop:** Test whether an earlier-stint driver residual predicts
  a later independent stint and future teammate order. Control for stop timing
  and team strategy. Stop if apparent skill disappears under matched contexts
  or fails to transfer across seasons/teams.

**E104/E145 status:** E104's matched-compound driver-minus-teammate degradation slope had near-zero next-race persistence and lost to standings on teammate order. E145 tested a distinct same-compound, field-relative **team** slope with 1,701 historical team-race measurements. Its past-only forecast slightly improved zero-baseline MSE in 2020–22 but **tied zero in 2023–25** (paired interval crosses zero; forecast/outcome correlation 0.029). Neither measurement supports a podium feature. Revisit only with a new, independently validated stint-state or tyre-construction signal.

**E150 same-circuit update:** Adding a team's earlier slope at the exact circuit to E145's global forecast worsened degradation MSE in both inner and outer periods, despite a small outer correlation signal. The fixed measurement gate failed before a podium model. Do not tune the shrinkage on these inspected races; see E150.

### A4. Mechanical failures as distinct, exposure-based hazards

- **Hypothesis/data:** Separate mechanical retirement, collision, and other
  non-finishes in existing official historical status data. Use prior completed
  racing laps/distance, recent mechanical failure type, team/supplier history,
  and time since failures. Include right-censored exposure for finishers.
- **What is new:** E5/E18 tested generic retirement rates/models; E15/E22
  identified retirement-related headroom. Cause-specific exposure is a narrower
  hypothesis than another rolling DNF percentage.
- **First test / stop:** Audit status categories and test out-of-time mechanical
  failure Brier/log-loss against a smoothed team baseline, then test incremental
  podium impact. Sparse front-runner failures may make this unidentifiable.
  Distance since a failure is **not** the mileage of an installed engine; do
  not invent component inventories without documents or component identifiers.

**E101/E136 status:** The exposure-based mechanical hazard was attempted, but the official 2023–25 result rows name **zero** mechanical failures and use 153 generic `Retired` statuses. FastF1's result `Status` is joined from the same Jolpica source, so it does not repair the cause labels. The cause-specific and generic survival adjustments made identical podium selections. Park A4 until a permissible independent structured cause source exists; do not relabel generic retirements as mechanical negatives or infer engine inventory from elapsed laps.

### A5. Teammate order from matched, comparable race segments

- **Hypothesis/data:** Use same-race, same-compound, similar-age, clean-air
  teammate comparisons, with uncertainty and shrinkage, to estimate driver
  advantage. Keep race pace separate from qualifying strength.
- **What is new:** E21's dedicated teammate model was worse than `ens`;
  E80's two-layer Kalman also stalled. Reopening this is justified only by
  materially cleaner paired observations from A1/A2, not a new classifier.
- **First test / stop:** Beat the existing implied teammate ordering on inner
  data, then evaluate podium errors specifically where both teammates contend.
  Stop if overall head-to-head gains come from irrelevant midfield pairs.

### A6. Pre-season testing: broader coverage and comparable long runs

- **Hypothesis/data:** Follow up E86/E87 with already available testing laps:
  distinguish compounds, stint ages, and comparable running windows; carry
  uncertainty rather than treating an absolute test time as race pace.
- **What is new:** This is an extension of a positive avenue, not an untouched
  discovery. E86's simple long-run measure was weaker than best-lap pace, so
  complexity must earn its place. E87's podium sample was small.
- **First test / stop:** Freeze one simple blend using eligible 2020–22 data
  where available, then evaluate early rounds of 2023–25 and report coverage.
  Use missing-data fallbacks, not complete-case headline selection. Do not tune
  using E86's already-inspected 2026 signal. Stop if benefit depends on one year
  or vanishes after matched coverage; this avenue affects few races per season.

**E124 update:** The existing 2023–25 testing cache also supplied complete pre-season lap counts. Lower team mileage had weak, inconsistent association with rounds 1–3 all-cause DNFs and underperformed the past team DNF feature. This reliability proxy should not be promoted to a podium selector; it does not alter E86's distinct pace finding.

### A7. Actual failure warnings in earlier-session behaviour

- **Hypothesis/data:** Where historical coverage permits, summarize repeated
  abnormal speed loss, shortened running, and unexplained session non-running
  from completed prior weekends. Test whether they add to mechanical hazards.
- **What is new:** E97 measured a normal energy-deployment signature, not
  unusual within-car changes preceding a mechanical failure.
- **First test / stop:** Predefine anomalies relative to the car's own past,
  label missing feeds separately, and test next-event failure enrichment.
  Traffic, lift-and-coast, deliberate programmes, and sensor gaps are competing
  explanations. Stop if removing them destroys the association.

**E120 update:** A fixed short-practice-running indicator from completed prior weekends did not improve all-cause retirement calibration or strict podium overlap. The local FastF1 practice archive lacks 2019/2022 and most of 2021; OpenF1 fills 2023–25, but the 2020–22 inner period cannot validate this particular feature. Do not treat missing feeds as failure warnings.

### A8. Near-podium pace gaps and uncertainty

- **Hypothesis/data:** Use A1/A2's uncertainty estimates to forecast each
  contender's chance of beating the third/fourth-place boundary, rather than
  rewarding only full-field rank correlation. Focus diagnostics on P2–P6.
- **What is new:** E99 improved full-field team order without improving podium
  overlap in E100. This tests whether improved information reaches the relevant
  boundary; it is not evidence that reweighting old features alone will help.
- **First test / stop:** Measure changes in podium probabilities, third-versus-
  fourth discrimination, and strict overlap with identical candidate coverage.
  Reject variants whose gains occur only in ordering backmarkers. With correct
  marginal podium probabilities, selecting the three largest already maximizes
  expected overlap: a joint simulator or diversity rule alone cannot improve it.

## Second batch: eligible but lower priority

| ID | Avenue and what differs from the log | Smallest useful test / main reason to stop |
|---|---|---|
| B1 | **Collision risk conditional on probable neighbours.** Estimate driver-pair contact risk from prior incidents and an entirely past-only forecast of starting order. E41 measured average start gains; E18 used broad retirement risk. | First audit whether existing data identifies incident partners and exposure. Do not infer fault from status alone. Compare with simple driver/circuit rates; stop for sparse pairs or no incremental calibration. Never condition on the actual target grid. |
| B2 | **Overtaking/defending conversion conditional on pace advantage.** Estimate how drivers convert comparable pace gaps into passes or retained positions using earlier lap charts; distinguish pits, cautions, and retirements from passes. | Validate inferred passes on a small audited sample, then predict future conversion residuals. E9/E11 already used execution/strategy summaries: this needs cleaner event-level information, not another finish-minus-pace feature. Stop if timing data cannot distinguish causes. |
| B3 | **Strategy exposure rather than average strategy quality.** From prior stints estimate pit-loss, traffic rejoin, and undercut sensitivity by circuit/compound; combine with a pre-weekend distribution of plausible race states. | Require an ablation over E9/E18/E75-style strategy/simulation inputs. Do not learn from the target race's actual safety-car timing or pit decisions. Stop if a complex simulation merely reproduces the old podium ranking. |
| B4 | **Supplier-wide mechanical failure clusters.** Test whether recent same-cause failures across cars sharing a supplier predict another failure, beyond each team's own risk. | Compare A4 with/without the cluster feature. E9/E11 already pooled PU reliability, so only time-local, cause-specific residual information qualifies as new. Stop if driven by one supplier-year or retrospective cause reclassification. |
| B5 | **High-speed/slow-speed limitations measured in race trim.** Revisit car/track compatibility only after A1 demonstrates repeatability; use measured corner responses rather than coarse scalar circuit similarity. | Forecast next-race residual pace using train-only geometry and shrinkage. E10/E14/E74/E79/E98 are strong negative evidence; stop before model tuning if the new measurement still lacks persistence or incremental forecast value. |
| B6 | **Driver adaptation after a team switch.** Use historical transfers to distinguish portable skill from temporary adaptation in matched teammate/stint measurements. | Backtest complete historical transfers with time-ordered training. E70 already tried rookie/experience flags and E80 a driver offset. Stop if too few independent transfers, or gains reduce to those existing features. |
| B7 | **Tyre construction/compound changes visible in historical structured data.** Track season-specific compound identities and observed stint response rather than treating C3 or 'medium' as physically identical forever. | Verify historical identifiers and change dates are available before cutoff; otherwise mark unknown. Test generalization across seasons. E98 already used historical hardness, so a hardness flag alone is a repeat. |
| B8 | **Forecast qualifying variability from earlier sessions.** Estimate each contender's distribution of qualifying performance and integrate its consequence for podium probability, using no target-weekend qualifying data. | Compare with direct podium forecasting using exactly the same past inputs. E17 shows actual qualifying is valuable, not that it is forecastable. This rearranges existing information and has low priority unless better measurements or new supervision improve forecast calibration. |
| B9 | **Thermal and wind sensitivity from matched historical running.** Estimate car-specific pace responses within previous events, with race/track controls, rather than a broad wet-skill flag. | First check that the interaction repeats in a later event. E16 found no gain even with true weather, so this is a low-priority diagnostic. Any target-event weather input would require a forecast issued before cutoff; actual target weather is forbidden. |

**E139 adaptation update (2026-11-03):** An earlier-transfer, shrinkage-adjusted forecast of switchers' teammate-relative pace in entries 6–10 barely improved selected-window MSE outer (0.46712 to 0.46595; paired interval spans both directions) and tied a pre-event points comparator at 8/11 faster-teammate calls. Stable-teammate/pace-coverage selection is retrospective, so this is a narrow measurement gate. Do not add an adaptation podium feature without independent later transfer evidence; see E139.

**E121 strategy update:** A fixed team-specific rate of costly pit-lane times from earlier races did not forecast the next costly stop better than a past global rate. This stops that tail-risk statistic before podium modelling; a validated stop-by-stop undercut/rejoin measurement remains untested.

**E152–E153 passing update:** Clean, green, close adjacent-car comparisons yielded 14,467 development opportunities and a small past-driver conversion Brier gain over a global rate, passing a prespecified measurement gate. Adding that one history feature to the frozen podium logit then scored **0.581 outer strict overlap**, below frozen `ens` **0.586** and standings **0.590**, with no calibration gain. Stop this passing-rate feature; the user requested a pause after E153. See the experiment log.

**E125 strategy update:** Relative first-stop timing from earlier races showed only a small rank association and worsened timing MSE versus a zero team-specific deviation in every 2023–25 season. Do not use that past team timing average to drive a race simulator. It does not measure a validated undercut/rejoin response.

**E144 undercut gate:** A fixed close, green-flag, adjacent-car first-stop definition yielded only **57 opportunities in 2020–25**, with no team reaching ten. It failed the registered 100-opportunity/five-team feasibility threshold, so no team-rate forecast or podium feature was fitted. The observed 60.8% conversion rate over all 2018–25 eligible pairs is selected and confounded by tyre offsets, traffic, and strategy; it is not a repeatable team skill estimate. See E144.

**E123 source update (2026-11-02 local time):** B1's pair-contact labels are unavailable from the structured OpenF1 race-control fields and the local 2018–25 FastF1 cache; its free-text message is excluded by the no-outside-documents rule. B9's matched lap/weather response is unavailable from the same cache, and the local weather table has only race-level summaries. Park both until an eligible structured source exists. See `EXPERIMENTS.md` for the exact gate and references.

**E196–E197 weather-source update (2026-11-16):** E196 confirmed the three fixed 2019 pilot feeds require authentication (primary 401, mirror404). E197 used the project's existing authorized account/helper and obtained 373 minute samples covering all 3,151 clean laps, with backward-only weather matches. The fixed source gate passes: Hungary/Italy have ≥3°C p95−p05 track-temperature variation and ample dry laps; Spain has 2.7°C. B9 is reopened for a separately registered, confound-controlled identifiability/repeatability test. No thermal-response model, winner score or forecast-weather input exists yet. Acquisition provenance remains account-authenticated even when these files are subsequently read from cache; no target-weekend weather is allowed. B1 partner labels remain unavailable.

**E198 thermal-support update:** the fixed E197 pilot fails the next design-only screen: 0/20 Spain, 15/20 Hungary and 0/20 Italy drivers pass full and both disjoint weather-block halves, short of two events with ≥10 passing drivers. Cubic race progression, stint, compound-age, position and close controls leave median full residual temperature RMS 0.156/0.805/0.406°C. No lap-time response coefficient was fitted. Close this formulation without removing controls, lowering thresholds or selecting Hungary alone. Source data remain usable for genuinely distinct registered questions; no winner-accuracy gain or proof that all thermal effects are absent.

**E199 leader-pressure update:** all21 cached 2019 timing streams support fresh, backward-only leader/chaser gaps for 993/1,049 clean leader laps. Only two driver-stints contain five close and five comfortable observations against a common pursuer/compound, so the registered contrast gate fails and no pace response is fitted. This closes that fixed leader-management formulation. Stream IntervalToPositionAhead is a distinct candidate source for an audited following-exposure measurement; earlier E102/E111/E191 used crossing-time proxies. Audit update freshness/coverage and material disagreement before any new response fit, without retuning closed methods.

**E136 reliability-source update (2026-11-03):** FastF1's race-result `Status` is joined from Ergast/Jolpica, not an independent F1 timing-feed cause field. Cached 2023 Bahrain and 2025 Australia retirements were all generic `Retired`, matching the official results table. A4/B4 remain parked until a permissible independent structured cause source is found; see E136.

**E133–E134 grid-risk update (2026-11-03):** Earlier driver-specific grid errors had a small within-race signal for forecasting a top-six contender's qualifying setback, but their out-of-time Brier gain over a global variance was tiny and race-paired uncertainty included zero. A single fixed Gaussian top-six-probability vote then fell to **0.5714 outer strict overlap**, versus standings **0.5905**. Stop B8 distribution votes on these repeatedly inspected races; a larger grid simulator is not justified by this evidence.

## Conditional structured-data avenues

These are structured datasets rather than document extraction. Public,
unauthenticated APIs can be checked under the user's instruction to continue
experiments; accounts, purchases, licences, and restrictive scraping still
need separate approval. Coverage and usefulness must be verified before use.

| ID | Candidate | Required feasibility check and first experiment |
|---|---|---|
| C1 | **Liquid historical podium odds**, including Betfair if accessible on acceptable terms. This differs from E61/E62's illiquid Polymarket quotes. | Establish that actual F1 podium markets, timestamps, matched volume, and usable spreads exist before the prediction cutoff. No closing/pre-race prices that incorporate the weekend. Ask before accounts, purchases, licences, or restrictive scraping. Benchmark market-only strict overlap against standings before any blend; do not substitute winner odds for podium probabilities. |
| C2 | **F1 Fantasy prices and pick rates**, captured before the weekend. | Verify historical snapshots and permitted access first. Prices/popularity may merely repackage previous results or fan preferences. Ownership at a game deadline after qualifying is inadmissible. Test incremental value conditional on standings and recent form; abandon if it only reconstructs them. |
| C3 | **Structured tyre allocations and specification changes**, without parsing external previews. | A timestamped permissible feed/export is required. Test team/driver compound-response interactions, not just circuit hardness already tried in E98. No target-weekend used-set counts. This is distinct from the excluded Pirelli-document route. |
| C4 | **Structured circuit changes**, such as resurfacing, geometry, DRS-zone or pit-lane changes announced before cutoff. | Verify a permissible structured source and historical availability. Test one prespecified interaction with past measured tyre/drag sensitivity. Do not backfill modern track metadata into older races. If only articles/event notes provide it, it is excluded by the document constraint. |
| C5 | **Cross-series driver evidence for newcomers.** Compare teammate-relative stint and qualifying performance before F1 using a permissible structured source. | Evaluate on historical entrants, separately from established drivers. No such source was verified for this task. Stop if domain shift or low podium exposure makes it irrelevant. This differs from E70's experience flags but likely has limited aggregate upside. |

**E118 access update (2026-11-02):** Betfair historical exchange files require a registered account and purchase; public material did not establish F1 podium-market coverage or usable pre-weekend volume. F1 Fantasy's terms prohibit automated extraction from the game. C1/C2 remain conditional and supplied no model input or backtest. See `EXPERIMENTS.md` for primary-source links and the strict baseline.

**E132 market update (2026-11-02):** Kalshi's public API provided timestamped 2025 driver-podium bids and asks without an account. At a fixed Thursday 00:00 UTC cutoff, market-only top-three picks scored **0.722 strict overlap versus standings 0.685** on the same 18 available races (paired difference +0.037, 95% CI [0.000,+0.093]). There are no 2020–24 Kalshi podium markets in this series, and six 2025 races lack usable pre-cutoff quotes/markets. This is a small, repeatedly inspected development subset, not a new frozen model or evidence for 0.750. Betfair's own guide also clarifies that its Basic historical tier is free but lacks volume; account access and F1 podium coverage remain unresolved. See E132.

**E137/E146 prospective update (2026-11-04):** Fixed Thursday 00:00 UTC cutoffs and immutable capture scripts now cover the four remaining 2026 races (Brazil and rounds 21–23), with local runs scheduled five minutes after each cutoff and before FP1. The public Kalshi series currently exposes Brazil only; Las Vegas, Qatar, and Abu Dhabi are pre-registered as unavailable if no matching market and three valid quotes exist by their cutoffs. No prospective market picks or race outcomes have been scored yet. See E137 and E146.

**User scope update after E150:** The user asked to steer away from the Kalshi API. The four saved one-time jobs now run only E148's Manifold capture; the Kalshi prospective plans in E135/E137/E146 are superseded. Historical Kalshi experiments remain logged but are not a source for future acquisition. The model and 0.750 strict gate are unchanged. See E151.

**E147–E149 market update (2026-11-04):** Timestamped Manifold bets supplied 14 usable 2024 races, but market-only strict overlap was **0.524**, below same-race standings **0.548** and frozen `ens` **0.571**. Do not promote it. Kalshi's saved 2025 prices still scored **0.722** versus standings **0.685** on 18 races; a new probability diagnostic found market Brier **0.06930** versus frozen logit **0.07111** on 357 matched quoted entrants, with a race-paired interval crossing zero, and effectively tied log-loss. This is suggestive, repeatedly inspected evidence below the 0.750 gate. The four existing one-time 2026 cutoff jobs now freeze Manifold alongside Kalshi; no prospective Manifold price or 2026 result has been scored. See E147–E149.

**E119 tyre-data update (2026-11-02):** The cached FastF1 lap feed distinguishes fresh from used sets, but its modern soft/medium/hard names are relative to each event and do not identify C1–C5. The fixed fresh-set correction worsened outer next-race pace correlation and tied standings at strict podium overlap. C3 still needs a separate permissible, timestamped absolute-compound source; the current feed cannot provide it.

**C3 source recheck (2026-11-03):** [OpenF1's documented stint field](https://openf1.org/docs/) also exposes relative `SOFT/MEDIUM/HARD` labels. A search found no verified permissible structured 2020–25 archive of absolute C1–C5 nominations with historical pre-weekend timestamps. This is a source gap, not evidence that such a feed cannot exist. No Pirelli preview or other outside-document text entered a model.

**E151 recheck:** OpenF1 still exposes only relative stint labels, and the inspected Pirelli developer portal did not establish a public, timestamped historical F1 nomination feed. FIA/Pirelli previews remain excluded by the user's no-outside-documents instruction. Park C3 until a permissible structured source and its pre-weekend provenance are verified.

**E138 newcomer update (2026-11-03):** Across 131 development races, the 17 drivers first recorded in F1 during 2020–25 had zero podiums in their first five entries and four in their first 20 (1.0% of 393 podium slots). Junior-series data might still inform later performance, but its direct early-podium leverage is small. The official F2 website restricts extraction/copying, and the inspected third-party pipelines scrape it. Park C5 pending a permissible independent structured source; see E138.

## Outside-document avenues: recorded, excluded from the current scope

| ID | Originally proposed avenue | Status and possible distinct information |
|---|---|---|
| X1 | FIA power-unit/component usage reports | **Excluded.** Could identify actual component inventory and accumulated use; A4's exposure proxies must not pretend to know this. |
| X2 | FIA car-presentation/upgrade submissions | **Excluded.** Could identify hardware changes or different specifications between teammates. No presumed gain: an upgrade label is not a measured performance improvement. |
| X3 | Archived pre-weekend expert pecking orders, extracted by an LLM | **Excluded.** Potential external consensus signal. Publication dates alone would not solve revised-page or pretrained-model outcome-memory leakage; evidence-grounded extraction and blinded audits would be necessary if scope ever changes. |
| X4 | Pirelli previews: nominated compounds, abrasion, tyre stress, strategy descriptions | **Excluded.** C3 only applies if an acceptable structured alternative exists. Historical average hardness in E98 is already tested. |
| X5 | Team announcements about upgrades, spare shortages, known faults, driver health, or personnel changes | **Excluded.** These might identify discontinuities absent from result histories, but no text-derived feature is allowed under the current instruction. |
| X6 | FIA event notes, carry-over sanctions, technical directives, or planned penalties from documents | **Excluded.** Potentially useful availability/constraint information. Do not substitute an actual target-weekend grid penalty as a supposedly pre-weekend input. |

## Supporting work that improves evidence, not a route to a metric pass

1. **Availability and provenance audit.** Store event time, publication/availability
   time, source version, and extraction time separately. A historical row dated
   before a prediction is not sufficient if published later. An uncertain
   timestamp is unavailable, not silently assumed admissible. Report missingness
   by era/team, because filtering to well-covered races can inflate results.
2. **Measurement uncertainty and retirement censoring.** Audit whether short
   races, early retirements, wet running, or traffic make the pace target itself
   misleading. Report sensitivity to reasonable target constructions. Do not
   choose a new pace label based on which produces the largest podium score;
   actual official podium labels stay fixed.
3. **Residual information audit.** Test whether each new block predicts mistakes
   left after standings/form. Include a feature-source permutation/placebo that
   preserves time/race structure, alongside the required label-shuffle null.
   A low-dimensional model with one new block is the starting point.
4. **Prospective pre-weekend prediction ledger.** Save immutable input hashes,
   cutoff, eligible drivers, code version, model, and top three before the
   weekend. Future unobserved races provide stronger evidence than another
   search over the repeatedly viewed outer window. This does not authorize
   refreshing or evaluating 2026 in this task.
5. **Metric family with uncertainty.** Keep strict overlap first. Then show
   winner accuracy, at-least-two-of-three, top-five coverage, NDCG@3, and
   calibrated podium Brier/log-loss with explicit denominators. E88 already
   did this; extend its reporting to new experiments, not present it as a new
   predictive avenue or a way to claim the 75% target was met.

## Experiment contract for any future implementation

1. **Pre-register a small batch.** Pick one measurement hypothesis and at most
   a few prespecified variants. Tune on inner 2020–22 only; use outer 2023–25
   for the specified comparison. Both windows have already informed E0–E100,
   so call new outer results development evidence, not a pristine holdout.
2. **Run chronological prediction.** For race K, every fit, imputation, scaling,
   circuit template, feature selection, and hyperparameter choice uses only
   data available before K's cutoff. Keep the `tie=None` pre-weekend path and
   `W.assert_pre_weekend()` active. Training on K's labels for a different
   driver is still leakage. Do not use the harness's grid-based empty-training
   fallback in a pre-weekend experiment; explicitly audit cold starts.
3. **Extend the source-specific leak test.** Scramble/delete target-weekend
   and later source rows; target features and predictions must remain identical.
   Also test publication-time exclusion for late-issued data describing an
   earlier event. Mutation-check that introducing a target row makes the test
   fail. Run all existing guards whenever features change:

   ```sh
   PYTHONPATH=. .venv/bin/python -m tests.test_no_leak
   PYTHONPATH=. .venv/bin/python -m tests.test_preweekend
   PYTHONPATH=. .venv/bin/python -m tests.test_profile_leak
   PYTHONPATH=. .venv/bin/python -m tests.test_guard
   PYTHONPATH=. .venv/bin/python -m tests.test_metrics
   ```

4. **Compare identical race sets.** Report strict overlap for the new method,
   plain standings, and frozen `ens` on inner and outer separately. Align
   `(season, round)` before `W.paired_boot`; show paired differences and CIs,
   per-season results, and a season-block sensitivity analysis. Run the
   label-shuffle null (`experiments/a_yardstick.py` is the existing reference).
   A new feature need not make a shuffled-label model exactly random when an
   unshuffled standings component is retained; isolate what the null tests.
5. **Respect limited power and repeated searching.** The logged detection
   threshold is about five percentage points at 131 races under its assumptions;
   it is not a universal CI width or a guarantee of equivalent power in subsets.
   Record all variants and failed sources. A small lift can be promising without
   establishing superiority; correlation gains alone do not meet the goal.
6. **Keep 2026 restricted.** Do not inspect new 2026 labels, summaries, or
   race-level residuals during development. Freeze a final candidate before
   any permitted final comparison; log every additional look and all prior
   exposure, including E86/E93. Previously observed 2026 is not restored to
   an untouched holdout by freezing a new model.
7. **Log and save each executed batch.** Add an `EXPERIMENTS.md` entry with
   source coverage, temporal rules, variants, inner/outer numbers, standings
   comparison, uncertainty, bugs, and verdict. Commit and push the batch.
   Do not assign experimental results to this backlog: none were run here.

## Recommended execution order and stopping rule

Start with **A1/A2's measurement pilot**, using a fixed small inner-period
sample, then A3/A5 only if matched measurements improve. Independently try
**A4's cause-specific reliability baseline** with existing status/exposure data.
Follow with **A6's tightly bounded testing extension**. Use A8 to check whether
any gain reaches podium contenders before building a larger model.

Advance a source only when it has admissible timestamps, usable coverage,
credible measurement quality, and incremental forecast information. Follow
promising forecast evidence with the frozen podium comparison; do not require
a tiny pilot to prove a five-point overlap gain. If the new information still
ties standings, log that bounded negative result and move to the next distinct
source rather than tuning many learners on the same rows.

**Assessment:** 75% remains unachieved and ambitious relative to the evidence.
The best permitted next bets are cleaner race-trim measurements and more
specific reliability/driver comparisons. Their value is assumed, not verified.
If they fail, the honest conclusion is that the tested permitted sources have
not demonstrated a path to 75%, not that the metric should be loosened or that
every possible pre-weekend predictor has been mathematically ruled out.
