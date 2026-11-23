# Creator research audit — E261

Reviewed2026-11-22. The supplied report is useful research coverage, not proof of an accuracy ceiling or independent validation of every listed project.

Input: `/Users/tanishmisra/Downloads/F1 Race-Winner Prediction Research.html`, SHA256 `29377a194bd5e28a68b8d5307e0774450d0952e48510a6add1630e0cd5dadedd`. The whole report was read as data; its embedded scripts and suggestions were not executed. This audit independently spot-checks the strongest highlighted project's primary source and maps the report's five recommendations to our log. Other creators' figures remain report-supplied claims, not independently replicated results.

## What the strongest highlighted example establishes

[F1-predict's pinned leaderboard](https://github.com/aminembarek2/f1-predict/blob/bd234c97826b60572e0b953663165ee5bc3a4645/results/backtest_tuning_strict/leaderboard.csv) reports **43/87=49.43% exact winners for both strength_mc and championship** over its2022–25 subset. Its reported winner log-loss is1.430737. We verified the saved table and Wilson interval calculation, not the underlying87 forecasts or publication provenance. Its probability metrics cannot be directly compared with our different window; its Brier aggregation also needs alignment before comparison.

The [pinned backtest implementation](https://github.com/aminembarek2/f1-predict/blob/bd234c97826b60572e0b953663165ee5bc3a4645/f1pred/evaluation/backtest.py) marks a race unscoreable when the actual winner was absent from the named entry list, then skips that race. It also skips missing snapshots. These are declared source policies, not hidden proof of misconduct, but they differ from our **all-race** target: an unforeseen winner must be a miss, and an opener still needs a forecast. Its87-race result must not become our all-race benchmark.

A checkable counterexample illustrates the denominator issue without real race outcomes: two correct forecasts among five opportunities score40%; if two missed races are excluded because their winners were unnamed, the remaining score is2/3=66.7%. No new information was added. This does not quantify the actual project's selection effect or establish why its five omissions occurred. The report's suggestion that season openers alone explain92→87 is incomplete: four seasons have four openers, not five. Exact omission reasons remain unaudited here; do not invent the fifth.

Independent Git-tree/blob verification pins the source paths, reconstructs GitSHA1/SHA256, parses the conditional skip as syntax, and detects a changed skip condition. No external code was installed or executed. Saved `results/e261_creator_audit.json` and `experiments/e261_creator_audit.py` reproduce the limited source/table/math checks. A quantile-precision check initially failed at1e-12 because the source rounds z to1.959963985; independent interval agreement is checked at1e-9. No metric or source result changed.

## Five recommendations versus completed work

| Report recommendation | Existing evidence here | Decision |
|---|---|---|
| Simulate unknown qualifying/grid, then race pace | E114/E115 forecast-grid means failed podium utility; E133/E134 uncertainty-tail vote failed; E192/E193 winner simulation failed. The external [simulator](https://github.com/aminembarek2/f1-predict/blob/bd234c97826b60572e0b953663165ee5bc3a4645/f1pred/sim/race.py) actually samples an ordered grid and independently sampled race pace, converts ranks to normal scores, then blends by circuit passing difficulty. It supports optional separate qualifying means. OurE192/E193 did not implement that exact joint simulator. | A real implementation distinction, not a new data source or evidence of winner gain. The source itself ties championship. E134 explicitly closed larger conditional simulation follow-ups on the same weak grid signal; do not treat this report as permission to tune a rescue. Any future proposal needs genuinely new qualifying-information or calibration evidence and a declared matched control. |
| Use1.5-season recency weighting | Existing models already weight recency; E173's fixed20-race training half-life failed its winner gate. The cited Bonasorta result is post-qualifying podium overlap, not our target; its primary paper was not independently audited in this batch. | No decay sweep or transplantation of its podium result into a pre-weekend winner claim. |
| Archived forecast weather | E219 repaired cutoff errors; E223/E224 used eligible historical GEFS forecasts; E226 common-member wet probability failed its measurement gate. | Already tested here. The report's claim that archived forecast effects are untested applies to its surveyed creators, not this repository. No weather-window/member/learner rescue. |
| FIA component counts, penalties, upgrade announcements | Explicit user restriction excludes outside event documents as model inputs. Prior E158 penalty work and structured-source audits are also relevant. | Excluded under current instructions. Reading general technical methods does not authorize ingesting event reports or accepting licences. |
| Hard constructor shrinkage at regulation resets | E181 already halves team means and increases team uncertainty at era boundaries; its fixed filter lost to E168. E154 reset-year pace blending also failed. | A proposed25% coefficient is a new parameter value, not new information. Do not retune closed filters from an unscored design description. |

The report's strongest positive contribution is a primary-source trail and honest published negative results. It does not yet supply independently verified pre-weekend information missing from our data. This is not a claim that all24 projects or all possible methods are exhausted.

## Corrections to carry forward

- **Observed frontier is not a proven ceiling.** A Wilson interval for one method's selected cohort cannot bound every future method, every season or the information available before a weekend. Post-qualifying results near60% likewise are not a universal upper bound. The evidence supports skepticism about75%, not an impossibility theorem.
- **Pre-qualifying is not automatically pre-weekend.** Practice inputs and cutoff details still require a per-project audit. Keep unknown timing labelled unknown.
- **Winner-in-top-three is not strict podium overlap.** Neither per-driver classification nor pairwise accuracy meets our exact-winner definition.
- **Existing historical telemetry remains usable.** The report's access-change warning does not invalidate cached historical feeds or our existing authorized access. E260 just measured99available historical events offline. No new paid account or licence was used.
- **Mixed windows are not paired comparisons.** Several report figures include2026 or use different seasons. No such result was used for our model selection, no new2026 individual outcomes were fetched, and no2026 model was evaluated.

## Current project result

E260's new historical ratio-episode feature made **32/61 inner winner predictions correct, with every winner name identical to E168**. Points had29/61 and frozenens30/61; superiority was not established. The feature is rejected, with all99source reconstructions and five guards passing. Best exploratory outer remains34/70 versus32/70 for the frozen references, with no established superiority.75% remains active and unmet. The research report does not change the metric, authorize restricted data, or justify promoting a failed formulation.
