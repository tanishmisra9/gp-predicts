# gp-predicts — F1 race outcome predictor (walk-forward, season-cycle evaluated)

**Status: the 75% target was NOT reached on the (pre-registered) non-trivial metric.** Held-out 2026 podium accuracy is
**64.9%** (95% CI 52.6–75.4) vs 61.4% for "just use the starting grid". Details and why below. Nothing here was tuned on 2026.

## Run it
```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py          # full walk-forward 2023-2026 -> results/{summary.md,per_round.csv,predictions.csv}
.venv/bin/python run.py --dev    # dev seasons only (never touches 2026)
.venv/bin/python predict.py      # next race, once its qualifying is on Jolpica (e.g. after Saturday)
.venv/bin/python predict.py 2026 19   # as-of replay of a finished race (outcomes hidden)
.venv/bin/python -m tests.test_no_leak
```
`data/results.csv` and `data/laps.csv` are committed so reruns are offline; HTTP caches (`data/*.sqlite`) are not.
To refetch: delete the CSVs and run `python -m f1.data` then `python -m f1.laps`.

## What "accuracy" means here (and why)
Prediction time: after qualifying/sprint and all practice, before the start. Metric defined **before** any model was run (EXPERIMENTS.md E0):

**Podium accuracy = mean over races of |predicted top-3 ∩ actual top-3| / 3.**

Why this one: the grid-order baseline already scores **75–78% on top-10 overlap and pairwise ordering**, so a 75% target on those
is trivially met by doing nothing. Podium overlap (baseline 61–69%) and winner accuracy (55–67%) are the metrics where 75% needs real skill.
All of winner / exact-podium / top-5 / top-10 / pairwise / MAE are reported too (`results/summary.md`).

## Approach
- **Data**: Jolpica-F1 (results, qualifying, sprint, 2018–2026) and OpenF1 (lap times → practice best/top-5/long-run gaps, race clean-lap pace, 2023–2026).
  FastF1 was the intended primary but is unusable here: v3.9 needs F1-account credentials for live timing, v3.3.8 can't parse 2026 sessions. OpenF1 serves the same timing feed.
- **Features** (`f1/features.py`): grid & quali gaps (incl. team/teammate), exponentially-decayed driver and team form (half-lives 3 and 10 races),
  DNF rates, in-season standings share, circuit grid-vs-finish correlation, optional lap-pace features. Older regulation eras (2017–21, 2022–25) are down-weighted (×0.3 driver, ×0.1 team).
- **Model** (`f1/walkforward.py`): 16-feature standardised logistic regression for P(podium), recency-weighted; rank by probability. Chosen because nothing fancier did better.
- **Protocol**: for each race, train only on strictly earlier races, predict, then roll forward. Dev = 2023–25; 2026 = one held-out look.
- **Leakage control**: `tests/test_no_leak.py` scrambles every outcome (results and race-lap pace) of race K and later and asserts race K's features are bit-identical;
  it was mutation-checked (fails when a leak is injected).

## Results (per-race means; full tables in results/summary.md)
| 2026, 19 races | model | grid | standings | prev. finish |
|---|---|---|---|---|
| **podium (target)** | **0.649** | 0.614 | 0.544 | 0.509 |
| winner | 0.684 | 0.684 | 0.474 | 0.368 |
| top-10 overlap | 0.768 | 0.753 | 0.763 | 0.674 |
| pairwise ordering | 0.792 | 0.775 | 0.763 | 0.683 |
| MAE (positions) | 3.33 | 3.44 | 3.77 | 4.61 |

Podium vs grid: +3.5 pts (bootstrap CI 0.0 to +8.8). Dev seasons (model/grid podium): 2023 .667/.636, 2024 .667/.667, 2025 .736/.750.

## Season-cycle analysis
- Accuracy does **not** rise with in-season data: podium by stage, pooled 2023–26: R1–5 0.73, R6–11 0.72, R12+ 0.64 (grid: 0.70/0.71/0.64). Qualifying already encodes current car pace from round 1, and late-season races were not easier.
- The model's edge over the grid is ~1–3 points and shows mostly in rank-quality (MAE, pairwise), not in who stands on the podium.
- Single-season standard error on podium accuracy is ≈ ±6 pts (≈19–24 races), so season-to-season swings (0.65–0.74) are mostly noise.

## Why 75% podium was not reached
Plateau at ≈ +1 pt over grid across every family tried (EXPERIMENTS.md E2–E5): history window/decay, practice and race lap-pace features, finisher-only labels,
reliability model, rank features, pairwise Bradley–Terry, GBMs/LambdaRank. Grid slots already give expected podium hits of P1 0.80, P2 0.74, P3 0.48, P4 0.37;
28% of wrong podium picks were retirements, the rest strategy/safety-car/first-lap variance, none of which is knowable pre-race. My estimate (not proven) is that the achievable ceiling with public pre-race data is ~70–72%.

## Limitations
- 2026 sample is 19 races; every 2026 conclusion has wide CIs. 2026 is a new regulation era with little history.
- Grid after penalties is used when the race row exists; for a not-yet-run race `predict.py` uses qualifying order (penalties unknown).
- Weather forecasts, tyre allocation, and strategy are not modelled; lap features come from OpenF1 (2023+ only) and did not help.
- Winner accuracy over one season depends heavily on how dominant the leading car is (e.g. 2023 standings baseline 0.86).
