# gp-predicts — F1 podium predictor (walk-forward, season-cycle evaluated)

**Status: the 75% podium target is NOT reached — and for the "no weekend data" setting it cannot be, by a measured bound.**

| setting | held-out 2026 podium accuracy | grid / best baseline | dev 2023-25 |
|---|---|---|---|
| **A. Pre-weekend** (no practice/quali/grid of the target race) | **0.544** (95% CI 0.44–0.63) | standings 0.526 | 0.53 / 0.56 / 0.67 |
| B. Post-qualifying (grid known) | 0.649 (95% CI 0.53–0.75) | grid 0.614 | 0.67 / 0.67 / 0.74 |
| *Oracle: true race pace of the target race known (diagnostic, post-race info)* | *0.79 (2022-25: 0.74)* | | |

Podium accuracy = mean over races of |predicted top-3 ∩ actual top-3| / 3 (defined before any model was run; EXPERIMENTS.md E0, E7).
The oracle row is the key: even *perfect knowledge of every driver's race pace* yields only ~74% podium overlap on 2022-25, because strategy,
safety cars, first-lap incidents and retirements scramble the order. A forecast made before the weekend has to estimate pace **and** survive that
noise, so 75% sits at/above the ceiling. Nothing tried (E2-E8) beat "championship standings" by more than noise.

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
race pace `./scripts_run_ff1.sh <path-to-.env-with-F1_USERNAME/F1_PASSWORD> [first last]` then `python -m f1.ff1 merge` (needs FastF1>=3.9 and an F1 account).

## Data (and why it is split this way)
- **Jolpica-F1 (Ergast successor)**: official results, grid, qualifying, sprints, status, 2018-2026. Source of truth for labels.
- **FastF1**: race laps -> clean-lap race pace per driver, 2018-2026 (validated: corr 0.993 vs. OpenF1 on 1,718 driver-races). I tried FastF1 for
  results too and rejected it: its grid/points/status come from Jolpica internally (NaN for many races when rate-limited) and its timing-based
  finishing order ignores post-race penalties (e.g. 2019 Canada), so it is a worse label source.
- **OpenF1** (`data/laps.csv`): practice/quali lap gaps, used only by the post-quali experiments (they did not help, E4).

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
