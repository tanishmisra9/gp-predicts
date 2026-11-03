# Prospective pre-weekend forecast: 2026 round 20, São Paulo

Frozen at **2026-11-03 02:58:49 UTC**, before the 2026-11-06–08 race weekend. Run from parent commit `43e03da1376f57c2304710aadb9833612e475045` with the `predict.py` guard/standings-output change in this batch. Command: `PYTHONPATH=. .venv/bin/python predict.py`.

The local `data/results.csv` ended at round 19 (Mexico City, 2026-11-01). The forecast used those and strictly earlier results plus existing historical pace data. It assumed the Mexico City driver/team entry list for São Paulo. The target race date, name, and circuit came from the structured Jolpica calendar; no São Paulo practice, qualifying, grid, sprint, telemetry, or result was used. No outside-document text was used as a model input.

| Predicted rank | Driver | Team |
|---:|---|---|
| 1 | antonelli | mercedes |
| 2 | max_verstappen | red_bull |
| 3 | hamilton | ferrari |
| 4 | leclerc | ferrari |
| 5 | russell | mercedes |
| 6 | norris | mclaren |
| 7 | piastri | mclaren |
| 8 | hadjar | red_bull |
| 9 | gasly | alpine |
| 10 | lawson | rb |

**Frozen strict podium set:** antonelli, max_verstappen, hamilton. **Plain standings podium set:** antonelli, max_verstappen, hamilton. Strict overlap, winner accuracy, and all other outcome scores: **pending**. This is a prospective observation, not a new model-selection or holdout result. Do not revise the set after target-weekend information appears.

While checking the local results cutoff, round 19's actual top five was viewed. That is a further look at the previously inspected 2026 holdout, logged here; it did not change the frozen model or this selection.

## Prospective market comparator registered 2026-11-03

Kalshi's public API lists `KXF1RACEPODIUM-BRAZ26` with 22 driver-podium contracts, opened 2026-11-03 05:06 UTC. At the 2026-11-03 availability check, only 4 contracts had traded (110.06 total contracts), and the median bid–ask spread was 0.24. Those early quotes are **not** a forecast and did not change the frozen picks above.

Use E132's unchanged rule: for each contract take the last hourly bid/ask midpoint whose bar ends at or before **2026-11-05 00:00 UTC**, rank drivers by descending midpoint with driver-name ties, and score the top three against the official result only after the race. If fewer than three drivers have a valid bid/ask pair by that cutoff, record market coverage as unavailable rather than substituting later prices. This cutoff is 39.5 hours before the scheduled first practice on 2026-11-06 15:30 UTC. Freeze the extracted quote table before scoring; do not tune a blend or revise the original model selection.

The E137 capture command is `PYTHONPATH=. .venv/bin/python -m experiments.c1_capture_2026_r20`. Run it after the cutoff and before first practice. It saves the one-time quote table to `data/parts/kalshi_prethu_2026_r20.csv` and prints the market top three if at least three contracts have valid quotes. The command rejects earlier or later execution and refuses to overwrite a saved table. Commit the table and top three before any weekend outcome is read. **Capture remains pending.**
