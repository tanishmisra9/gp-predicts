# E256: pit-lane time versus stationary stop time

A source audit found no newly qualified stationary-service input for the historical pilot. No forecast or accuracy evaluation was run.

## Primary reader evidence

OpenF1's own [pinned pit-ingestion implementation](https://github.com/br-g/openf1/blob/853ae3398a243487c52306b99093f067c951cb50/src/openf1/services/ingestor_livetiming/core/processing/collections/pit.py) distinguishes these branches:

| Feed | Field interpreted for lane time | Field interpreted for stationary time |
|---|---|---|
| PitStop | PitLaneTime | PitStopTime |
| PitStopSeries | nested PitLaneTime | nested PitStopTime |
| PitLaneTimeCollection | Duration | None |

This is evidence of the independent reader's implementation. It is not an official specification, an independent timing-ground-truth validation, or a complete inspection of every raw historical field.

## Historical metadata check

The saved E229 index inventories for2019SpainR5,HungaryR12 andItalyR14 each contain PitLaneTimeCollection. None contains PitStop or PitStopSeries.

PitStop was omitted from E229's original three-name candidate set. The new primary implementation identifies it as a relevant service-time topic, but checking the already saved inventories still finds none. E229's original registration and results were not rewritten.

The verifier parses the pinned source as syntax, without executing it. It checks branch-specific field assignments, the literal unset stationary-time argument, the Git blob hash and the saved inventory hash. Synthetic mapping changes alter the checks. A separate inventory reconstruction reproduces all three results.

## Decision

Do not call the historical lane-duration field a crew-service measurement. No new raw F1 feed, source year, account or model was acquired. E121's lane-tail and E210's telemetry-stationary formulations remain closed; this audit does not justify changing them.

A genuinely independent permissible source of historical stationary timing, with units and availability verified, would still require measurement and temporal utility tests before any winner feature. Public availability of a modern reader alone does not supply that dataset. Best exploratory winner accuracy remains34/70(48.6%);75% remains unmet.
