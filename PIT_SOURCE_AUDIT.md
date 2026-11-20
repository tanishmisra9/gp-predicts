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

## E257: DHL discovery and access qualification (2026-11-20)

This is a post-inspection source audit, not a preregistered measurement or forecast. No DHL timing page or timing endpoint was downloaded, no timing records were imported, and no external code was executed.

The [DHL site terms](https://inmotion.dhl/en/terms-of-use), inspected on 2026-11-20, condition reproduction on informational/noncommercial use, unchanged material, attribution and trademark restrictions. Interactive features have purpose limits. These conditions do not establish permission for our extraction/transformation/model workflow; this audit is not a legal determination that all such use is prohibited. No licence was accepted. Do not assume a public endpoint or a third-party repository grants upstream data rights.

An independent implementation supplies a useful discovery lead: [FormularWeb pit reader, pinned source](https://github.com/frank-fan-818/FormularWeb/blob/a8f3ece1db2fd2b66024fa879f6e52cf93751072/api/_lib/pit-stops.mjs), Git blob `99b0e03434f57204e42e537ec3161dc0d24370fd`. Its catalog parser recognizes `data-type=pit_stop` and `/api/f1-award-element-data/<number>` references. Its event reader requests an event parameter and parses chart records containing duration, driver, lap, points and irregular status. It separately obtains lane records from Jolpica and contemporary stationary records from OpenF1. This is evidence of that consumer's intended architecture, not verification of actual DHL responses, historical coverage, units, publication time, completeness or permission. Source inspected, never installed/copied/run. No race labels or example timing payloads were read.

The [OpenF1 maintainer discussion #182](https://github.com/br-g/openf1/discussions/182) also distinguishes lane and stationary times. A contributor reports that native PitStopSeries appears from the 2024 US GP; that report is not an exhaustive archive audit. Even if correct, late-2024 coverage cannot supply the registered 2020–22 inner training/evaluation window. Do not replace the inner window with a favorable late-era subset or inspect restricted 2026 races to rescue coverage.

**Qualification remains incomplete.** Structured response references are a real new source lead, but no dataset or model input is admitted. Actual historical coverage, stationary semantics, all-stop versus award-selected inclusion, irregular/slow-stop censoring, correction/publication timing and access rights remain unverified. Fastest-stop awards alone cannot identify a team's full delay/failure distribution without a selection model and independent observations. This is a missing-data caution, not a measured statement about DHL's actual inclusion rules.

Do not execute the discovered endpoints, rehost copies, scrape tables or fit a feature under the current authorization. E121/E210/E229 remain closed. Further work on this route requires independently established source eligibility and a separately registered bounded coverage/selection audit; it must not silently accept terms or bypass upstream restrictions. No accuracy claim follows from finding a URL pattern.
