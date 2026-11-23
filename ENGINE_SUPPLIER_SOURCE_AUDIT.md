# Historical engine-supplier source audit — E269

Reviewed 2026-11-22. **No F1DB data-record bodies have been acquired or used.** This is a technical-source and file-metadata audit, pending the user's licence approval.

## Why this is worth checking

The existing `f1/profile.py` supplier map is explicitly hand-entered and unverified. On the 2,060 official 2018–22 driver entries, it assigns a supplier to 1,454 and leaves **606 unmapped (29.4%)**:

| Season | Mapped | Unmapped | Unmapped team IDs |
|---|---:|---:|---|
| 2018 | 294 | 126 | force_india, renault, toro_rosso |
| 2019 | 252 | 168 | alfa, racing_point, renault, toro_rosso |
| 2020 | 204 | 136 | alfa, alphatauri, racing_point, renault |
| 2021 | 352 | 88 | alfa, alphatauri |
| 2022 | 352 | 88 | alfa, alphatauri |

These are mapping-coverage counts, **not verified physical assignments or prediction errors**. The frozen winner model does not use this map; older profile/simulation experiments do. Completing it would not automatically improve any forecast. E193 preserved the old mapping to test source fidelity and explicitly left a better physical source untested. Its failed simulator remains closed.

## Primary source verified

[F1DB](https://github.com/f1db/f1db/tree/9dbbb9f67dc5fa3d499061efb56d24aa4f1e261d), pinned to commit `9dbbb9f67dc5fa3d499061efb56d24aa4f1e261d`, publishes structured historical data and schemas.

- The [race-result schema](https://github.com/f1db/f1db/blob/9dbbb9f67dc5fa3d499061efb56d24aa4f1e261d/src/schema/current/splitted/f1db-races-race-results.schema.json) requires string `engineManufacturerId`, `constructorId`, `driverId` and `driverNumber` fields. Required strings can still be empty; the schema alone does not prove usable records.
- The [entrant-engine schema](https://github.com/f1db/f1db/blob/9dbbb9f67dc5fa3d499061efb56d24aa4f1e261d/src/schema/current/splitted/f1db-seasons-entrants-engines.schema.json) links year, entrant, constructor, manufacturer and engine IDs. An annual list is not evidence of which combination applied before a particular race.
- Git tree and independent directory listings identify exactly **103 race-result file paths and five annual entrant files** for 2018–22: 21/21/17/22/22 race files. This is file metadata, not verified record coverage.
- The technical reader loads per-race YAML. The inspected result converter changes timing/position representations, and the output mapper attaches race/year/round. These code paths do not validate historical supplier attribution or establish publication timestamps.
- Technical file contents were independently hashed through Git blob and contents APIs. The actual record files were not requested. One initially guessed mapper filename returned 404; the correct path was obtained from the pinned tree.

No full database release, all-time driver statistics, 2023–26 records, betting/Fantasy data, or event-document text has been imported. E118 already recorded Fantasy extraction restrictions; this work does not reopen that route.

## Approval needed

The pinned [LICENSE](https://github.com/f1db/f1db/blob/9dbbb9f67dc5fa3d499061efb56d24aa4f1e261d/LICENSE) identifies **Creative Commons Attribution 4.0 International**. Its text says that exercising the licensed rights means accepting its terms. It includes attribution, licence-notice and modification-notice conditions when material is shared.

The user's mission explicitly requires asking before accepting licences. **Permission to use this dataset has not been assumed.** Reading technical documentation and identifying a candidate source does not authorize importing its records into this project. No account, purchase or licence-acceptance action has occurred.

Proposed approval: use the pinned 2018–22 supplier records under the stated CC BY 4.0 terms for this research, beginning with the audit below. Retain source, contributor, licence, snapshot and transformation notices with any acquired data. This does not authorize other sources, accounts, spending, restrictive scraping or a production-model change.

## Concrete next step after approval

Before acquiring records, register the exact source projection, matching rules and coverage criteria. Then:

1. Read only the 103 listed historical race-result files and five associated entrant files. Project historical identities and recorded supplier/engine links; ignore finishing order, grid, qualifying, points and free-text retirement reasons. Jolpica remains the label source.
2. Match historical car numbers to the existing cached historical FastF1 identities and official result keys. Missing or ambiguous matches stay unknown. This uses completed source weekends only and requires no new native F1 request.
3. Audit recorded manufacturer identity separately from physical OEM/hardware. Rebadged engines must not be unified from memory. Do not turn a season's union of entries into a race-specific assignment.
4. Report actual per-year coverage, disagreements with the legacy map, ambiguous brands and provenance limitations. Establish whether the source supplies a real correction before proposing a feature.
5. Any later forecast uses only completed source events before both the target race and its Thursday cutoff. New-season/missing history needs an explicit unknown fallback. Target-event supplier rows and future season aggregates cannot supply features.
6. A source pass permits a separately registered winner experiment against standings, frozen `ens`, the winner baseline and matched controls/nulls on all 61 inner races. It requires source poisoning/prefix tests and fresh guards. It does not reopen E193 or justify a parameter sweep.

Actual record quality, physical mapping, pre-weekend provenance and predictive utility are **unverified**. No new accuracy result exists. E268 remains 33/61 inner winners, tied with the existing numeric GBM; exploratory outer remains 34/70 versus frozen references 32/70, without established superiority. The 75% exact-winner goal is active and unmet.

Reproducibility: `experiments/e269_supplier_source_audit.py`, `tests/test_supplier_source_audit.py`, `results/e269_supplier_source_manifest.json` and `results/e269_legacy_supplier_coverage.json`. The manifest records only technical hashes and candidate record-file metadata, not record contents.
