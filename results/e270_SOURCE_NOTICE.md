# F1DB historical supplier projection

Source: F1DB and its contributors, https://github.com/f1db/f1db/tree/9dbbb9f67dc5fa3d499061efb56d24aa4f1e261d

Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/
Original licence: https://github.com/f1db/f1db/blob/9dbbb9f67dc5fa3d499061efb56d24aa4f1e261d/LICENSE

User approved historical supplier-data use on 2026-11-22. Source is pinned; original notices and licence apply. No endorsement by F1DB is implied.

Modifications: retain only historical identity and recorded engine supplier/engine links from the 103 race-result and five annual-entrant files in the E269 manifest. Normalize car numbers to decimal strings and strip identifier whitespace. Join to existing offline FastF1 car-number/code identities and official Jolpica entry keys; add legacy-map coverage/comparison fields. Exclude source finishing position, grid, points, times and retirement text from exported projections. Annual entrant links provide consistency checks only. Recorded brand does not establish physical manufacturer or historical publication time. No prediction model uses these records in E270.

Exact source paths, Git blob hashes and SHA256 receipts accompany the projection; raw hash-verified YAML stays in local `/tmp/gp-predicts-e270-source`, outside versioned model inputs. This notice accompanies `e270_projection.json` and `e270_matched.csv`.

E270 completed 2026-11-23: all 108 permitted files verified; matched coverage 2,051/2,060 official entries. Nine unknown identity joins remain missing. Brand labels, including tag-heuer, bwt-mercedes and rbpt, are retained without physical-OEM equivalence. The independent audit exports season/team counts and transparent legacy comparisons only. No historical publication-time or predictive-performance qualification follows from this projection.
