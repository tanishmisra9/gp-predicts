# Historical mapped-circuit geometry — E280 approval proposal

**Status: prepared and tested offline; no OSM records acquired. Explicit user approval is required before acquisition.** Approval of the F1DB supplier audit does not cover this source. The user prohibited new licences without approval.

The proposed next step is a small source audit, not a forecast experiment. Historical mapped geometry could add information about circuit changes that current static descriptors omit. We do not yet know whether the maps identify the Grand Prix layout accurately, whether any differences represent physical changes, or whether this could improve winner prediction. E278 showed a need for new correct winning-team choices; it did not establish this source's usefulness.

## Licence and access

OpenStreetMap identifies its map data as licensed under [ODbL](https://www.openstreetmap.org/copyright). The provider describes attribution and share-alike conditions; this is not an obligation-free dataset. The proposed scope includes retaining the licence and attribution notices with the source and any permitted derivative projection. Attribution: **© OpenStreetMap contributors**, with links to [the copyright page](https://www.openstreetmap.org/copyright) and [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). No account, purchase or new software dependency is proposed.

The [Overpass operator's usage guidance](https://dev.overpass-api.de/overpass-doc/en/preface/commons.html) permits small public queries and describes resource limits. This pilot would use six small, sequential requests rather than a bulk download or continuous service. No main OSM editing API, tiles, Nominatim, alternate server or native F1 feed would be accessed.

The [historical-query documentation](https://dev.overpass-api.de/overpass-doc/en/analysis/museum.html) describes dated database states. It also warns that a way can change geometry without a new way version because its nodes move. The pilot therefore retrieves and validates historical node records as well as ways. Map edit timestamps support database-state chronology; they do not certify survey accuracy, physical track-change dates or review completeness.

## Exact proposed acquisition

The endpoint is `https://overpass-api.de/api/interpreter`. There are exactly six GET queries, one for each row below. Each uses a fixed rectangle of ±0.035 degrees around existing local circuit coordinates. Those coordinates are geographic locators, not new model features or evidence of historical layout accuracy.

| Circuit | Season / round | Rectangle centre (latitude, longitude) | Snapshot UTC |
|---|---|---|---|
| Catalunya | 2019 / 5 | 41.570000, 2.261110 | 2019-05-08 23:59:59 |
| Catalunya | 2022 / 6 | 41.570000, 2.261110 | 2022-05-18 23:59:59 |
| Hungaroring | 2019 / 12 | 47.578900, 19.248600 | 2019-07-31 23:59:59 |
| Hungaroring | 2022 / 13 | 47.578900, 19.248600 | 2022-07-27 23:59:59 |
| Monza | 2019 / 14 | 45.615600, 9.281110 | 2019-09-04 23:59:59 |
| Monza | 2022 / 16 | 45.615600, 9.281110 | 2022-09-07 23:59:59 |

These are one second before the respective Thursday 00:00 UTC cutoff. The three sites match the established 2019 source pilots; 2019/2022 contrast earlier/later inner-era snapshots. They were not selected from OSM changes or winner-error outcomes. No map records have been inspected.

The exact queries, rectangles, hashes and local calendar/locator hashes are in [the query manifest](results/e280_osm_query_proposal.json). The query selects `highway=raceway` ways and their referenced nodes, using a dated state and `out meta`. It does not request event documents, race results, driver statistics, betting data or current geometry.

Limits: one attempt per query, at least two seconds between requests, 25-second server runtime declaration, 64 MiB server-memory declaration, 35-second HTTP timeout, 2 MiB streamed response cap each / 12 MiB maximum total. No redirects, retries, authentication or mirror fallback. Any non-200, network failure, timeout, oversize body, server remark or invalid schema stops the entire acquisition; unattempted snapshots remain unknown. No expansion of dates, rectangles, tags, sites or thresholds to force a pass.

## Data handling and source gate

Retain only node/way identifiers, versions, UTC timestamps, node coordinates, ordered way-node references and the `raceway` classification. Exclude names, notes, descriptions, changeset comments, user names, user IDs, URLs and other text from the exported projection. A bounded raw response may contain those fields; if approved, retain it outside model inputs/versioned raw data solely for hash and projection audit. Keep an attribution/licence/snapshot/transformation notice with the projection. No upstream code is executed.

All six snapshots must return bounded valid JSON and at least one raceway with two distinct referenced nodes. Every node and way must have valid identifiers/versions and a timestamp no later than its requested snapshot, strictly before Thursday. Every referenced historical node must resolve. Duplicate identities, missing/extra nodes, nonfinite coordinates, malformed timestamps, unsupported element types and empty snapshots fail. A current server database-base timestamp is not interpreted as the element's publication time.

Passing this gate would establish only a usable historical mapping projection. It would not identify the correct closed Grand Prix route, measure resurfacing, prove a physical change or permit a model. A separate registration and independent geometry validation would be required next. This does not reopen E264's denied position feed or its closed mapping pipeline. Any later winner experiment must retain all 61 inner races and missing-source fallbacks, compare standings/ens/E168 and matched controls/null, pass paired admission and fresh leakage guards. No outer or 2026 evaluation is authorized.

## Work already completed

[The offline builder/projector](experiments/e280_osm_proposal.py) has no HTTP acquisition entry point. [Tests](tests/test_osm_proposal.py) independently verify all six dates, rectangles and query strings; they check historical node/way consistency, a node moving without a way-version change, ignored-text poisoning, positive coordinate responsiveness and 14 malformed/history-unsafe cases. The manifest is explicitly marked `approval_received: false` and `acquisition_performed: false`.

After approval, implement and test the bounded request runner before making these exact requests. Then independently verify all raw responses, request/hash receipts, projections, timestamps, references, counts and source gate. Approval is limited to this six-snapshot ODbL source audit; it does not authorize additional datasets, requests, licences, spending, accounts or production changes.

**Approval requested:** “Approve the six-snapshot OSM geometry pilot under ODbL.” Until that explicit response, no OSM data request will be made. The 75% exact pre-weekend winner goal remains active and unmet; this proposal supplies no accuracy gain.
