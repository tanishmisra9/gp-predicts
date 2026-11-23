# Historical mapped-circuit geometry — E280 approval proposal

**Completed 2026-11-23: source gate failed at the first request.** User approval was explicit: “Approve the six-snapshot OSM pilot”. The registered Catalunya 2019 query ended in `ReadTimeout` without an HTTP response. The global stop prevented the other five requests. Zero map bodies/records were acquired. Their availability remains unknown; this is not evidence that historical geometry is absent. No retry, alternate endpoint, longer timeout or changed query was attempted.

The [independent source audit](results/e280_source_summary.json) verifies the one attempted exact URL, terminal failure, five unattempted snapshots, zero body bytes and unchanged receipt files. Actual geometry/projection/poison checks could not run because no source body was received. The empty [projection artifact](results/e280_osm_projection.json) carries attribution/licence context only. The original proposal and approved scope below remain recorded for provenance; their pending/next-step statements are historical and superseded by this result. The fixed pilot is closed without rescue.

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

The [bounded runner](experiments/e280_osm_acquire.py) was prepared offline while approval was pending. [Fake-response tests](tests/test_osm_acquire.py) cover all six exact requests, spacing, complete/prefix-cache reuse, pre-send checkpointing, interrupted attempts, concurrent/stale locks, sticky failures, redirects/denials/timeouts, response closure and the inclusive 2 MiB accepted-body boundary. Actual HTTP is forbidden during tests. The command-line entry point does not acquire data; an operator confirmation flag cannot itself supply user permission. The user's explicit response above supplies permission for this scope.

Requests' 35-second setting is its connection/read timeout, not a total wall-clock deadline. The 2 MiB cap bounds the accepted/stored decompressed JSON body; detecting an oversized response can read an additional chunk before closing, and transport buffering is not an exact wire-byte quota. These are implementation limitations, not relaxed source acceptance limits: every accepted body remains at most 2 MiB and at most six bodies can be accepted. No retry or source-gate change is authorized by this clarification.

Next, commit the tested runner before these exact requests, then independently verify all raw responses, request/hash receipts, projections, timestamps, references, counts and source gate. Approval is limited to this six-snapshot ODbL source audit; it does not authorize additional datasets, requests, licences, spending, accounts or production changes.

**Approval received:** “Approve the six-snapshot OSM pilot.” The immutable query proposal's `approval_received: false` records its pre-approval creation state; this dated update supersedes that status without changing the query manifest. The 75% exact pre-weekend winner goal remains active and unmet; approval supplies no accuracy gain.

## E283 independent-access documentation screen (2026-11-23)

HeiGIT documents an OSM-history API with historical geometry extraction, but its current v2 requires an account/API key. The migration guide's planned v1 shutdown date does not prove present access; inspected v1 documentation URLs redirect to v2. No service health/metadata/extraction request, map body or account action occurred. Usage terms, quota, credentials and actual source coverage remain unqualified. See `results/e283_osm_access_screen.json` and E283 in EXPERIMENTS.md for primary references and the limited claim audit.

E280 approval and failed-source stop remain unchanged. This documentation screen supplies no authority for extra requests, alternate hosts, new accounts or another acquisition. Any independently qualified future access requires its own concrete scope; no such proposal or approval request is made here.

## E285 additional access proposal prepared (2026-11-23; pending)

[HEIGIT_GEOMETRY_SOURCE_PROPOSAL.md](HEIGIT_GEOMETRY_SOURCE_PROPOSAL.md) now provides a separately scoped proposal for a user-owned free HeiGIT account/key, secure local key use, pinned PyArrow26.0.1 and six additional requests through the documented v2 extraction service. Public provider status and policy were checked; actual access/quota/source coverage remain untested. Pure query/projection tests passed; no account, dependency or map action occurred. Actual Parquet/auth/transport integration must pass after approval and before acquisition.

This is a new authorization request. E280 approval remains recorded and limited to its original exact Overpass scope; its timeout stop is unchanged. No old endpoint, alternate Overpass host, extra snapshot or model input is authorized by the proposal itself.


## E285 approval and integration update (2026-11-23)

Explicit “Approve the scoped HeiGIT pilot” received. Exact six-request/account/key/PyArrow scope already authorized; no new approval needed. Pinned wheel verified, isolated in research .venv-e285 after actual shared NumPy1.x import failure; initial shared addition removed without NumPy upgrade. Actual synthetic Parquet/CRS/selected-column corruption/cap/poison/response and allsix fake-response process-boundary tests PASS. Dedicated Keychain function tested ONLY with dummy subprocess output through actual Requests preparation; real key and network remain untested. See e285_decoder_installation.json / e285_decoder_verification.json and current setup instructions in HEIGIT_GEOMETRY_SOURCE_PROPOSAL.md. Await user account/quota/secure-item setup, then exact bounded acquisition and independent full-source audit. No real map request/source/accuracy result; E280timeoutstop remains unchanged.


E285 independent auditor implementation8bf665e now passes real synthetic all-six and source/receipt/projection corruption/poison/response/unknown-state checks. Uses separate projection/WKB logic, same pinned native PyArrow parser. No actual map/source/key action or coverage result; existing user setup confirmation remains pending. See e285_independent_audit_verification.json. E280timeoutstop unchanged.
