# E285: scoped HeiGIT historical geometry pilot — approval pending

This proposes a new, bounded access route to the same six OSM snapshots. Historical circuit geometry may eventually supply information about layout changes; no source coverage, correct Grand Prix route, physical change or winner benefit has been established. The goal remains 75% exact winners across all races before the weekend.

E280's approved Overpass pilot remains stopped after its first timeout. No retry, mirror or changed Overpass query is proposed. Its approval explicitly excluded additional requests and accounts in [OSM_GEOMETRY_SOURCE_AUDIT.md](OSM_GEOMETRY_SOURCE_AUDIT.md). This proposal requires **additional authorization** for a different provider, free account/key use and a pinned decoder dependency. It does not ask for the existing OSM approval again.

## Exact additional authorization requested

1. Use a user-owned **free HeiGIT account/API key**. Reuse an existing account if there is one; the provider permits one account per person. The user handles signup/login, identity, terms consent, email verification and secure key provision. No invented identity, duplicate account, automatic newsletter, paid plan, billing or provider contact.
2. Permit a local request process to use that key **only for these six requests**, in the `Authorization` header. No key in chat, URL, manifest, logs or receipts; no `.env` or unrelated secret lookup. A task-specific macOS Keychain item named `gp-predicts-heigit` is the proposed local credential channel, subject to explicit user approval. The model must not receive the key.
3. Download/install **PyArrow 26.0.1 (Apache-2.0)** only in the research virtual environment, preserving its packaged licence/notices and installation receipt. Official PyPI metadata declares no runtime dependencies. The compatible wheel is 37,290,209 bytes (about37MB), pinned SHA256 `217497a51fd41abb74502ba2f4900872e0944c226945c02ee8a1cb4f81df3392`. No Shapely/GeoPandas or other package is proposed. No wheel has been downloaded or installed.
4. After the offline integration checks below pass, make **at most six additional historical geometry GET attempts** at `https://api.heigit.org/ohsome-api/v2/extraction/features.parquet`, exactly as listed below. No other dataset, map query, service preflight, date, site, tag, relation or model input is authorized by this proposal.

No account action, terms acceptance, key lookup, dependency acquisition or map request has occurred. The offline CLI cannot perform those actions.

## Evidence for this access route

The [current API reference](https://docs.ohsome.org/ohsome-api/stable/reference/api.html) documents Parquet extraction and [historical snapshot times](https://docs.ohsome.org/ohsome-api/v2/reference/time.html). The [provider status page](https://status.heigit.org/) lists v2 extraction as operational; this is a provider claim, not a successful request or proof of source coverage. The [migration guide](https://docs.ohsome.org/ohsome-api/stable/migration_guide.html) requires an API key. No legacy v1 route is proposed or claimed live.

The [account terms](https://account.heigit.org/info/tos) cover ohsome, prohibit abusive use, require attribution and allow one account per person. The [public plans](https://account.heigit.org/info/plans) component advertises standard extraction250/day and15/minute; the [FAQ](https://docs.ohsome.org/ohsome-api/stable/faq.html) independently describes the15/minute limit. These are public policy metadata, not verified user entitlement or remaining quota. After authorized account setup, confirm sufficient remaining quota in the dashboard before acquisition; no quota upgrade or additional API probe. Stop on denial or exhaustion.

Because public account pages render through JavaScript, their public route, terms, plans and signup components were read as static documents. No code executed, signup/auth/challenge endpoint called or account created. Their hashes are in [the metadata receipt](results/e285_source_metadata.json). Current [Apache Arrow docs](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html) support reading a bounded in-memory file with selected columns; [Parquet's geospatial specification](https://parquet.apache.org/docs/file-format/types/geospatial/) defines WKB/CRS axis order. Context7 did not return a direct PyArrow documentation source, so those primary references were used.

## Six exact requests

Every GET uses `filter=highway=raceway and type:way` and `clip=false`. `aoi` is longitude-minimum, latitude-minimum, longitude-maximum, latitude-maximum. Each timestamp is one second before its target Thursday00UTC. These bounds/times come unchanged from the immutable E280 proposal; no map values were inspected to select them.

| Circuit | Season / round | aoi | time UTC |
|---|---|---|---|
| Catalunya | 2019 / 5 | 2.226110,41.535000,2.296110,41.605000 | 2019-05-08T23:59:59Z |
| Catalunya | 2022 / 6 | 2.226110,41.535000,2.296110,41.605000 | 2022-05-18T23:59:59Z |
| Hungaroring | 2019 / 12 | 19.213600,47.543900,19.283600,47.613900 | 2019-07-31T23:59:59Z |
| Hungaroring | 2022 / 13 | 19.213600,47.543900,19.283600,47.613900 | 2022-07-27T23:59:59Z |
| Monza | 2019 / 14 | 9.246110,45.580600,9.316110,45.650600 | 2019-09-04T23:59:59Z |
| Monza | 2022 / 16 | 9.246110,45.580600,9.316110,45.650600 | 2022-09-07T23:59:59Z |

Exact request hashes and bounds are in [the offline manifest](results/e285_heigit_query_proposal.json). Header values and keys are deliberately absent.

## Limits and stop rule

One attempt per request, serial, at least5seconds between starts. Connection timeout10seconds/read timeout35seconds. Accepted returned Parquet file after HTTP content decoding≤2MiB each/12MiB total. These bound accepted file bytes, not exact wire bytes or total wall-clock runtime. Declared selected-column uncompressed sizes≤16MiB, rows≤10,000 and coordinates≤100,000 per snapshot; these are acceptance checks, not a hard total process-memory bound.

**Any** timeout, network error, status other than200, redirect, denial/quota error, oversize file, schema/CRS/geometry/chronology error globally stops acquisition. No retries, mirrors, legacy endpoints, longer timeouts, changed query/filter/bounds, paid upgrade or account rotation. Unattempted snapshots remain unknown. Reserve attempts before sending and preserve sticky failure/resume receipts; do not restart an uncertain live attempt.

## Projection, notices and verification

Select identity/version/minor-version/edit-time/geometry-type/WKB/clipped fields and tags solely to verify `highway=raceway`. Never decode contributor identity columns or export names, comments, URLs, other text, changesets, or future `valid_to_timestamp`. Validate CRS evidence from the file; unknown CRS fails. Read a local in-memory file only, with no remote filesystem or dynamic extension/plugin loading.

Retain all returned records or fail; never drop an unsupported record to obtain a pass. Require positive way IDs/version, nonnegative minor-version, no duplicates/clipping, finite complete2DLineString/Polygon coordinates, closed polygon rings, and edit timestamp≤snapshot<Thursday. Geometry bounding boxes must overlap the requested rectangle. All six snapshots must pass and each must contain at least one LineString. This checks a well-formed historical projection; it does **not** prove topology, polygon simplicity, correct Grand Prix route, survey accuracy, physical-change dates or publication timing. The pure projector does not check self-intersection.

Bounded raw files may be held temporarily outside model inputs for the independent source/schema/hash/dual-projection audit, then deleted. Contributor columns remain undecoded. Retain only the permitted projection and provenance/hash/count receipts, with **© OpenStreetMap contributors** and **ohsome platform by HeiGIT** attribution and links to [OSM copyright](https://www.openstreetmap.org/copyright), [ODbL1.0](https://opendatacommons.org/licenses/odbl/1-0/) and [provider copyright](https://ohsome.org/copyrights). Account/software approval does not change map-data obligations.

Already completed: pure query-builder/WKB-projector tests independently check all six dates and axis order; hand-written and opposite-endian WKB; ignored future/text/outcome-field poisoning; earlier coordinate responsiveness; duplicate/malformed/nonfinite/future/clipped/unsupported/unknown-CRS/cap negatives; zero HTTP and immutable original manifest. [Implementation](experiments/e285_heigit_proposal.py), [tests](tests/test_heigit_proposal.py).

**Additional offline work completed:** the bounded runner passed fake-session tests for pre-send checkpoints, clean-prefix resume, sticky failures/interruption, locks, file caps, cache corruption/orphan rejection, default session settings and dummy-key redaction. Fake byte envelopes were deliberately NOT real Parquet. Real HTTP adapter calls were asserted zero. An additional installed-Requests preparation test verifies one dummy-key request through its actual preparation/merge path, with credential/proxy helper spies and positive controls; no adapter is reached. Actual Keychain/key/service integration remains untested. The runner has no CLI acquisition, actual Keychain provider or Parquet decoder.

**Still pending:** explicit approval, package install, actual synthetic Parquet roundtrip/column/CRS/cap checks, secure Keychain integration and authenticated request-preparation/redaction checks, then real acquisition and independent audit. Required integration tests must pass after approval and before the first map request. A source pass would permit only a separate registered geometry-validation study, followed by a distinct all61inner winner experiment if justified. No model/outer/2026 evaluation or accuracy claim follows from this proposal.
