# E285 source notice — proposal only; no map data acquired

Proposed map source: HeiGIT ohsomeAPIv2, six historical2019/2022snapshots listed in e285_heigit_query_proposal.json. Attribution: **© OpenStreetMap contributors; ohsome platform by HeiGIT**.

Map licensing: [OpenStreetMap copyright](https://www.openstreetmap.org/copyright), [ODbL1.0](https://opendatacommons.org/licenses/odbl/1-0/), [provider copyright](https://ohsome.org/copyrights). The prior E280approval is recorded and limited to its original acquisition; it does not approve the additional E285actions.

Transformation proposed: select historical way identity/version/minor-version/edit-time and WKB coordinates; verify raceway classification/chronology/CRS; omit contributor/text/future-validity fields. No geometry, physical-change or prediction result exists. Map timestamps are database-state evidence, not certified survey/publication dates.

Proposed decoder: PyArrow26.0.1, Apache-2.0, pinned compatible wheel/hash in e285_source_metadata.json. No package downloaded/installed, service terms accepted, account created, API key read or request sent. Keep package licence/notices and install receipt if explicitly approved.

The local proposal/projector and its synthetic fixtures are newly authored research code; fixtures contain no acquired OSM records. Actual Parquet/transport/source checks remain unexecuted.

2026-11-23 update: explicit scoped approval received. Same pinned Apache-2.0 wheel installed in isolated research .venv-e285 after shared NumPy1.x incompatibility; initial shared addition removed. Packaged licences/notices retained and hashed in e285_decoder_installation.json. No map data/account/key action yet; preceding text is historical preapproval status.
