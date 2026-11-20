# Native SPFeed score schema: E249 source audit

The saved SPFeed numeric projection has a distinct, named score branch that merits further qualification. **No new forecasting result or physical measurement is established.** E231's failed DriverScore interpretation remains closed.

## Primary implementation evidence

An independent client declares `Scores.graph` with six fields: Steering, GforceLat, GforceLong, Brake, Performance and Throttle. Each corresponding class declares integer-list properties for 20 driver codes. Its program subscribes to SPFeed through the F1 live-timing SignalR hub.

- [Pinned schema declarations](https://github.com/claudiopizzillo/F1client/blob/ee9730b34c013fd7f0b3c335e82dbb0de650f54e/TimingData.cs).
- [Pinned client subscription](https://github.com/claudiopizzillo/F1client/blob/ee9730b34c013fd7f0b3c335e82dbb0de650f54e/Program.cs).
- [Pinned commit](https://github.com/claudiopizzillo/F1client/commit/ee9730b34c013fd7f0b3c335e82dbb0de650f54e).

GitHub reports repository creation on 2019-11-28 and author/committer timestamps on that date. The commit signature is unverified. These facts support a historical-client interpretation; user-set commit dates and current repository metadata do not certify an original publication time for every file or archived feed value.

The GitHub connector supplied the pinned source independently of the new verifier. The verifier then fetched the pinned commit/tree/blob objects through GitHub REST, checked tree membership and recomputed each Git blob SHA1 from its bytes, recording SHA256 as well. It did not execute or incorporate the third-party implementation. Synthetic removal of a field/subscription changes its checks, so the verifier can fail.

## What follows, and what remains unknown

| Question | Evidence and limit |
|---|---|
| Does the named branch appear in an independent client? | Yes: six graph fields and per-driver integer lists. |
| Is it a live-feed client? | The code subscribes to SPFeed at the live hub. Successful historical execution was not independently observed. |
| Are the values actual steering angles, brake pressure or measured g forces? | Unknown. Field names and integer lists do not establish physical units or formulas. |
| Is list coordinate 1 a verified score with coordinate 0 as its axis? | Unknown. A few saved projection examples suggest this representation; the full schema has not been qualified. |
| Do archived values match information available by historical prediction cutoffs? | Unverified. Client declarations do not establish the publication history or computation of each archived value. |
| Does this resolve DriverScore's failed lap-axis mapping? | No. No DriverScore array was transposed, relabeled or admitted. |
| Does it improve winner accuracy? | Untested. No new model or prediction was fitted. |

## Bounded next action

Before any historical feature, register a separate full-cohort qualification of the **SPFeed Scores graph** in the three existing 2019 numeric projections. Check exact driver identities, explicit indexed dictionary patches, value types, update coverage and missingness for all six declared fields. Treat initialization arrays whose indices were lost by E230 as ambiguous, rather than assigning them coordinates after looking at values. Preserve all rejected paths and counts; no field may be selected because it correlates with winners.

This can begin offline with existing projections. It does not authorize acquiring other years, guessing physical units, retuning E231, treating timing prefixes as certified publication, or fitting a winner model. Any source pass still needs semantic/provenance resolution and a separately registered temporal/incremental utility test.

The best exploratory outer winner result remains 34/70 (48.6%), versus points/ens 32/70 (45.7%). Those are reused development observations with no established superiority. The 75% exact pre-weekend winner goal remains unmet.
