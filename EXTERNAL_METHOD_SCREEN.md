# E253: external driver-skill method screen

This review found no demonstrated75% pre-weekend exact-winner result. No external ratings or junior-series results were imported.

## Candidate and scope

The [F1/F2 skill project](https://github.com/wilchatham/F1-F2-driver-skill-model/tree/4ab7ae5379bff1f4fa5b616bd629d2e1192981f8) studies qualifying-based driver rankings. Its [pinned notebook](https://github.com/wilchatham/F1-F2-driver-skill-model/blob/4ab7ae5379bff1f4fa5b616bd629d2e1192981f8/F1_Bradley_Terry_Model.ipynb) includes teammate Bradley–Terry comparisons, a corrected field-relative scale alignment, and a later mixed-effects model. The reviewed description and ranking code do not supply an all-race chronological pre-weekend winner score. Copying fitted2021–24 ratings into earlier forecasts would be inadmissible.

The junior-data route remains subject to the existingE138 restrictions. Reading methodology does not authorize extracting official F2 pages/PDFs or importing rehosted data derived from them.

## Independent identification check

Consider log lap time = circuit baseline + driver effect + car effect. Smaller effects mean faster performance. Two disconnected groups have driversA1/A2 in carA andB1/B2 in carB.

| Effect | World1 | World2 |
|---|---:|---:|
| A1 | −.015 | +.025 |
| A2 | −.005 | +.035 |
| B1 | +.005 | −.035 |
| B2 | +.015 | −.025 |
| CarA | +.020 | −.020 |
| CarB | −.020 | +.020 |

Every observed driver+car sum is identical between worlds. Driver effects and car effects each sum to zero in both worlds. Therefore observed laps, teammate gaps, normalized ratios and field-relative pace are unchanged. Yet the best driver on equal machinery isA1 inWorld1 andB1 inWorld2.

This is an exact rational-arithmetic counterexample. A separate design-matrix null-vector check includes both centering constraints. A negative control that shifts only drivers changes the observations; adding a genuineA1/B1 teammate comparison also breaks this ambiguity.

This limits an unconditional claim of recovering global car-neutral skill from disconnected comparisons alone. It does **not** invalidate conditional mixed-effects priors or prove that forecasting total driver+car pace is impossible. Priors can choose a decomposition, and useful forecasts can exist without uniquely identified components. They still need chronological validation against the actual winner objective.

## Decision and distinct next avenue

Do not reproduce this project as evidence of75% winner accuracy. It supplies no newly admissible dataset or demonstrated improvement over our baselines. Existing portable-graph and joint-filter failures remain closed.

A distinct source question is worth qualifying: can **older structured qualifying teammate links**, entirely before2018, connect groups that are disconnected in the current timing-history window? This would add actual earlier observations rather than choose arbitrary offsets. Before any acquisition, audit existing local historical files and the earlier stop rules, fix one source/year range, verify exact driver/team identities, then measure topology only. A source/topology pass would still need a separately registered temporal forecast test; it would not reopenE116/E175 by itself or justify importing external skill ratings. No historical data for this avenue was acquired inE253.

Winner accuracy is unchanged. The best exploratory outer result remains34/70(48.6%), versus points/ens32/70(45.7%), with no established superiority and the documented roster-timing assumption. The75% goal remains active and unmet.
