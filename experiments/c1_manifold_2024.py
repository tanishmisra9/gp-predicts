"""E147: fixed, read-only Manifold historical pre-weekend quote audit."""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from f1.data import load_results


API = "https://api.manifold.markets/v0/"
TERMS = ("2024 Formula 1", "2024 Grand Prix podium", "2024 F1 podium")
ALIASES = {
    1: ("bahrain",), 2: ("saudi",), 3: ("australia",),
    4: ("japan",), 5: ("chin",), 6: ("miami",),
    7: ("emilia", "imola"), 8: ("monaco",), 9: ("canad",),
    10: ("spanish", "spain"), 11: ("austria",), 12: ("british",),
    13: ("hungar",), 14: ("belg",), 15: ("dutch",),
    16: ("italian", "monza"), 17: ("azerbaijan",), 18: ("singapore",),
    19: ("united states", "austin"), 20: ("mexico",),
    21: ("sao paulo", "são paulo", "brazil"), 22: ("las vegas",),
    23: ("qatar",), 24: ("abu dhabi",),
}
OUT = Path("data/parts/manifold_2024_audit.json")


def norm(value):
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", value.lower()).encode("ascii", "ignore").decode()).strip()


def race_for(question):
    q = norm(question)
    if "2024" not in q or "podium" not in q or not ("prix" in q or "gran premio" in q):
        return None
    matched = [r for r, aliases in ALIASES.items() if any(norm(a) in q for a in aliases)]
    return matched[0] if len(matched) == 1 else None


def eligible_bets(bets, cutoff_ms):
    """A quote cannot depend on any target-weekend or later fill."""
    return [b for b in bets if b.get("createdTime", cutoff_ms + 1) <= cutoff_ms
            and b.get("isFilled") is True and b.get("isCancelled") is False
            and not b.get("isRedemption") and not b.get("isAnte")
            and b.get("limitProb") is None and b.get("answerId")
            and isinstance(b.get("probAfter"), (int, float))
            and 0 <= b["probAfter"] <= 1 and b.get("amount", 0) > 0
            and b.get("fills") and all(f.get("timestamp", cutoff_ms + 1) <= cutoff_ms for f in b["fills"])]


def get_bets(session, market_id, cutoff_ms):
    rows, cursor = [], None
    while True:
        params = {"contractId": market_id, "beforeTime": cutoff_ms + 1, "limit": 1000}
        if cursor:
            params["before"] = cursor
        response = session.get(API + "bets", params=params, timeout=30)
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < 1000:
            return rows
        cursor = page[-1]["id"]


def driver_for(answer, entrants):
    text = norm(answer)
    hits = [d for d in entrants if norm(d.replace("_", " ").split()[-1]) in text.split()]
    return hits[0] if len(hits) == 1 else None


def reconstruct_quotes(bets, answers, cutoff_ms):
    latest = {}
    valid = eligible_bets(bets, cutoff_ms)
    for b in sorted(valid, key=lambda x: (x["createdTime"], x["id"])):
        if answers.get(b["answerId"]):
            latest[answers[b["answerId"]]] = {"prob": b["probAfter"], "betTime": b["createdTime"],
                                               "lastFillTime": max(f["timestamp"] for f in b["fills"]), "betId": b["id"]}
    return latest, valid


def audit():
    session = requests.Session()
    discovered = {}
    for term in TERMS:
        response = session.get(API + "search-markets", params={"term": term, "sort": "newest", "filter": "all", "limit": 1000}, timeout=30)
        response.raise_for_status()
        for market in response.json():
            discovered[market["id"]] = market
    races = load_results().query("season == 2024")
    race_meta = races[["round", "date", "race"]].drop_duplicates().set_index("round")
    candidates = []
    for m in discovered.values():
        r = race_for(m.get("question", ""))
        if r is None or m.get("outcomeType") != "MULTIPLE_CHOICE":
            continue
        cutoff_ms = int((race_meta.loc[r, "date"] - pd.Timedelta(days=3)).timestamp() * 1000)
        if m.get("createdTime", cutoff_ms + 1) > cutoff_ms:
            continue
        response = session.get(API + "market/" + m["id"], timeout=30)
        response.raise_for_status()
        full = response.json()
        if full.get("shouldAnswersSumToOne") is not False:
            continue
        entrants = set(races.loc[races["round"] == r, "driver"])
        answers = {a["id"]: driver_for(a["text"], entrants)
                   for a in full.get("answers", []) if a.get("createdTime", cutoff_ms + 1) <= cutoff_ms}
        latest, bets = reconstruct_quotes(get_bets(session, m["id"], cutoff_ms), answers, cutoff_ms)
        candidates.append({"round": r, "marketId": m["id"], "question": m["question"],
                           "createdTime": m["createdTime"], "cutoffMs": cutoff_ms,
                           "eligibleBets": len(bets), "eligibleAmount": sum(b["amount"] for b in bets),
                           "quotes": latest})
    selected = {}
    for r in range(1, 25):
        options = [c for c in candidates if c["round"] == r and len(c["quotes"]) >= 3]
        if options:
            selected[r] = sorted(options, key=lambda c: (-c["eligibleAmount"], c["createdTime"], c["marketId"]))[0]
    output = {"searchedMarkets": len(discovered), "candidateCount": len(candidates),
              "availableCount": len(selected), "candidates": candidates,
              "selected": selected,
              "status": [{"round": r, "race": race_meta.loc[r, "race"],
                          "cutoffMs": int((race_meta.loc[r, "date"] - pd.Timedelta(days=3)).timestamp() * 1000),
                          "candidateMarkets": sum(c["round"] == r for c in candidates),
                          "marketId": selected[r]["marketId"] if r in selected else None,
                          "quotedDrivers": len(selected[r]["quotes"]) if r in selected else 0}
                         for r in range(1, 25)]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    return output


def score(audit_data):
    if audit_data["availableCount"] < 10:
        return None
    import numpy as np
    import f1.walkforward as W
    from f1.data import DATA
    from f1.features import build_features, pre_weekend_cols

    X = build_features(load_results(), pd.read_csv(DATA / "racepace_ff1.csv"))
    F = pre_weekend_cols(X)
    W.MODELS["standings"] = lambda tr, te, f, **kw: (-te.d_season_pts.fillna(0) * 100 + te.d10_order.fillna(11) * 1e-3).values
    W.MODELS["logit"] = lambda tr, te, f, **kw: W.m_logit_pod(tr, te, F, C=0.03, hl=100)

    def shuffled(tr, te, f, **kw):
        t = tr.copy()
        t["podium"] = np.random.default_rng(0).permutation(t.podium.values)
        return W.m_logit_pod(t, te, F, C=0.03, hl=100)

    W.MODELS["null_shuffled_labels"] = shuffled
    R, P = W.walk_forward(X, [2024], ["standings", "logit", "null_shuffled_labels"],
                          ens={"ens": ["standings", "logit"]}, tie=None)
    selected = {int(r): c for r, c in audit_data["selected"].items()}
    rows = []
    for r, c in sorted(selected.items()):
        actual = set(X[(X.season == 2024) & (X["round"] == r)].nsmallest(3, "order").driver)
        market = sorted(c["quotes"], key=lambda d: (-c["quotes"][d]["prob"], d))[:3]
        row = {"round": r, "marketId": c["marketId"], "quotes": len(c["quotes"]),
               "marketPick": market, "actual": sorted(actual), "market": len(set(market) & actual) / 3}
        for model in ("standings", "ens", "null_shuffled_labels"):
            p = P[(P.season == 2024) & (P["round"] == r) & (P.model == model)]
            row[model] = float(R[(R.season == 2024) & (R["round"] == r) & (R.model == model)].podium.iloc[0])
            row[model + "Pick"] = p.sort_values(["score", "driver"]).head(3).driver.tolist()
        rows.append(row)
    result = {"n": len(rows), "strict": {name: float(np.mean([x[name] for x in rows]))
                                             for name in ("market", "standings", "ens", "null_shuffled_labels")},
              "paired": {name: {"diff": float(d), "ci": [float(z) for z in ci], "pNonpositive": float(p)}
                         for name in ("standings", "ens", "null_shuffled_labels")
                         for d, ci, p in [W.paired_boot([x["market"] for x in rows], [x[name] for x in rows])]},
              "races": rows}
    audit_data["score"] = result
    OUT.write_text(json.dumps(audit_data, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    result = audit()
    print("searched", result["searchedMarkets"], "candidates", result["candidateCount"], "available", result["availableCount"])
    for row in result["status"]:
        print(row["round"], row["race"], "candidates", row["candidateMarkets"], "quoted", row["quotedDrivers"])
    scored = score(result)
    if scored:
        print("strict overlap", scored["strict"])
        print("paired market minus references", scored["paired"])
        for row in scored["races"]:
            print(row["round"], "market", round(row["market"], 3), "standings", round(row["standings"], 3),
                  "ens", round(row["ens"], 3), "null", round(row["null_shuffled_labels"], 3),
                  "marketPick", row["marketPick"])
