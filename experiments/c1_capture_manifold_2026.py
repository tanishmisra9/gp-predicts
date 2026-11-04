"""E148: immutable prospective Manifold podium quotes at fixed pre-weekend cutoffs."""

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from experiments.c1_manifold_2024 import API, driver_for, get_bets, norm, reconstruct_quotes
from f1.data import load_results


RACES = {
    20: ("São Paulo Grand Prix", "sao paulo", "2026-11-05 00:00:00", "2026-11-06 15:30:00"),
    21: ("Las Vegas Grand Prix", "las vegas", "2026-11-19 00:00:00", "2026-11-20 00:30:00"),
    22: ("Qatar Grand Prix", "qatar", "2026-11-26 00:00:00", "2026-11-27 13:30:00"),
    23: ("Abu Dhabi Grand Prix", "abu dhabi", "2026-12-03 00:00:00", "2026-12-04 09:30:00"),
}
TERMS = ("2026 Formula 1", "2026 Grand Prix podium", "2026 F1 podium")


def capture(round_, now=None, session=None, entrants=None):
    name, alias, cutoff_text, first_text = RACES[round_]
    cutoff = pd.Timestamp(cutoff_text, tz="UTC")
    first = pd.Timestamp(first_text, tz="UTC")
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if not cutoff <= now < first:
        raise ValueError("Prospective capture requires cutoff <= now < first session")
    cutoff_ms = int(cutoff.timestamp() * 1000)
    session = session or requests.Session()
    if entrants is None:
        entrants = set(load_results().query("season == 2026 and round == 19").driver)
    discovered = {}
    for term in TERMS:
        response = session.get(API + "search-markets", params={"term": term, "sort": "newest", "filter": "all", "limit": 1000}, timeout=30)
        response.raise_for_status()
        for m in response.json():
            discovered[m["id"]] = m
    candidates = []
    for m in discovered.values():
        question = norm(m.get("question", ""))
        if not ("2026" in question and "podium" in question and alias in question
                and m.get("outcomeType") == "MULTIPLE_CHOICE"
                and m.get("createdTime", cutoff_ms + 1) <= cutoff_ms):
            continue
        response = session.get(API + "market/" + m["id"], timeout=30)
        response.raise_for_status()
        full = response.json()
        if full.get("shouldAnswersSumToOne") is not False:
            continue
        answers = {a["id"]: driver_for(a["text"], entrants)
                   for a in full.get("answers", []) if a.get("createdTime", cutoff_ms + 1) <= cutoff_ms}
        quotes, bets = reconstruct_quotes(get_bets(session, m["id"], cutoff_ms), answers, cutoff_ms)
        candidates.append({"marketId": m["id"], "question": m["question"],
                           "createdTime": m["createdTime"], "eligibleBets": len(bets),
                           "eligibleAmount": sum(b["amount"] for b in bets), "quotes": quotes})
    usable = [c for c in candidates if len(c["quotes"]) >= 3]
    chosen = sorted(usable, key=lambda c: (-c["eligibleAmount"], c["createdTime"], c["marketId"]))[0] if usable else None
    top = (sorted(chosen["quotes"], key=lambda d: (-chosen["quotes"][d]["prob"], d))[:3]
           if chosen else [])
    return {"round": round_, "race": name, "cutoff_utc": cutoff.isoformat(), "captured_utc": now.isoformat(),
            "searchedMarkets": len(discovered), "candidateMarkets": len(candidates),
            "status": "available" if chosen else "insufficient_quotes_or_market",
            "selected": chosen, "top_three": top}


def main(round_, now=None, session=None, entrants=None):
    output = Path(f"data/parts/manifold_prethu_2026_r{round_}.json")
    if output.exists():
        raise FileExistsError(output)
    record = capture(round_, now, session, entrants)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as out:
        json.dump(record, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")
    print(output, record["status"], "top three", record["top_three"])
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("round", type=int, choices=RACES)
    main(parser.parse_args().round)
