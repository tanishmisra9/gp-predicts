"""E147 source-time guard: target-weekend and later bets cannot change a quote."""

import json

from experiments.c1_manifold_2024 import OUT, reconstruct_quotes


def bet(id_, created, fill, prob):
    return {"id": id_, "createdTime": created, "fills": [{"timestamp": fill}],
            "probAfter": prob, "answerId": "a", "amount": 10,
            "isFilled": True, "isCancelled": False}


def main():
    cutoff = 1000
    before = bet("before", 900, 900, 0.6)
    base, _ = reconstruct_quotes([before], {"a": "driver"}, cutoff)
    later_created = bet("later", 1100, 1100, 0.99)
    later_filled = bet("filled_later", 950, 1100, 0.01)
    changed, _ = reconstruct_quotes([before, later_created, later_filled], {"a": "driver"}, cutoff)
    assert base == changed == {"driver": {"prob": 0.6, "betTime": 900, "lastFillTime": 900, "betId": "before"}}
    earlier, _ = reconstruct_quotes([before, bet("earlier", 950, 950, 0.7)], {"a": "driver"}, cutoff)
    assert earlier["driver"]["prob"] == 0.7
    saved = json.loads(OUT.read_text())
    assert all(q["betTime"] <= c["cutoffMs"] and q["lastFillTime"] <= c["cutoffMs"]
               for c in saved["candidates"] for q in c["quotes"].values())
    assert saved["availableCount"] == len(saved["selected"])
    print("Manifold source-time guard passed")


if __name__ == "__main__":
    main()
