"""E167: retirement reasons from Wikipedia race classification tables -> data/parts/wiki_retirements_2018_2025.csv"""
import re, sys, time, io
import pandas as pd, requests
from f1.data import DATA, load_results
UA = {"User-Agent": "gp-predicts research script (non-commercial)"}
MECH = ["engine", "power unit", "hydraulic", "gearbox", "electric", "electronic", "brake", "suspension", "leak", "overheat", "turbo", "mgu", "exhaust", "transmission", "battery", "driveshaft", "wheel nut", "fuel", "oil", "water", "pneumatic", "throttle", "steering", "cooling", "vibration", "halo", "chassis", "mechanical", "clutch", "differential", "energy", "wing", "floor", "undertray", "radiator", "sensor", "alternator", "ers", "drs", "control"]
INC = ["collision", "accident", "spun", "crash", "damage", "puncture", "debris", "contact"]


def cause(txt):
    t = str(txt).lower()
    if re.match(r"^\+?\d|^\d+:\d", t) or "lap" in t and re.search(r"\+\s*\d+ lap", t): return None
    if any(k in t for k in INC): return "incident"
    if any(k in t for k in MECH): return "mechanical"
    return "other"


def page_title(season, name):
    n = name.replace(" Grand Prix", "").strip()
    return f"{season}_{n.replace(' ', '_')}_Grand_Prix"


def main(first=2018, last=2025):
    res = load_results(); names = res[["season", "round", "race"]].drop_duplicates(["season", "round"]); col = "race"
    rows = []
    for _, r in names[(names.season >= first) & (names.season <= last)].drop_duplicates(["season", "round"]).iterrows():
        title = page_title(int(r.season), str(r[col]))
        try:
            h = requests.get(f"https://en.wikipedia.org/wiki/{title}", headers=UA, timeout=30); h.raise_for_status()
            tabs = [t for t in pd.read_html(io.StringIO(h.text)) if any("Retired" in str(c) for c in t.columns)]
            t = [t for t in tabs if len(t) >= 15][-1]
            tcol = [c for c in t.columns if "Retired" in str(c)][0]; dcol = [c for c in t.columns if "Driver" in str(c)][0]
            for _, x in t.iterrows():
                c = cause(x[tcol])
                rows.append(dict(season=int(r.season), round=int(r["round"]), driver=str(x[dcol]), text=str(x[tcol]), cause=c))
            print(r.season, r["round"], title, "ok", flush=True)
        except Exception as e:
            print("FAIL", r.season, r["round"], title, type(e).__name__, str(e)[:60], flush=True)
        time.sleep(1.0)
    pd.DataFrame(rows).to_csv(DATA / f"parts/wiki_retirements_{first}_{last}.csv", index=False)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
