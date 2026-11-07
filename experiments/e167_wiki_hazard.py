"""E167: rerun E101's cause-specific hazard with Wikipedia retirement causes filling Jolpica's missing 2023-25 labels."""
import unicodedata
import pandas as pd
import experiments.a4_mechanical as A
from f1.data import DATA, load_results as _load


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()


def patched():
    r = _load(); w = pd.read_csv(DATA / "parts/wiki_retirements_2018_2025.csv")
    w["key"] = w.driver.map(lambda n: norm(n).split()[-1]); r["key"] = r.driver.map(lambda d: norm(d).split("_")[-1])
    m = r.merge(w[["season", "round", "key", "cause"]].drop_duplicates(["season", "round", "key"]), on=["season", "round", "key"], how="left")
    dnf = ~(m.status.isin(["Finished", "Lapped"]) | m.status.str.startswith("+"))
    print("DNF rows", int(dnf.sum()), "| with wiki cause", int((dnf & m.cause.notna()).sum()),
          "| by cause:", m.loc[dnf, "cause"].value_counts(dropna=False).to_dict())
    m.loc[dnf & (m.cause == "mechanical"), "status"] = "Mechanical"
    m.loc[dnf & (m.cause == "incident"), "status"] = "Collision"
    print(m.assign(dnf=dnf).groupby("season").apply(lambda g: pd.Series({"dnf": int(g.dnf.sum()), "mech_named": int((g.dnf & g.status.isin(A.MECH)).sum()), "unknown": int((g.dnf & g.status.isin(A.UNKNOWN)).sum())}), include_groups=False).T.to_string())
    return m.drop(columns=["key", "cause"])


A.load_results = patched
if __name__ == "__main__":
    A.main()
