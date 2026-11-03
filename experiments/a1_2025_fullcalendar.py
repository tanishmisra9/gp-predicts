"""E142: unchanged 2024 X/Y profile vote on every 2025 Grand Prix."""

import pandas as pd

import f1.walkforward as W
from experiments.a1_2025_podium_vote import main as podium_vote
from experiments.a1_2025_team_transfer import PART as PILOT, evaluate, extract
from f1.data import DATA, load_results


SESSIONS = {
    "bahrain": 9472, "jeddah": 9480, "albert_park": 9488, "suzuka": 9496,
    "shanghai": 9673, "miami": 9507, "imola": 9515, "monaco": 9523,
    "villeneuve": 9531, "catalunya": 9539, "red_bull_ring": 9550,
    "silverstone": 9558, "hungaroring": 9566, "spa": 9574,
    "zandvoort": 9582, "monza": 9590, "baku": 9598, "marina_bay": 9606,
    "americas": 9617, "rodriguez": 9625, "interlagos": 9636,
    "vegas": 9644, "losail": 9655, "yas_marina": 9662,
}
PART = DATA / "parts/race_xy_multistint_all_2024.csv"


def main():
    if not PART.exists():
        pd.read_csv(PILOT).to_csv(PART, index=False)
    source = extract(SESSIONS, PART)
    print("profile driver-races by circuit", source.groupby("circuit").code.nunique().reindex(SESSIONS).to_dict())
    results = load_results().query("season <= 2025")
    pace = pd.read_csv(DATA / "racepace_ff1.csv")
    r, _ = evaluate(source, results, pace, SESSIONS)
    print("2025 team pace Spearman", r.round(4).to_string())
    print("mean", r[["standings", "profile", "prior_pace", "blend"]].mean().round(4).to_dict())
    defined = r.standings.notna()
    print("profile minus standings paired on defined standings races", defined.sum(),
          W.paired_boot(r.loc[defined, "profile"], r.loc[defined, "standings"])[:2])
    podium_vote(PART, SESSIONS)


if __name__ == "__main__":
    main()
