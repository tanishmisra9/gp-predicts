"""Static circuit descriptors known before any weekend -> data/circuit_static.csv: altitude (Open-Meteo elevation API at the circuit's coordinates)."""
import pandas as pd
from f1.data import DATA
from f1.weather import _circuits, _s

def build():
    xy = _circuits(); rows = []
    for cid, (lat, lon) in xy.items():
        j = _s.get("https://api.open-meteo.com/v1/elevation", params=dict(latitude=lat, longitude=lon), timeout=30).json()
        rows.append(dict(circuit=cid, lat=lat, lon=lon, altitude=(j.get("elevation") or [None])[0]))
    d = pd.DataFrame(rows); d.to_csv(DATA / "circuit_static.csv", index=False); return d

if __name__ == "__main__":
    d = build(); print(d.sort_values("altitude", ascending=False).head(8).to_string(index=False))
