"""Registered nine-HEAD metadata pilot; no forecast values or credentials."""
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATES = ("20190508", "20200812", "20220518")
NAMES = ("pgrb2a/gec00.t12z.pgrb2af96",
         "pgrb2a/gec00.t12z.pgrb2af096",
         "atmos/pgrb2ap5/gec00.t12z.pgrb2a.0p50.f096")
HOST = "https://noaa-gefs-pds.s3.amazonaws.com/"


def before_cutoff(row):
    """An old S3 modification receipt is supporting evidence, not full provenance."""
    if row.get("status") != 200 or not str(row.get("content_length", "")).isdigit():
        return False
    if int(row["content_length"]) <= 0:
        return False
    try:
        modified = parsedate_to_datetime(row.get("last_modified", ""))
        cutoff = datetime.fromisoformat(row["cutoff"])
        return modified.tzinfo is not None and modified <= cutoff
    except (TypeError, ValueError, OverflowError):
        return False


def main():
    rows = []
    for day in DATES:
        init = datetime.strptime(day + "12", "%Y%m%d%H").replace(tzinfo=timezone.utc)
        cutoff = init.replace(hour=0).timestamp() + 86400
        cutoff = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
        for name in NAMES:
            url = HOST + f"gefs.{day}/12/" + name
            row = dict(url=url, initialization=init.isoformat(), cutoff=cutoff,
                       accessed=datetime.now(timezone.utc).isoformat())
            try:
                with requests.head(url, timeout=20, allow_redirects=False) as response:
                    row.update(status=response.status_code,
                               content_length=response.headers.get("Content-Length"),
                               last_modified=response.headers.get("Last-Modified"),
                               etag=response.headers.get("ETag"))
            except requests.RequestException as exc:
                row["error"] = type(exc).__name__
            row["modification_before_cutoff"] = before_cutoff(row)
            rows.append(row)
    result = dict(rows=rows, dates_with_pre_cutoff_receipt=sum(
        any(r["modification_before_cutoff"] for r in rows if f"gefs.{day}/" in r["url"])
        for day in DATES), forecast_values_acquired=False,
        model_admissibility_established=False)
    (ROOT / "results/e220_gefs_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
