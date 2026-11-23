"""E269 technical schema/metadata audit only; F1DB record acquisition requires approval."""
import base64
import hashlib
import json
import re
import subprocess

from f1.data import ROOT, load_results

REPO = "f1db/f1db"
PIN = "9dbbb9f67dc5fa3d499061efb56d24aa4f1e261d"
BLOBS = {
    "LICENSE": "4ea99c213c5c0c005ae4e80df8e52169d06896ec",
    "README.md": "1ff1c2b35ff0e7ba8a7a496397d8342f5e7f336d",
    "src/schema/current/splitted/f1db-races-race-results.schema.json": "ad4c7c0ef9a041d415d624eab6e53d07b5c2cf95",
    "src/schema/current/splitted/f1db-engines.schema.json": "a4bdfc68e2d58c207b571f43ffdbfa2e4a1e0c1a",
    "src/schema/current/splitted/f1db-seasons-entrants-engines.schema.json": "e3c54c1d5865e88158e4351bbf6188b41109e3bb",
    "src/schema/current/splitted/f1db-seasons-entrants-constructors.schema.json": "97c8a0e2352bf3472c58cfc10c80aeba2f8dafdf",
    "build-logic/src/main/kotlin/com/f1db/plugin/reader/F1DBReader.kt": "1f0c883c26858b48d90d84dac9db1b2bde797c51",
    "build-logic/src/main/kotlin/com/f1db/plugin/reader/databind/RaceResultConverter.kt": "4398b35a8c7ea2e9d9ac12ba77ac14f451e60540",
    "build-logic/src/main/kotlin/com/f1db/plugin/reader/databind/RaceResultMixIn.kt": "04f29a82597921e0bf6f1958ace9a45b15f1dd03",
    "build-logic/src/main/kotlin/com/f1db/plugin/schema/mapper/RaceRaceResultMapper.kt": "6dfec0d4caf1c8ebf3d9a64d29fa68faba417519",
}


def required_string(schema, definition, field):
    item = schema["definitions"][definition]
    return field in item["required"] and item["properties"][field]["type"] == "string"


def metadata_scope(tree):
    races = []; entrants = []
    for row in tree:
        if row["type"] != "blob":
            continue
        path = row["path"]
        race = re.fullmatch(r"src/data/seasons/(2018|2019|2020|2021|2022)/races/(\d+)-[^/]+/race-results.yml", path)
        entrant = re.fullmatch(r"src/data/seasons/(2018|2019|2020|2021|2022)/entrants.yml", path)
        if race:
            races.append(dict(season=int(race[1]), round=int(race[2]), path=path, git_blob=row["sha"]))
        elif entrant:
            entrants.append(dict(season=int(entrant[1]), path=path, git_blob=row["sha"]))
    return races, entrants


def main():
    tree = json.loads(subprocess.check_output(["gh", "api", f"repos/{REPO}/git/trees/{PIN}?recursive=1"]))
    assert not tree.get("truncated", False)
    paths = {r["path"]: r["sha"] for r in tree["tree"] if r["type"] == "blob"}
    decoded = {}; sources = {}
    # Only technical files in this exact allowlist are fetched, never src/data bodies.
    for path, digest in BLOBS.items():
        assert not path.startswith("src/data/") and paths[path] == digest
        blob = json.loads(subprocess.check_output(["gh", "api", f"repos/{REPO}/git/blobs/{digest}"]))
        raw = base64.b64decode(blob["content"])
        assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest() == digest == blob["sha"]
        sources[path] = dict(git_blob=digest, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        decoded[path] = raw.decode()
    race_path = "src/schema/current/splitted/f1db-races-race-results.schema.json"
    race_schema = json.loads(decoded[race_path])
    assert required_string(race_schema, "RaceResult", "engineManufacturerId")
    assert required_string(race_schema, "RaceResult", "driverNumber")
    changed = json.loads(decoded[race_path]); changed["definitions"]["RaceResult"]["required"].remove("engineManufacturerId")
    assert not required_string(changed, "RaceResult", "engineManufacturerId")
    changed["definitions"]["RaceResult"]["required"].append("engineManufacturerId")
    changed["definitions"]["RaceResult"]["properties"]["engineManufacturerId"]["type"] = ["string", "null"]
    assert not required_string(changed, "RaceResult", "engineManufacturerId")
    engine_schema = json.loads(decoded["src/schema/current/splitted/f1db-seasons-entrants-engines.schema.json"])
    assert required_string(engine_schema, "SeasonEntrantEngine", "engineId")
    assert required_string(engine_schema, "SeasonEntrantEngine", "engineManufacturerId")
    assert "Attribution 4.0 International" in decoded["LICENSE"] and "Section 3 -- License Conditions." in decoded["LICENSE"]
    races, entrants = metadata_scope(tree["tree"])
    official = load_results().loc[lambda d: d.season.between(2018, 2022), ["season", "round"]].drop_duplicates()
    assert len(races) == 103 and len(entrants) == 5
    assert {(r["season"], r["round"]) for r in races} == set(official.itertuples(index=False, name=None))
    counts = {year: sum(r["season"] == year for r in races) for year in range(2018, 2023)}
    result = dict(repo=REPO, pin=PIN, technical_sources=sources, race_file_metadata=races,
                  entrant_file_metadata=entrants, race_file_counts=counts,
                  licence_label="CC BY 4.0", data_licence_approval_received=False,
                  data_record_bodies_acquired=False, actual_record_coverage_verified=False,
                  physical_supplier_mapping_verified=False, preweekend_history_verified=False,
                  winner_fit=False, source_lead_qualified_for_model=False)
    with (ROOT / "results/e269_supplier_source_manifest.json").open("x") as file:
        json.dump(result, file, indent=2); file.write("\n")
    print("E269 pinned technical/schema hashes/103race+5entrant metadata/negative controls PASS; records not acquired")


if __name__ == "__main__":
    main()
