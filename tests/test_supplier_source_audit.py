"""E269 source metadata scope and required-field negative controls."""
from experiments.e269_supplier_source_audit import BLOBS, metadata_scope, required_string


def main():
    assert all(not path.startswith("src/data/") for path in BLOBS)
    tree = [dict(type="blob", path=p, sha="metadata-only") for p in [
        "src/data/seasons/2018/races/01-australia/race-results.yml",
        "src/data/seasons/2022/races/22-abu-dhabi/race-results.yml",
        "src/data/seasons/2019/entrants.yml",
        "src/data/seasons/2022/races/01-bahrain/sprint-race-results.yml",
        "src/data/seasons/2026/races/01-australia/race-results.yml",
        "src/data/seasons/2017/entrants.yml",
        "src/data/drivers/example.yml"]]
    races, entrants = metadata_scope(tree)
    assert [(r["season"], r["round"]) for r in races] == [(2018, 1), (2022, 22)]
    assert [r["season"] for r in entrants] == [2019]
    assert metadata_scope([]) == ([], [])
    schema = {"definitions": {"Row": {"required": ["engine"], "properties": {"engine": {"type": "string"}}}}}
    assert required_string(schema, "Row", "engine")
    schema["definitions"]["Row"]["required"] = []
    assert not required_string(schema, "Row", "engine")
    schema["definitions"]["Row"]["required"] = ["engine"]
    schema["definitions"]["Row"]["properties"]["engine"]["type"] = ["string", "null"]
    assert not required_string(schema, "Row", "engine")
    print("E269 technical-body allowlist/old-year+race-only metadata/nullable+missing negative controls PASS")


if __name__ == "__main__":
    main()
