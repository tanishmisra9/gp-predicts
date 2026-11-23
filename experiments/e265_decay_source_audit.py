"""Pinned, static creator-method audit; never imports third-party model code."""
import ast
import base64
import hashlib
import json
import subprocess
from f1.data import ROOT

REPO = "sussolinoss/sports-ml-predictions"
PIN = "99b1a4bdd127c1c9e9f107e6a754f8e07e461b10"
BLOBS = {
    "f1/f1_podium.py": "a6ac8c86d88cb93f04d37af2eebec9475f35c37e",
    "f1/experiments/f1_exp_decay.py": "2ff21748153b7f7f52d74bb55d1952aa76c44d9a",
    "f1/experiments/f1_exp_decay_verify.py": "dbddd9dbba5620d0c7893c9aa9357b815c79902c",
    "f1/experiments/f1_exp_tau_fine.py": "e6eb2d7c12a1a07c260a90b79343810eaea35c4c",
    "f1/experiments/f1_exp_ablation.py": "a3331e8c3984160a69deb13cfc4e7adae7b693c7",
    "f1/documentation/f1_paper.tex": "8b10403a32ac4f76a1db2ee7132b79a6dcf33351",
}


def facts(source):
    tree = ast.parse(source)
    tau_loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == "tau"]
    scored = [n for n in tau_loops if any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        and c.func.id == "_precision_at3" and c.args
        and isinstance(c.args[0], ast.Name) and c.args[0].id == "te"
        for c in ast.walk(n))]
    medians = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == "nanmedian"]
    return dict(tau_values=[ast.literal_eval(n.iter) for n in tau_loops],
                direct_test_precision_call_inside_tau_loop=bool(scored),
                test_features_in_median=any(isinstance(n, ast.Name) and n.id == "Xte"
                                          for call in medians for n in ast.walk(call)))


def main():
    tree = json.loads(subprocess.check_output([
        "gh", "api", f"repos/{REPO}/git/trees/{PIN}?recursive=1"]))
    assert not tree.get("truncated", False)
    paths = {r["path"]: r["sha"] for r in tree["tree"] if r["type"] == "blob"}
    assert all(paths[p] == sha for p, sha in BLOBS.items())
    saved = {}; sources = {}
    for path, sha in BLOBS.items():
        raw = json.loads(subprocess.check_output(["gh", "api", f"repos/{REPO}/git/blobs/{sha}"]))
        body = base64.b64decode(raw["content"])
        assert hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() == sha == raw["sha"]
        saved[path] = dict(git_blob=sha, sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
        # Never display or parse the paper's outcome tables; hash bytes only.
        if path.endswith(".py"):
            sources[path] = body.decode()
    decay = sources["f1/experiments/f1_exp_decay.py"]
    measured = {path: facts(source) for path, source in sources.items()}
    assert measured["f1/experiments/f1_exp_decay.py"] == dict(
        tau_values=[[None, 5.0, 3.0, 2.0, 1.5]], direct_test_precision_call_inside_tau_loop=True,
        test_features_in_median=True)
    for suffix in ("decay_verify", "tau_fine"):
        assert measured[f"f1/experiments/f1_exp_{suffix}.py"]["direct_test_precision_call_inside_tau_loop"]
    # Distinguish validation scoring and train-only imputation from observed code.
    assert not facts(decay.replace("_precision_at3(te,", "_precision_at3(va,"))["direct_test_precision_call_inside_tau_loop"]
    assert not facts(decay.replace("np.vstack([Xtr, Xva, Xte])", "Xtr"))["test_features_in_median"]
    podium = ast.parse(sources["f1/f1_podium.py"])
    cols = next(ast.literal_eval(n.value) for n in podium.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "FEATURE_COLS" for t in n.targets))
    assert {"grid", "quali_gap_ms", "teammate_gap_ms"} <= set(cols)
    result = dict(repo=REPO, pin=PIN, blobs=saved, static_facts=measured,
                  headline_uses_weekend_features=True, external_code_executed=False,
                  model_fit=False, forecast_replication=False, new_2026_evaluation=False,
                  individual_2026_table_exposed_during_initial_paper_read=True,
                  author_adaptive_test_selection_established=False)
    with (ROOT / "results/e265_decay_source_audit.json").open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    print("E265 pinned hashes/static tau scoring/test-feature median/negative-controls audit PASS")


if __name__ == "__main__":
    main()
