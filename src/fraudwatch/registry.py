"""Model registry: SQLite metadata + filesystem artifacts.

Promotion/rollback semantics mirror MLflow's registry workflow (versions,
a champion pointer ~= the @champion alias, explicit promotion) -- see
docs/decisions.md for why this equivalent is shipped, and mlflow_export.py
for the MLflow integration used when MLflow is installed.

Artifact trust: a model file is only loaded if its SHA-256 matches the
hash recorded at registration time and a manifest is present. A tampered
or unregistered artifact is rejected, never unpickled.
"""
import hashlib, json, pickle, sqlite3, time
from pathlib import Path

from . import config


def _db(db_path=None):
    db_path = Path(db_path) if db_path else config.REGISTRY_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS models(
        version INTEGER PRIMARY KEY, kind TEXT, path TEXT, sha256 TEXT,
        threshold REAL, metrics TEXT, params TEXT, fingerprint TEXT, created REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS promotion_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT, version INTEGER, action TEXT,
        passed INTEGER, detail TEXT, ts REAL)""")
    return con


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(model, kind, threshold, metrics, params, fingerprint, models_dir=None, db_path=None) -> int:
    models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    con = _db(db_path)
    row = con.execute("SELECT COALESCE(MAX(version),0)+1 FROM models").fetchone()
    version = int(row[0])
    vdir = models_dir / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)
    path = vdir / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    digest = _sha(path)
    manifest = {"version": version, "kind": kind, "sha256": digest,
                "threshold": threshold, "fingerprint": fingerprint}
    (vdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    con.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?)",
                (version, kind, str(path), digest, threshold,
                 json.dumps(metrics), json.dumps(params), fingerprint, time.time()))
    con.commit(); con.close()
    return version


def get(version, db_path=None) -> dict | None:
    con = _db(db_path)
    r = con.execute("SELECT version,kind,path,sha256,threshold,metrics,params,fingerprint FROM models WHERE version=?",
                    (version,)).fetchone()
    con.close()
    if not r:
        return None
    return {"version": r[0], "kind": r[1], "path": r[2], "sha256": r[3], "threshold": r[4],
            "metrics": json.loads(r[5]), "params": json.loads(r[6]), "fingerprint": r[7]}


def list_versions(db_path=None) -> list[dict]:
    con = _db(db_path)
    rows = con.execute("SELECT version FROM models ORDER BY version").fetchall()
    con.close()
    return [get(r[0], db_path) for r in rows]


def champion(db_path=None) -> int | None:
    con = _db(db_path)
    r = con.execute("SELECT value FROM state WHERE key='champion'").fetchone()
    prev = con.execute("SELECT value FROM state WHERE key='previous_champion'").fetchone()
    con.close()
    return int(r[0]) if r else None


def _set_champion(version, db_path=None):
    con = _db(db_path)
    cur = con.execute("SELECT value FROM state WHERE key='champion'").fetchone()
    if cur and int(cur[0]) != version:
        con.execute("INSERT OR REPLACE INTO state VALUES('previous_champion',?)", (cur[0],))
    con.execute("INSERT OR REPLACE INTO state VALUES('champion',?)", (str(version),))
    con.commit(); con.close()


# Promotion gates: declared tradeoff -- catch fraud (recall) while keeping
# false positives operationally tolerable, and beat the current champion.
GATES = {"min_pr_auc": 0.25, "min_recall": 0.35, "min_precision": 0.20,
         "max_fpr": 0.05, "min_pr_auc_improvement_over_champion": 0.0}


def evaluate_gates(version, db_path=None) -> dict:
    rec = get(version, db_path)
    if rec is None:
        return {"passed": False, "reasons": ["version not registered"]}
    m = rec["metrics"]
    reasons = []
    if m["pr_auc"] < GATES["min_pr_auc"]:
        reasons.append(f"pr_auc {m['pr_auc']:.3f} < {GATES['min_pr_auc']}")
    if m["recall"] < GATES["min_recall"]:
        reasons.append(f"recall {m['recall']:.3f} < {GATES['min_recall']}")
    if m["precision"] < GATES["min_precision"]:
        reasons.append(f"precision {m['precision']:.3f} < {GATES['min_precision']}")
    if m["fpr"] > GATES["max_fpr"]:
        reasons.append(f"fpr {m['fpr']:.3f} > {GATES['max_fpr']}")
    champ = champion(db_path)
    if champ is not None and champ != version:
        cm = get(champ, db_path)["metrics"]
        if m["pr_auc"] < cm["pr_auc"] + GATES["min_pr_auc_improvement_over_champion"]:
            reasons.append(f"pr_auc {m['pr_auc']:.3f} does not beat champion v{champ} ({cm['pr_auc']:.3f})")
    return {"passed": not reasons, "reasons": reasons, "gates": GATES, "metrics": m}


def promote(version, db_path=None) -> dict:
    result = evaluate_gates(version, db_path)
    con = _db(db_path)
    con.execute("INSERT INTO promotion_log(version,action,passed,detail,ts) VALUES (?,?,?,?,?)",
                (version, "promote", int(result["passed"]),
                 json.dumps(result["reasons"]), time.time()))
    con.commit(); con.close()
    if result["passed"]:
        _set_champion(version, db_path)
    result["champion_after"] = champion(db_path)
    return result


def rollback(db_path=None) -> dict:
    con = _db(db_path)
    prev = con.execute("SELECT value FROM state WHERE key='previous_champion'").fetchone()
    cur = champion(db_path)
    con.close()
    if not prev:
        return {"rolled_back": False, "reason": "no previous champion recorded", "champion": cur}
    target = int(prev[0])
    # Rolling back swaps champion/previous so a second rollback rolls forward.
    con = _db(db_path)
    con.execute("INSERT OR REPLACE INTO state VALUES('previous_champion',?)", (str(cur),))
    con.execute("INSERT OR REPLACE INTO state VALUES('champion',?)", (str(target),))
    con.execute("INSERT INTO promotion_log(version,action,passed,detail,ts) VALUES (?,?,?,?,?)",
                (target, "rollback", 1, json.dumps([f"from v{cur}"]), time.time()))
    con.commit(); con.close()
    return {"rolled_back": True, "from_version": cur, "champion": target}


def set_initial_champion(version, db_path=None):
    """Bootstrap deployment only: the first model ships without a gate
    comparison (there is no champion to beat). Documented in decisions.md."""
    _set_champion(version, db_path)
    con = _db(db_path)
    con.execute("INSERT INTO promotion_log(version,action,passed,detail,ts) VALUES (?,?,?,?,?)",
                (version, "initial_deployment", 1, json.dumps(["bootstrap"]), time.time()))
    con.commit(); con.close()


def load_model(version, db_path=None):
    """Load a registered model ONLY if its artifact hash still matches."""
    rec = get(version, db_path)
    if rec is None:
        raise ValueError(f"model version {version} is not registered")
    path = Path(rec["path"])
    manifest_path = path.parent / "manifest.json"
    if not path.exists() or not manifest_path.exists():
        raise ValueError(f"model artifact/manifest missing for v{version}")
    if _sha(path) != rec["sha256"]:
        raise ValueError(f"UNTRUSTED artifact for v{version}: sha256 mismatch, refusing to load")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sha256") != rec["sha256"]:
        raise ValueError(f"UNTRUSTED artifact for v{version}: manifest/registry mismatch")
    with open(path, "rb") as f:
        return pickle.load(f), rec
