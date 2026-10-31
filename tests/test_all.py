import hashlib, json, pickle, sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudwatch import registry
from fraudwatch.features import validate_features
from fraudwatch.generator import generate
from fraudwatch.train import evaluate, time_split, train_one

SAMPLE = {"amount": 240.0, "hour": 2, "day_of_week": 2, "is_weekend": 0,
          "merchant_category": "crypto_kiosk", "channel": "online",
          "device_type": "phone_new", "country_match": 0, "velocity_24h": 3,
          "days_since_prev_tx": 0.5, "amount_ratio_to_customer_median": 3.1,
          "customer_age_days": 120}


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    art = tmp_path_factory.mktemp("art")
    db = art / "registry.db"; mdir = art / "models"
    df = generate(n=3000, seed=7)
    tr, va, te = time_split(df)
    model, t, vm = train_one(tr, va, "candidate")
    v = registry.register(model, "test-candidate", t, vm, {}, "fp-test", models_dir=mdir, db_path=db)
    registry.set_initial_champion(v, db_path=db)
    return {"db": db, "mdir": mdir, "version": v, "model": model, "threshold": t, "df": df, "val": va}


def test_generator_deterministic_and_imbalanced():
    a, b = generate(n=500, seed=3), generate(n=500, seed=3)
    assert a.equals(b)
    big = generate(n=8000, seed=42)
    assert 0.005 < big["is_fraud"].mean() < 0.15
    # not a trivial single-feature threshold: best amount-only AUC is far from perfect
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(big["is_fraud"], big["amount"]) < 0.85


def test_validation_rules():
    assert validate_features(SAMPLE) == []
    bad = dict(SAMPLE, merchant_category="unknown_shop")
    assert any("unknown category" in e for e in validate_features(bad))
    bad2 = dict(SAMPLE, amount=-5)
    assert any("out of range" in e for e in validate_features(bad2))
    bad3 = dict(SAMPLE); del bad3["hour"]
    assert any("missing field: hour" in e for e in validate_features(bad3))


def test_train_serve_parity(trained):
    from fraudwatch.generator import FEATURES
    df = trained["df"]
    batch, _ = evaluate(trained["model"], df.iloc[:50], trained["threshold"])
    import pandas as pd
    single = float(trained["model"].predict_proba(pd.DataFrame([df.iloc[0][FEATURES].to_dict()]))[:, 1][0])
    batch_proba = trained["model"].predict_proba(df[FEATURES].iloc[:50])[:, 1]
    assert abs(single - float(batch_proba[0])) < 1e-9


def test_api_flow_and_version_reload(trained, monkeypatch):
    import fraudwatch.config as cfg
    monkeypatch.setattr(cfg, "REGISTRY_DB", trained["db"])
    monkeypatch.setattr(cfg, "MODELS_DIR", trained["mdir"])
    from fastapi.testclient import TestClient
    import importlib, app.main as main
    importlib.reload(main)
    with TestClient(main.app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").json()["model_version"] == trained["version"]
        r = client.post("/api/predict", json=SAMPLE)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_version"] == trained["version"] and body["request_id"]
        assert 0.0 <= body["score"] <= 1.0 and body["decision"] in ("fraud", "legit")
        assert client.post("/api/predict", json=dict(SAMPLE, merchant_category="nope")).status_code == 422
        assert client.get("/api/model").json()["model_version"] == trained["version"]
        assert "fraudwatch_requests_total" in client.get("/metrics").text
        assert client.post("/admin/reload", headers={"x-admin-token": "wrong"}).status_code == 401
        ok = client.post("/admin/reload", headers={"x-admin-token": cfg.ADMIN_TOKEN})
        assert ok.status_code == 200 and ok.json()["loaded_version"] == trained["version"]


def test_missing_artifact_makes_not_ready(tmp_path, monkeypatch):
    import fraudwatch.config as cfg
    empty_db = tmp_path / "empty.db"
    monkeypatch.setattr(cfg, "REGISTRY_DB", empty_db)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "no_models")
    import importlib, app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app, raise_server_exceptions=False) as client:
        assert client.get("/readyz").status_code == 503
        assert client.post("/api/predict", json=SAMPLE).status_code == 503


def test_untrusted_artifact_rejected(trained):
    rec = registry.get(trained["version"], db_path=trained["db"])
    path = Path(rec["path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tamper")
        with pytest.raises(ValueError, match="UNTRUSTED"):
            registry.load_model(trained["version"], db_path=trained["db"])
    finally:
        path.write_bytes(original)
    model, rec2 = registry.load_model(trained["version"], db_path=trained["db"])
    assert rec2["version"] == trained["version"]


def test_failed_gate_and_rollback(trained):
    db = trained["db"]
    # a weak candidate whose metrics cannot pass the gates
    weak_metrics = {"pr_auc": 0.55, "recall": 0.2, "precision": 0.2, "fpr": 0.2,
                    "threshold": 0.5, "confusion": {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
                    "brier": 0.2, "prevalence": 0.05, "n": 100, "calibration_bins": []}
    v_weak = registry.register(trained["model"], "weak-test", 0.5, weak_metrics, {}, "fp", 
                               models_dir=trained["mdir"], db_path=db)
    result = registry.promote(v_weak, db_path=db)
    assert result["passed"] is False and result["reasons"]
    assert registry.champion(db_path=db) == trained["version"]
    # rollback requires a previous champion: promote nothing, set up two champions
    good_metrics = dict(weak_metrics, pr_auc=0.99, recall=0.9, precision=0.9, fpr=0.01)
    v_good = registry.register(trained["model"], "good-test", 0.5, good_metrics, {}, "fp",
                               models_dir=trained["mdir"], db_path=db)
    assert registry.promote(v_good, db_path=db)["passed"] is True
    assert registry.champion(db_path=db) == v_good
    rb = registry.rollback(db_path=db)
    assert rb["rolled_back"] is True and registry.champion(db_path=db) == trained["version"]
