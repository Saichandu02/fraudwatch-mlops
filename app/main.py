"""FraudWatch serving API.

The server loads ONE concrete registered model version (the registry
champion at startup). Changing the champion does NOT silently change this
process: an explicit POST /admin/reload (token required) re-reads the
champion pointer and loads that version, and /api/model always reports
the version this process has actually loaded -- verified by tests.
"""
import json, sys, time, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from fraudwatch import config, registry
from fraudwatch.features import SCHEMA_VERSION, validate_features
from fraudwatch.generator import FEATURES
from fraudwatch.train import explain

app = FastAPI(title="FraudWatch", version="0.1.0")
STATE = {"model": None, "rec": None, "importance": [], "requests": 0, "errors": 0, "latencies": []}


def load_champion():
    champ = registry.champion()
    if champ is None:
        raise RuntimeError("no champion set in registry")
    model, rec = registry.load_model(champ)  # raises on untrusted/missing artifact
    imp = []
    ipath = Path(rec["path"]).parent / "importance.json"
    if ipath.exists():
        imp = json.loads(ipath.read_text()).get("top_features", [])[:3]
    STATE.update(model=model, rec=rec, importance=imp)
    return rec


@app.on_event("startup")
def _startup():
    try:
        load_champion()
    except Exception as e:  # server still starts; /readyz reports not-ready
        STATE.update(model=None, rec=None)
        print(f"startup: no model loaded ({e})")


class PredictRequest(BaseModel):
    amount: float; hour: int; day_of_week: int; is_weekend: int
    merchant_category: str; channel: str; device_type: str; country_match: int
    velocity_24h: int; days_since_prev_tx: float
    amount_ratio_to_customer_median: float; customer_age_days: int


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "fraudwatch"}


@app.get("/readyz")
def readyz():
    if STATE["model"] is None:
        raise HTTPException(503, "model artifact missing / no champion loaded")
    return {"status": "ready", "model_version": STATE["rec"]["version"]}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lat = sorted(STATE["latencies"])
    p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0
    return (f"fraudwatch_requests_total {STATE['requests']}\n"
            f"fraudwatch_errors_total {STATE['errors']}\n"
            f"fraudwatch_predict_latency_p95_ms {p95:.2f}\n"
            f"fraudwatch_model_version {STATE['rec']['version'] if STATE['rec'] else 0}\n")


@app.get("/api/model")
def api_model():
    if STATE["model"] is None:
        raise HTTPException(503, "no model loaded")
    r = STATE["rec"]
    return {"model_version": r["version"], "kind": r["kind"], "threshold": r["threshold"],
            "registry_metrics_validation": r["metrics"], "data_fingerprint": r["fingerprint"],
            "schema_version": SCHEMA_VERSION, "is_champion": registry.champion() == r["version"],
            "champion_in_registry": registry.champion(),
            "top_features_global": STATE["importance"]}


@app.post("/api/predict")
def predict(req: PredictRequest, request: Request):
    if STATE["model"] is None:
        STATE["errors"] += 1
        raise HTTPException(503, "no model loaded")
    payload = req.model_dump()
    errors = validate_features(payload)
    if errors:
        STATE["errors"] += 1
        raise HTTPException(422, {"errors": errors})
    start = time.perf_counter()
    row = pd.DataFrame([{k: payload[k] for k in FEATURES}])
    try:
        score = float(STATE["model"].predict_proba(row)[:, 1][0])
    except Exception as e:
        STATE["errors"] += 1
        raise HTTPException(500, f"scoring failed: {e}")
    STATE["latencies"].append((time.perf_counter() - start) * 1000)
    STATE["requests"] += 1
    threshold = float(STATE["rec"]["threshold"])
    return {"request_id": str(uuid.uuid4()), "score": round(score, 4), "threshold": threshold,
            "decision": "fraud" if score >= threshold else "legit",
            "model_version": STATE["rec"]["version"],
            "explanation": {"top_features": STATE["importance"],
                            "signals": explain(STATE["model"], row),
                            "note": "Global permutation importance + rule signals for this row. "
                                    "Non-causal; SHAP not installed in this environment."}}


@app.post("/admin/reload")
def admin_reload(x_admin_token: str = Header(default="")):
    if x_admin_token != config.ADMIN_TOKEN:
        raise HTTPException(401, "invalid admin token")
    before = STATE["rec"]["version"] if STATE["rec"] else None
    try:
        rec = load_champion()
    except Exception as e:
        raise HTTPException(503, str(e))
    return {"reloaded": True, "previous_loaded_version": before, "loaded_version": rec["version"]}


@app.post("/admin/rollback")
def admin_rollback(x_admin_token: str = Header(default="")):
    if x_admin_token != config.ADMIN_TOKEN:
        raise HTTPException(401, "invalid admin token")
    result = registry.rollback()
    if not result["rolled_back"]:
        raise HTTPException(409, result["reason"])
    rec = load_champion()
    result["loaded_version_after_reload"] = rec["version"]
    return result


@app.get("/", response_class=HTMLResponse)
def home():
    v = STATE["rec"]["version"] if STATE["rec"] else "none"
    return f"""<html><head><title>FraudWatch</title></head><body>
<h1>FraudWatch — explainable fraud scoring (synthetic data, portfolio demo)</h1>
<p>Loaded model version: <b>{v}</b>. Try <a href="/docs">/docs</a>,
<a href="/api/model">/api/model</a>, <a href="/metrics">/metrics</a>.</p>
<p>POST /api/predict with the documented feature schema to score a fictional transaction.
This is not a real banking decision service.</p></body></html>"""
