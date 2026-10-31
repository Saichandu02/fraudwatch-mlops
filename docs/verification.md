# Verification — what actually ran (2026-10-31, this VM)

Environment: Python 3.12.3, 4 cores, ~8 GB RAM, **no Docker**.
Installed in `.venv`: scikit-learn 1.9.1, **xgboost 3.4.1** (candidate really
is XGBoost — its first install failed on /tmp space; fixed with TMPDIR on
disk), **mlflow 3.17.2** (runs logged, SQLite backend). SHAP not installed —
permutation importance used and labelled (see decisions.md).

## Generator learnability failure — found, fixed, re-verified
First generator version produced PR-AUC ≈ 0.014–0.017 for all models
(prevalence 1.3%): the interaction segments were ~0.1% of rows, so the
label was effectively noise to the models. The generator was redesigned
(compromised windows + triple interactions, prevalence 4.45%) and every
number below is from the redesigned generator (seed 42). Full story:
`docs/decisions.md` §3.

## Training (scripts/run_training.py --n 20000) — executed
- Data: n=20,000, prevalence 0.0445, fingerprint `cfd35d1b982bfded`,
  schema fingerprint `e8f87fdf122af84d`.
- Splits (chronological): train 12,000 (fraud 4.42%), validation 4,000
  (4.78%), test 4,000 (4.23%).
- Test metrics (test touched once): baseline v1 PR-AUC 0.4204 / P 0.4785 /
  R 0.5917 / FPR 0.0285 / Brier 0.1066, confusion 3722/109/69/100;
  **candidate v2 (XGBoost) PR-AUC 0.4571 / P 0.5580 / R 0.4556 / FPR 0.0159 /
  Brier 0.0286, confusion 3770/61/92/77, threshold 0.37**;
  weak v3 PR-AUC 0.1009. Validation (gate basis): v1 0.5002, v2 0.5519,
  v3 0.1516. MLflow: all three runs logged (experiment `fraudwatch`).

## Promotion gates — executed (validation metrics only)
- Initial champion: **v1** (documented bootstrap deployment).
- **FAILED gate — v3 promote attempt, passed=false**, reasons:
  pr_auc 0.152 < 0.25; recall 0.298 < 0.35; precision 0.193 < 0.2;
  fpr 0.063 > 0.05; pr_auc 0.152 does not beat champion v1 (0.500).
- Candidate v2 promote attempt: **passed=true**, champion after = **2**.

## Serving over HTTP — executed (uvicorn :8023)
`eval/http_demo.md` holds the verbatim responses:
- `GET /readyz` → model_version 2; `GET /api/model` → v2 xgboost, thr 0.37.
- Predict (high-risk sample) → **score 0.985, decision fraud, model_version 2**,
  request_id dd16619a-…, 3 explanation signals.
- `POST /admin/rollback` → champion 1, server reloaded v1; predict →
  **model_version 1**, score 0.9972, threshold 0.83.
- Rollback again (rolls forward) → predict → **model_version 2** again.
  Version sequence **2 → 1 → 2** is the rollback proof.
- Benchmark (`eval/benchmark.json`): 200 sequential predicts after
  20 warm-up, localhost: p50 10.27 ms, p95 13.08 ms, mean 10.71 ms.

## Drift (scripts/simulate_drift.py) — executed
PSI significant: amount **0.307**, velocity_24h **0.2844**,
days_since_prev_tx **0.635**; categorical TVD: merchant 0.186, channel
0.180, device 0.128. Report states the shift requests review and does NOT
claim accuracy loss (labels not yet arrived). `eval/drift_report.md`.

## Tests — executed
`pytest -q` → **7 passed**: generator determinism/imbalance + amount-only
AUC < 0.85 (not trivial), schema validation (unknown category, range,
missing field), train/serve parity (single-row vs batch < 1e-9), API flow
(422 on invalid, 401 on bad admin token, reload), missing-artifact 503s,
tampered-artifact rejection (UNTRUSTED, hash mismatch), failed gate +
rollback in the registry.

## Config-only — NOT executed
Docker build / Docker Compose (no Docker on this VM), Streamlit UI run
(file delivered, calls the same API; the built-in `/` page was served),
public deployment (no authorized account), GitHub Actions CI (workflow
committed; remote run not claimed), SHAP (not installed; optional extra).
