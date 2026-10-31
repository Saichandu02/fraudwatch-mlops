# Progress

- [x] Deterministic synthetic generator (delayed labels, drift mode) —
      redesigned once after verification showed v1 was unlearnable
- [x] Versioned schema + shared preprocessing pipeline, parity test
- [x] Baseline / XGBoost candidate / weak candidate trained (n=20,000)
- [x] Registry: registration, hash-verified loads, gates, promotion, rollback
- [x] Failed promotion gate demonstrated (v3, 5 reasons, validation metrics)
- [x] Candidate promoted (v2 champion); MLflow runs logged (SQLite backend)
- [x] FastAPI serving + live HTTP demo incl. rollback version proof 2→1→2
- [x] Drift simulation (PSI) with no-overclaim interpretation
- [x] Tests: 7 passed
- [x] Docs: README, architecture, demo, decisions, model card, verification
- [ ] Public deployment (needs an authorized account) — pending by design
- [ ] SHAP explanations (optional dependency not installed) — permutation used
- [ ] Docker execution (no Docker in build env) — compose delivered as config
