# Demo walkthrough (60–90 seconds)

Prerequisite: quickstart in README (training run + server on :8023).

1. Open `http://localhost:8023/` — page shows loaded model version **2**.
2. Run `bash scripts/smoke_test.sh` (set `FRAUDWATCH_BASE=http://localhost:8023`):
   health, model card, and a high-risk transaction scored **0.985 → fraud**,
   with the three explanation signals.
3. Run `.venv/bin/python scripts/run_http_demo.py http://localhost:8023`:
   rollback → the same transaction is now scored by **model_version 1**
   (threshold 0.83); roll forward → back to **version 2**. This is the
   promotion/rollback proof; output is saved to `eval/http_demo.md`.
4. Show `eval/evaluation_report.md` (baseline vs candidate vs weak, test
   metrics) and the promotion line: v3's gate **failed** (5 reasons),
   v2 passed.
5. Show `eval/drift_report.md`: PSI significant on amount (0.307),
   velocity_24h (0.284) and days_since_prev_tx (0.635) — and the note that
   this is a distribution signal, not an accuracy claim.
6. Optional: Streamlit UI (`ui/streamlit_app.py`) — move the amount/ratio
   sliders and watch the decision flip.

Recording: pending (no screen-recording tooling in the build environment);
this script plus `eval/http_demo.md` is the verifiable substitute.
