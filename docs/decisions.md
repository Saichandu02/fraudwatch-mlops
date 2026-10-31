# Decisions

1. **Registry: SQLite + filesystem, MLflow-mirrored.** MLflow was in fact
   installed and used in the build environment (3.17.2, SQLite backend —
   MLflow 3.x refuses its old file store, so `mlflow_export.py` uses
   `sqlite:///mlflow.db` and never lets an MLflow failure block training).
   The SQLite registry remains the system of record for serving because it
   gives exact champion/rollback + artifact-hash semantics with zero
   services; MLflow holds the experiment history. Gates, versions and the
   champion pointer mirror MLflow's registry workflow (versions + alias).
2. **Candidate = XGBoost, honestly obtained.** XGBoost 3.4.1 installed and
   is the shipped candidate (validation PR-AUC 0.5519 vs baseline 0.5002).
   Its wheel initially failed with "No space left" because pip unpacks in
   /tmp (a 512 MB tmpfs here); setting `TMPDIR` to disk fixed it. The
   HistGradientBoosting fallback in `train.py` remains for environments
   where XGBoost cannot install, and `candidate_kind_name()` names which
   one actually trained — the registry records the true kind string.
3. **Generator v2 — learnability bug found and fixed.** The first generator
   (independent rare traits + noise) produced near-random models
   (PR-AUC ≈ 0.02 at 1.3% prevalence) — the "signal" segments were ~0.1% of
   rows. Rather than lowering the bar, the data-generating process was
   redesigned (v2): latent compromised windows where observable traits
   co-occur, explicit triple interactions, ~4.5% prevalence. The final
   numbers are from v2; the v1 failure is part of the verification record.
4. **Threshold = max F1 on validation.** Declared tradeoff: fraud misses
   and false alarms are both operationally costly; F1 at the operating
   point is the simple, reproducible compromise. A cost-weighted threshold
   is a documented extension, not silently mixed in.
5. **Explanations without SHAP.** SHAP was not installed (heavy numba
   stack; optional in `requirements-optional.txt`). Shipped explanations
   are global permutation importance (PR-AUC drop, validation sample) +
   row-level rule signals, labelled non-causal in every response. No
   SHAP values are claimed anywhere.
6. **Reload semantics over alias magic.** A champion change does nothing
   to a running server until `/admin/reload` (or the rollback endpoint,
   which reloads itself). `/api/model.is_champion` can therefore briefly
   be false — that honesty is the point of the demo.
7. **Weak model for the failed gate.** v3 is a real trained model
   (logistic regression with all features except `amount` masked out),
   registered like any candidate. Its gate failure uses its genuine
   validation metrics, not hand-written bad numbers.
