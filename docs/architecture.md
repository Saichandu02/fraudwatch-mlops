# Architecture

```
generator -> (data/transactions.csv) -> train (time split 60/20/20)
  -> registry (SQLite metadata + artifacts/models/v{n}/model.pkl + manifest.json sha256)
       -> promotion gates (validation metrics) -> champion pointer
  -> serving (FastAPI) loads champion's concrete version at startup
       -> /admin/reload re-reads champion; /admin/rollback swaps champion+previous and reloads
  -> drift (PSI vs reference) -> eval/drift_report.*
MLflow (optional, installed in build env): training runs mirrored to a local
SQLite backend, experiment "fraudwatch", tagged with the data fingerprint.
```

Key invariants:
- **One shared preprocessing pipeline** object is pickled inside each model;
  serving never re-implements transformations (parity test enforces it).
- **Registry metrics are validation metrics.** Test metrics exist only in
  `eval/` — promotion can never peek at the test period.
- **Artifact trust:** load = manifest present + SHA-256(file) == registry ==
  manifest. Anything else raises and nothing is unpickled.
- **Loaded version is state, not a lookup:** `/api/model` and every predict
  response report the version this process loaded; a champion change has no
  effect until reload/rollback — by design, and proven in `eval/http_demo.md`.

Public deployment route (not executed — no account authorized): single VM,
Docker Compose `api` service behind an HTTPS reverse proxy, `artifacts` and
`data` on persistent volumes, MLflow/Postgres under the `full` compose
profile bound to localhost only. Admin endpoints stay token-gated; the
MLflow UI is never exposed publicly.
