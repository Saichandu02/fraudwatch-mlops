"""Optional MLflow integration.

If MLflow is installed, training runs are logged to a local file store
(./mlruns, gitignored) with params/metrics/data fingerprint, mirroring the
SQLite registry. If MLflow is not installed this module is a no-op and the
registry remains the system of record -- that state is recorded in
docs/verification.md, never hidden.
"""
import os


def log_run(kind, params, metrics, fingerprint):
    try:
        import mlflow
    except Exception:
        return {"mlflow": "not installed -- run logged to SQLite registry only"}
    try:
        # MLflow 3.x refuses the file store by default; use a local SQLite backend.
        default_uri = "sqlite:///" + str(os.path.join(os.getcwd(), "mlflow.db"))
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_uri))
        mlflow.set_experiment("fraudwatch")
        with mlflow.start_run(run_name=f"fraudwatch-{kind}"):
            mlflow.log_params({k: str(v)[:450] for k, v in params.items()})
            flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(flat)
            mlflow.set_tag("data_fingerprint", fingerprint)
        return {"mlflow": "logged (SQLite backend ./mlflow.db, experiment 'fraudwatch')"}
    except Exception as e:  # MLflow is a mirror, never a training blocker
        return {"mlflow": f"logging failed ({type(e).__name__}: {e}) -- SQLite registry remains system of record"}
