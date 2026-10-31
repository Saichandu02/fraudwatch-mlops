#!/usr/bin/env python3
"""Generate data -> time split -> train baseline/candidate/weak ->
register v1/v2/v3 -> initial champion v1 -> FAILED gate for v3 ->
promote v2 -> final TEST evaluation (test period touched once, here).

Writes eval/evaluation_report.json + .md with real numbers.
Usage: .venv/bin/python scripts/run_training.py [--n 20000]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.inspection import permutation_importance

from fraudwatch import config, registry
from fraudwatch.features import fingerprint, schema_fingerprint
from fraudwatch.generator import FEATURES, generate
from fraudwatch.mlflow_export import log_run
from fraudwatch.train import (candidate_kind_name, evaluate, time_split, train_one)

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    df = generate(n=args.n, seed=config.SEED)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_DIR / "transactions.csv", index=False)
    df.head(200).to_csv(config.DATA_DIR / "sample_transactions.csv", index=False)
    fp = fingerprint(df)
    print(f"data: n={len(df)} prevalence={df['is_fraud'].mean():.4f} fingerprint={fp} schema={schema_fingerprint()}")

    train, val, test = time_split(df)
    splits = {"train": {"n": len(train), "fraud_rate": round(float(train['is_fraud'].mean()), 4),
                        "from": str(train['timestamp'].min()), "to": str(train['timestamp'].max())},
              "validation": {"n": len(val), "fraud_rate": round(float(val['is_fraud'].mean()), 4),
                             "from": str(val['timestamp'].min()), "to": str(val['timestamp'].max())},
              "test": {"n": len(test), "fraud_rate": round(float(test['is_fraud'].mean()), 4),
                       "from": str(test['timestamp'].min()), "to": str(test['timestamp'].max())}}
    print("splits:", json.dumps(splits))

    report = {"dataset": {"n": len(df), "fingerprint": fp, "schema_fingerprint": schema_fingerprint(),
                          "seed": config.SEED, "prevalence": round(float(df['is_fraud'].mean()), 4)},
              "splits": splits, "models": {}, "promotion": {}}

    versions = {}
    for kind, label in [("baseline", "baseline"), ("candidate", "candidate"), ("weak_amount_only", "weak_amount_only")]:
        model, threshold, val_metrics = train_one(train, val, kind)
        test_metrics, _ = evaluate(model, test, threshold)
        kind_name = candidate_kind_name() if kind == "candidate" else ("logistic_regression" if kind == "baseline" else "logistic_regression_amount_only_degraded")
        params = {"kind": kind_name, "features": FEATURES, "threshold_rule": "max F1 on validation"}
        # Registry/gates use VALIDATION metrics only -- the test period must
        # not influence promotion decisions. Test metrics live in the report.
        v = registry.register(model, kind_name, threshold, val_metrics, params, fp)
        versions[kind] = v
        mlflow_status = log_run(kind_name, params, test_metrics, fp)
        report["models"][kind] = {"version": v, "kind": kind_name, "threshold": threshold,
                                  "validation_metrics": val_metrics, "test_metrics": test_metrics,
                                  "mlflow": mlflow_status}
        print(f"v{v} {kind_name}: threshold={threshold:.2f} test={json.dumps(test_metrics)}")
        # global permutation importance for the candidate (honest, non-causal)
        if kind == "candidate":
            rng_idx = np.random.default_rng(0).choice(len(val), size=min(800, len(val)), replace=False)
            pi = permutation_importance(model, val[FEATURES].iloc[rng_idx], val['is_fraud'].values[rng_idx],
                                        n_repeats=3, random_state=0, scoring="average_precision")
            imp = sorted(zip(FEATURES, pi.importances_mean), key=lambda x: -x[1])
            imp_list = [{"feature": f, "permutation_importance_pr_auc": round(float(s), 4)} for f, s in imp]
            report["feature_importance_permutation"] = imp_list
            vdir = config.MODELS_DIR / f"v{v}"
            (vdir / "importance.json").write_text(json.dumps(
                {"method": "permutation importance (PR-AUC drop, validation sample 800, 3 repeats). "
                           "Global and non-causal; SHAP not installed in this environment.",
                 "top_features": imp_list}, indent=2))

    # Promotion story (order matters; see docs/demo.md)
    registry.set_initial_champion(versions["baseline"])
    report["promotion"]["initial_champion"] = versions["baseline"]
    weak_attempt = registry.promote(versions["weak_amount_only"])
    report["promotion"]["weak_v3_attempt"] = weak_attempt
    print(f"FAILED-GATE attempt v{versions['weak_amount_only']}: passed={weak_attempt['passed']} reasons={weak_attempt['reasons']}")
    cand_attempt = registry.promote(versions["candidate"])
    report["promotion"]["candidate_v2_attempt"] = cand_attempt
    print(f"candidate v{versions['candidate']} promotion: passed={cand_attempt['passed']} champion_after={cand_attempt['champion_after']}")
    report["promotion"]["final_champion"] = registry.champion()
    report["versions"] = versions

    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Evaluation report (generated by scripts/run_training.py -- real run)", "",
             f"Dataset: n={len(df)}, prevalence={report['dataset']['prevalence']}, fingerprint={fp}", "",
             "| model | version | threshold | PR-AUC (test) | precision | recall | FPR | Brier | confusion (tn/fp/fn/tp) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for kind, m in report["models"].items():
        t = m["test_metrics"]; c = t["confusion"]
        lines.append(f"| {m['kind']} | v{m['version']} | {m['threshold']:.2f} | {t['pr_auc']} | {t['precision']} | {t['recall']} | {t['fpr']} | {t['brier']} | {c['tn']}/{c['fp']}/{c['fn']}/{c['tp']} |")
    lines += ["", f"Promotion: initial champion v{report['promotion']['initial_champion']}; "
              f"weak v{versions['weak_amount_only']} attempt passed={weak_attempt['passed']} ({'; '.join(weak_attempt['reasons'])}); "
              f"candidate v{versions['candidate']} attempt passed={cand_attempt['passed']}; "
              f"final champion v{report['promotion']['final_champion']}.",
              "", "Thresholds chosen on validation only; test period used once for the numbers above."]
    (ROOT / "eval" / "evaluation_report.md").write_text("\n".join(lines) + "\n")
    print("wrote eval/evaluation_report.json/.md")


if __name__ == "__main__":
    main()
