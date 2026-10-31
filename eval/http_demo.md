# HTTP demo (live, against http://localhost:8023)

## GET /api/model (initial)
HTTP 200
```json
{
  "model_version": 2,
  "kind": "xgboost",
  "threshold": 0.37000000000000005,
  "registry_metrics_validation": {
    "pr_auc": 0.5519,
    "precision": 0.6235,
    "recall": 0.5288,
    "fpr": 0.016,
    "threshold": 0.37,
    "confusion": {
      "tn": 3748,
      "fp": 61,
      "fn": 90,
      "tp": 101
    },
    "brier": 0.0294,
    "prevalence": 0.0478,
    "n": 4000,
    "calibration_bins": [
      {
        "bin": "0.0-0.1",
        "n": 3615,
        "mean_predicted": 0.0142,
        "observed_rate": 0.0152
      },
      {
        "bin": "0.1-0.2",
        "n": 147,
        "mean_predicted": 0.1444,
        "observed_rate": 0.1224
      },
      {
        "bin": "0.2-0.3",
        "n": 49,
        "mean_predicted": 0.2408,
        "observed_rate": 0.2449
      },
      {
        "bin": "0.3-0.4",
        "n": 38,
        "mean_predicted": 0.3492,
        "observed_rate": 0.2632
      },
      {
        "bin": "0.4-0.5",
        "n": 25,
        "mean_predicted": 0.443,
        "observed_rate": 0.36
      },
      {
        "bin": "0.5-0.6",
        "n": 26,
        "mean_predicted": 0.5444,
        "observed_rate": 0.5
      },
      {
        "bin": "0.6-0.7",
        "n": 23,
        "mean_predicted": 0.6515,
        "observed_rate": 0.7391
      },
      {
        "bin": "0.7-0.8",
        "n": 33,
        "mean_predicted": 0.748,
        "observed_rate": 0.6364
      },
      {
        "bin": "0.8-0.9",
        "n": 28,
        "mean_predicted": 0.8569,
        "observed_rate": 0.7857
      },
      {
        "bin": "0.9-1.0",
        "n": 16,
        "mean_predicted": 0.9384,
        "observed_rate": 0.875
      }
    ]
  },
  "data_fingerprint": "cfd35d1b982bfded",
  "schema_version": "1.0",
  "is_champion": true,
  "champion_in_registry": 2,
  "top_features_global": [
    {
      "feature": "device_type",
      "permutation_importance_pr_auc": 0.2825
    },
    {
      "feature": "country_match",
      "permutation_importance_pr_auc": 0.2684
    },
    {
      "feature": "merchant_category",
      "permutation_importance_pr_auc": 0.1928
    }
  ]
}
```

## POST /api/predict (high-risk synthetic tx)
HTTP 200
```json
{
  "request_id": "dd16619a-fb05-4dc1-bf07-9e2ab3cb751e",
  "score": 0.985,
  "threshold": 0.37000000000000005,
  "decision": "fraud",
  "model_version": 2,
  "explanation": {
    "top_features": [
      {
        "feature": "device_type",
        "permutation_importance_pr_auc": 0.2825
      },
      {
        "feature": "country_match",
        "permutation_importance_pr_auc": 0.2684
      },
      {
        "feature": "merchant_category",
        "permutation_importance_pr_auc": 0.1928
      }
    ],
    "signals": [
      "country mismatch combined with high 24h velocity",
      "amount well above this customer's median",
      "new device at a higher-risk merchant category"
    ],
    "note": "Global permutation importance + rule signals for this row. Non-causal; SHAP not installed in this environment."
  }
}
```

## POST /admin/rollback
HTTP 200
```json
{
  "rolled_back": true,
  "from_version": 2,
  "champion": 1,
  "loaded_version_after_reload": 1
}
```

## POST /api/predict after rollback
HTTP 200
```json
{
  "request_id": "5f03d02c-f4ab-43e4-837d-4eab027234fe",
  "score": 0.9972,
  "threshold": 0.8300000000000002,
  "decision": "fraud",
  "model_version": 1,
  "explanation": {
    "top_features": [],
    "signals": [
      "country mismatch combined with high 24h velocity",
      "amount well above this customer's median",
      "new device at a higher-risk merchant category"
    ],
    "note": "Global permutation importance + rule signals for this row. Non-causal; SHAP not installed in this environment."
  }
}
```

## POST /admin/rollback (rolls forward to previous)
HTTP 200
```json
{
  "rolled_back": true,
  "from_version": 1,
  "champion": 2,
  "loaded_version_after_reload": 2
}
```

## POST /api/predict after roll-forward
HTTP 200
```json
{
  "request_id": "209135f1-bb86-4fbf-ba86-50a1543fc065",
  "score": 0.985,
  "threshold": 0.37000000000000005,
  "decision": "fraud",
  "model_version": 2,
  "explanation": {
    "top_features": [
      {
        "feature": "device_type",
        "permutation_importance_pr_auc": 0.2825
      },
      {
        "feature": "country_match",
        "permutation_importance_pr_auc": 0.2684
      },
      {
        "feature": "merchant_category",
        "permutation_importance_pr_auc": 0.1928
      }
    ],
    "signals": [
      "country mismatch combined with high 24h velocity",
      "amount well above this customer's median",
      "new device at a higher-risk merchant category"
    ],
    "note": "Global permutation importance + rule signals for this row. Non-causal; SHAP not installed in this environment."
  }
}
```

## GET /metrics
```
fraudwatch_requests_total 3
fraudwatch_errors_total 0
fraudwatch_predict_latency_p95_ms 9.63
fraudwatch_model_version 2
```

**Version proof:** predict reported model_version 2 -> 1 -> 2 across rollback / roll-forward, with the server reloading each time.
