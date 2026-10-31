"""Training: time-based split, baseline vs candidate, validation threshold."""
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score)
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, fingerprint, schema_fingerprint
from .generator import FEATURES

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def time_split(df):
    """First 60% of time = train, next 20% = validation, last 20% = test.
    The test period is touched exactly once, for final reporting."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i, j = int(n * 0.6), int(n * 0.8)
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def make_model(kind):
    if kind == "baseline":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif kind == "candidate" and HAS_XGB:
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.06,
                            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                            random_state=42)
    elif kind == "candidate":
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                             max_depth=None, random_state=42)
    elif kind == "weak_amount_only":
        clf = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(kind)
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def candidate_kind_name():
    return "xgboost" if HAS_XGB else "hist_gradient_boosting (xgboost unavailable -- documented substitution)"


def choose_threshold(y_true, proba):
    """Declared tradeoff: maximise F1 on VALIDATION (fraud misses are costly,
    false alarms are operationally expensive -- F1 balances the two at the
    operating point; the cost framing is documented in the model card)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def metrics_at(y_true, proba, threshold):
    y_true = np.asarray(y_true); pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    # calibration: mean predicted vs observed in 10 bins + Brier score
    bins = []
    for lo in np.arange(0, 1, 0.1):
        m = (proba >= lo) & (proba < lo + 0.1)
        if m.sum() >= 10:
            bins.append({"bin": f"{lo:.1f}-{lo+0.1:.1f}", "n": int(m.sum()),
                         "mean_predicted": round(float(proba[m].mean()), 4),
                         "observed_rate": round(float(y_true[m].mean()), 4)})
    return {
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "threshold": round(float(threshold), 2),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "prevalence": round(float(y_true.mean()), 4),
        "n": int(len(y_true)),
        "calibration_bins": bins,
    }


def train_one(df_train, df_val, kind):
    """Returns (pipeline, threshold, validation_metrics). For the weak model,
    features are restricted to `amount` only (a deliberately degraded
    candidate used to demonstrate a failing promotion gate)."""
    if kind == "weak_amount_only":
        from sklearn.pipeline import Pipeline as P
        model = P([("prep", build_preprocessor()),
                   ("clf", LogisticRegression(max_iter=1000))])
        # train on full pipeline but it will be re-fit below on amount-masked data
        Xtr = df_train[FEATURES].copy(); Xv = df_val[FEATURES].copy()
        for c in FEATURES:
            if c not in ("amount",):
                if Xtr[c].dtype == object:
                    Xtr[c] = Xtr[c].mode()[0]; Xv[c] = Xv[c].mode()[0]
                else:
                    Xtr[c] = 0; Xv[c] = 0
        model.fit(Xtr, df_train["is_fraud"])
        proba = model.predict_proba(Xv)[:, 1]
    else:
        model = make_model(kind)
        model.fit(df_train[FEATURES], df_train["is_fraud"])
        proba = model.predict_proba(df_val[FEATURES])[:, 1]
    t = choose_threshold(df_val["is_fraud"].values, proba)
    return model, t, metrics_at(df_val["is_fraud"].values, proba, t)


def evaluate(model, df, threshold):
    proba = model.predict_proba(df[FEATURES])[:, 1]
    return metrics_at(df["is_fraud"].values, proba, threshold), proba


def explain(model, row_df, top_k=3):
    """Bounded, non-causal explanation: permutation importance computed on a
    fixed reference sample at load time is stored on the model wrapper in
    training; here we use a simpler signed contribution proxy -- the change
    in score when each feature is set to its training median/mode is NOT
    available at serve time, so we report the global top features and the
    row's riskiest raw signals. Labelled honestly in the API as
    'explanation.top_features' (global importance) + 'signals'."""
    signals = []
    r = row_df.iloc[0]
    if int(r["country_match"]) == 0 and int(r["velocity_24h"]) >= 2:
        signals.append("country mismatch combined with high 24h velocity")
    if float(r["amount_ratio_to_customer_median"]) > 2.0:
        signals.append("amount well above this customer's median")
    if r["device_type"] in ("phone_new", "browser_new") and r["merchant_category"] in ("crypto_kiosk", "online_market"):
        signals.append("new device at a higher-risk merchant category")
    if int(r["velocity_24h"]) >= 4:
        signals.append("very high transaction velocity in 24h")
    if int(r["hour"]) <= 4 or int(r["hour"]) >= 23:
        signals.append("late-night transaction hour")
    return signals[:top_k]
