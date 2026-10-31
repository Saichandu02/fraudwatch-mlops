"""Feature-distribution drift: Population Stability Index (PSI) per numeric
feature. Drift is a *distribution* signal -- it requests review/retraining.
It is NOT evidence of accuracy loss: performance needs labels, and labels
arrive 1-14 days late in this system (see generator)."""
import numpy as np

from .generator import NUMERIC


def psi(expected, actual, bins=10):
    e = np.asarray(expected, dtype=float); a = np.asarray(actual, dtype=float)
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    e_counts, _ = np.histogram(e, bins=qs)
    a_counts, _ = np.histogram(a, bins=qs)
    e_pct = np.clip(e_counts / max(len(e), 1), 1e-4, None)
    a_pct = np.clip(a_counts / max(len(a), 1), 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(reference_df, current_df):
    per_feature = {}
    for f in NUMERIC:
        v = psi(reference_df[f], current_df[f])
        per_feature[f] = {"psi": round(v, 4),
                          "level": "significant" if v >= 0.25 else ("moderate" if v >= 0.1 else "stable")}
    cat_shift = {}
    for f in ["merchant_category", "channel", "device_type"]:
        ref = reference_df[f].value_counts(normalize=True)
        cur = current_df[f].value_counts(normalize=True)
        cats = set(ref.index) | set(cur.index)
        tvd = 0.5 * sum(abs(ref.get(c, 0) - cur.get(c, 0)) for c in cats)
        cat_shift[f] = {"total_variation_distance": round(float(tvd), 4)}
    significant = [f for f, r in per_feature.items() if r["level"] == "significant"]
    return {"per_feature_psi": per_feature, "categorical_shift": cat_shift,
            "drift_detected": bool(significant), "significant_features": significant,
            "interpretation": ("Distribution shift detected on: " + ", ".join(significant) +
                               ". This requests review/retraining. It does NOT show accuracy "
                               "has degraded -- labels for the drifted period have not arrived yet.")
            if significant else "No significant distribution shift detected."}
