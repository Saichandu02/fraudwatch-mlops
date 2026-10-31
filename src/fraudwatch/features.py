"""Versioned preprocessing: the SAME pipeline object is used in training
and in serving, so train/serve parity is structural, not a promise."""
import hashlib, json
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .generator import FEATURES, NUMERIC, CATEGORICAL

SCHEMA_VERSION = "1.0"
# Allowed values / ranges enforced at the API boundary and in training.
SCHEMA = {
    "amount": {"type": "float", "min": 0.01, "max": 100000.0},
    "hour": {"type": "int", "min": 0, "max": 23},
    "day_of_week": {"type": "int", "min": 0, "max": 6},
    "is_weekend": {"type": "int", "min": 0, "max": 1},
    "merchant_category": {"type": "cat", "values": ["grocery", "fuel", "restaurant", "retail", "travel", "online_market", "crypto_kiosk", "pharmacy"]},
    "channel": {"type": "cat", "values": ["card_present", "online", "mobile_wallet", "atm"]},
    "device_type": {"type": "cat", "values": ["card", "phone_known", "phone_new", "browser_known", "browser_new"]},
    "country_match": {"type": "int", "min": 0, "max": 1},
    "velocity_24h": {"type": "int", "min": 0, "max": 100},
    "days_since_prev_tx": {"type": "float", "min": 0.0, "max": 3650.0},
    "amount_ratio_to_customer_median": {"type": "float", "min": 0.0, "max": 1000.0},
    "customer_age_days": {"type": "int", "min": 0, "max": 3650},
}


def validate_features(d: dict) -> list[str]:
    errors = []
    for name, rule in SCHEMA.items():
        if name not in d or d[name] is None:
            errors.append(f"missing field: {name}"); continue
        v = d[name]
        if rule["type"] == "cat":
            if v not in rule["values"]:
                errors.append(f"{name}: unknown category {v!r}")
        else:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                errors.append(f"{name}: not numeric"); continue
            if not np.isfinite(fv) or fv < rule["min"] or fv > rule["max"]:
                errors.append(f"{name}: out of range [{rule['min']}, {rule['max']}]")
    extra = set(d) - set(SCHEMA)
    if extra:
        errors.append(f"unexpected fields: {sorted(extra)}")
    return errors


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def fingerprint(df) -> str:
    h = hashlib.sha256()
    h.update(",".join(df.columns).encode())
    h.update(str(len(df)).encode())
    h.update(pd_hash(df))
    return h.hexdigest()[:16]


def pd_hash(df) -> bytes:
    import pandas as pd
    return pd.util.hash_pandas_object(df, index=True).values.tobytes()


def schema_fingerprint() -> str:
    return hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()[:16]
