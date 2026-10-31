"""Deterministic synthetic transaction generator.

Design notes (see data/DATA_CARD.md):
- Chronological transactions for repeated customers over 180 days.
- Fraud comes mainly from latent *compromised windows* (account-takeover
  bursts): for a few short windows per customer, the observable traits
  co-occur -- country mismatch, new device, risky merchant, larger amounts,
  high velocity -- and per-transaction fraud probability is much higher.
  The window itself is never a feature; a model must learn the conjunction
  of observable signals. Outside windows, fraud is rare background noise.
  The label is therefore NOT a threshold on any single exposed feature.
- Labels are delayed: `label_available_at` is 1-14 days after the event.
- A controlled drift scenario shifts amount / merchant / channel mix.
"""
import numpy as np
import pandas as pd

MERCHANTS = ["grocery", "fuel", "restaurant", "retail", "travel", "online_market", "crypto_kiosk", "pharmacy"]
CHANNELS = ["card_present", "online", "mobile_wallet", "atm"]
DEVICES = ["card", "phone_known", "phone_new", "browser_known", "browser_new"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _choice(rng, values, p, mask, out):
    n = int(mask.sum())
    if n:
        out[mask] = rng.choice(values, n, p=p)
    return out


def generate(n=20000, seed=42, drift=False, start="2025-01-01", days=180):
    rng = np.random.default_rng(seed)
    n_customers = 200
    cust_risk = rng.beta(2, 8, n_customers)  # latent, never a feature

    ts = pd.Timestamp(start) + pd.to_timedelta(rng.random(n) * days, unit="d")
    order = np.argsort(ts.values)
    ts = ts[order]
    customer = rng.integers(0, n_customers, n)[order]
    ts_days = (ts - pd.Timestamp(start)).days.values.astype(float) + \
        (ts - pd.Timestamp(start)).seconds.values / 86400.0

    # Latent compromised windows per customer: 2-5 windows of 3-7 days
    # (~10% of transaction time), during which traits co-occur (below).
    windows = {}
    for c in range(n_customers):
        k = rng.integers(2, 6)
        windows[c] = [(float(s), float(s + rng.uniform(3.0, 7.0)))
                      for s in rng.uniform(0, days - 7, size=k)]
    in_window = np.array([any(a <= d < b for a, b in windows[c])
                          for c, d in zip(customer, ts_days)])

    merchant_p = np.array([.20, .12, .14, .20, .08, .16, .03, .07])
    channel_p = np.array([.45, .30, .20, .05])
    device_p = np.array([.40, .25, .08, .20, .07])
    if drift:  # controlled drift: more online / crypto / travel, larger amounts
        merchant_p = np.array([.12, .08, .10, .15, .16, .22, .10, .07])
        channel_p = np.array([.25, .50, .20, .05])
        device_p = np.array([.30, .22, .16, .18, .14])
    merchant = rng.choice(MERCHANTS, n, p=merchant_p)
    channel = rng.choice(CHANNELS, n, p=channel_p)
    device = rng.choice(DEVICES, n, p=device_p)
    # Inside compromised windows the observable traits shift together.
    merchant = _choice(rng, MERCHANTS, np.array([.05, .03, .05, .12, .10, .30, .30, .05]), in_window, merchant)
    channel = _choice(rng, CHANNELS, np.array([.05, .70, .20, .05]), in_window, channel)
    device = _choice(rng, DEVICES, np.array([.05, .10, .45, .10, .30]), in_window, device)

    amount = rng.lognormal(mean=3.6, sigma=0.8, size=n)
    amount = amount * np.where(in_window, 2.6, 1.0)
    if drift:
        amount *= 1.6
    mismatch_p = np.where(merchant == "travel", 0.35, 0.06)
    mismatch_p = np.where(in_window, 0.55, mismatch_p)
    country_match = (rng.random(n) > mismatch_p).astype(int)

    df = pd.DataFrame({
        "timestamp": ts, "customer_id": [f"C{c:04d}" for c in customer],
        "amount": np.round(amount, 2), "merchant_category": merchant,
        "channel": channel, "device_type": device, "country_match": country_match,
    })
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Rolling, at-transaction-time features (chronological, prior txs only).
    velocity, since_prev, ratio, age_days = [], [], [], []
    last_ts, hist, first_ts = {}, {}, {}
    for row in df.itertuples():
        cid, t, amt = row.customer_id, row.timestamp, row.amount
        h = hist.setdefault(cid, [])
        prev = last_ts.get(cid)
        since_prev.append(365.0 if prev is None else max((t - prev).total_seconds() / 86400.0, 0.001))
        velocity.append(sum(1 for _, tt in h if (t - tt).total_seconds() <= 86400))
        med = np.median([a for a, _ in h]) if h else amt
        ratio.append(float(amt / med) if med > 0 else 1.0)
        first_ts.setdefault(cid, t)
        age_days.append(max((t - first_ts[cid]).days, 0))
        h.append((amt, t)); last_ts[cid] = t
    df["velocity_24h"] = velocity
    df["days_since_prev_tx"] = np.round(since_prev, 4)
    df["amount_ratio_to_customer_median"] = np.round(ratio, 4)
    df["customer_age_days"] = age_days

    night = ((df["hour"] <= 4) | (df["hour"] >= 23)).astype(float)
    new_device = df["device_type"].isin(["phone_new", "browser_new"]).astype(float)
    risky_merchant = df["merchant_category"].isin(["crypto_kiosk", "online_market"]).astype(float)
    logit = (
        -5.4
        + 3.4 * in_window
        + 1.2 * (1 - df["country_match"]) * (df["velocity_24h"] >= 2)
        + 1.0 * night * (df["amount_ratio_to_customer_median"] > 2.0)
        + 0.9 * risky_merchant * new_device
        + 1.6 * (1 - df["country_match"]) * new_device * risky_merchant
        + 1.2 * (df["velocity_24h"] >= 2) * (df["amount_ratio_to_customer_median"] > 2.0)
        + 0.5 * (1 - df["country_match"])
        + 0.8 * cust_risk[customer]
        + rng.normal(0, 0.5, n)
    )
    df["is_fraud"] = (rng.random(n) < _sigmoid(logit)).astype(int)
    df["label_available_at"] = df["timestamp"] + pd.to_timedelta(rng.integers(1, 15, n), unit="d")
    return df.reset_index(drop=True)


FEATURES = ["amount", "hour", "day_of_week", "is_weekend", "merchant_category",
            "channel", "device_type", "country_match", "velocity_24h",
            "days_since_prev_tx", "amount_ratio_to_customer_median", "customer_age_days"]
NUMERIC = ["amount", "hour", "day_of_week", "is_weekend", "country_match", "velocity_24h",
           "days_since_prev_tx", "amount_ratio_to_customer_median", "customer_age_days"]
CATEGORICAL = ["merchant_category", "channel", "device_type"]
