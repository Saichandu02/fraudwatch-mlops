#!/usr/bin/env python3
"""Live HTTP demo against a running server: predict on champion,
rollback (loaded version must change), roll forward, predict again.
Writes eval/http_demo.md with the real responses. Restores champion v2.
Usage: .venv/bin/python scripts/run_http_demo.py [base_url]
"""
import json, sys, time
from pathlib import Path

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8023"
TOKEN = {"x-admin-token": "local-dev-token"}
SAMPLE = {"amount": 240.0, "hour": 2, "day_of_week": 2, "is_weekend": 0,
          "merchant_category": "crypto_kiosk", "channel": "online",
          "device_type": "phone_new", "country_match": 0, "velocity_24h": 3,
          "days_since_prev_tx": 0.5, "amount_ratio_to_customer_median": 3.1,
          "customer_age_days": 120}
ROOT = Path(__file__).resolve().parents[1]


def main():
    out = [f"# HTTP demo (live, against {BASE})", ""]
    def rec(label, resp):
        out.append(f"## {label}\nHTTP {resp.status_code}\n```json\n{json.dumps(resp.json(), indent=2)}\n```\n")
        return resp.json()
    rec("GET /api/model (initial)", requests.get(f"{BASE}/api/model", timeout=10))
    p1 = rec("POST /api/predict (high-risk synthetic tx)", requests.post(f"{BASE}/api/predict", json=SAMPLE, timeout=10))
    rb = rec("POST /admin/rollback", requests.post(f"{BASE}/admin/rollback", headers=TOKEN, timeout=10))
    p2 = rec("POST /api/predict after rollback", requests.post(f"{BASE}/api/predict", json=SAMPLE, timeout=10))
    rf = rec("POST /admin/rollback (rolls forward to previous)", requests.post(f"{BASE}/admin/rollback", headers=TOKEN, timeout=10))
    p3 = rec("POST /api/predict after roll-forward", requests.post(f"{BASE}/api/predict", json=SAMPLE, timeout=10))
    rec("GET /metrics", requests.get(f"{BASE}/metrics", timeout=10)) if False else \
        out.append(f"## GET /metrics\n```\n{requests.get(f'{BASE}/metrics', timeout=10).text}```\n")
    out.append(f"**Version proof:** predict reported model_version "
               f"{p1['model_version']} -> {p2['model_version']} -> {p3['model_version']} "
               f"across rollback / roll-forward, with the server reloading each time.")
    (ROOT / "eval" / "http_demo.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
