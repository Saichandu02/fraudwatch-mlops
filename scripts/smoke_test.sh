#!/usr/bin/env bash
# Smoke test against a running server (default :8000). Start it first:
#   .venv/bin/uvicorn app.main:app --port 8000
set -euo pipefail
BASE=${FRAUDWATCH_BASE:-http://localhost:8000}
echo "== healthz"; curl -s "$BASE/healthz"; echo
echo "== readyz"; curl -s "$BASE/readyz"; echo
echo "== model"; curl -s "$BASE/api/model"; echo
echo "== predict (high-risk synthetic transaction)"
curl -s -X POST "$BASE/api/predict" -H 'Content-Type: application/json' -d '{
  "amount": 240.0, "hour": 2, "day_of_week": 2, "is_weekend": 0,
  "merchant_category": "crypto_kiosk", "channel": "online",
  "device_type": "phone_new", "country_match": 0, "velocity_24h": 3,
  "days_since_prev_tx": 0.5, "amount_ratio_to_customer_median": 3.1,
  "customer_age_days": 120}'; echo
echo "== metrics"; curl -s "$BASE/metrics"
