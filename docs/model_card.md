# Model card — FraudWatch champion (v2, XGBoost)

- **Task:** score a fictional transaction for fraud risk (portfolio demo).
- **Data:** synthetic, generated in-repo, seed 42, n=20,000, prevalence
  4.45%. See `data/DATA_CARD.md`. No real-world validity is claimed.
- **Features (schema v1.0):** amount, hour, day_of_week, is_weekend,
  merchant_category, channel, device_type, country_match, velocity_24h,
  days_since_prev_tx, amount_ratio_to_customer_median, customer_age_days.
  All computable at transaction time; rolling features use prior
  transactions of the same customer only.
- **Split:** chronological 60/20/20 (train 12,000 to 2025-04-19;
  validation 4,000 to 2025-05-24; test 4,000 to 2025-06-29).
- **Threshold:** 0.37, chosen by max F1 on validation.
- **Test performance (period used once):** PR-AUC 0.4571, precision 0.5580,
  recall 0.4556, FPR 0.0159, Brier 0.0286, confusion tn/fp/fn/tp =
  3770/61/92/77 at prevalence 4.23%.
- **Validation performance (gate basis):** PR-AUC 0.5519, precision 0.6235,
  recall 0.5288, FPR 0.0160. Calibration bins in `eval/evaluation_report.json`
  (predicted vs observed track reasonably; the 0.3–0.4 bin over-predicts).
- **Top global features (permutation, PR-AUC drop):** device_type 0.2825,
  country_match 0.2684, merchant_category 0.1928. Non-causal.
- **Intended use:** demonstrating training/registry/serving engineering.
- **Out of scope:** any real credit/fraud decision, any real customer data.
- **Risks/limits:** synthetic patterns are learnable by construction;
  a model promoted on synthetic gates would need complete re-validation,
  fairness review and label-delay handling on real data before any use.
