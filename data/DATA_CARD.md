# Data card — synthetic transactions

- **Source:** generated in-repo by `src/fraudwatch/generator.py` (seed 42).
  No real transactions, customers, or bank data. Fictional customers C0000-C0199.
- **Size:** 20,000 transactions over 180 days from 2025-01-01 (default run).
- **Label:** `is_fraud` sampled from a latent logit driven mainly by latent
  *compromised windows* (account-takeover bursts: 2-5 windows of 3-7 days per
  customer, during which country mismatch, new devices, risky merchants and
  larger amounts co-occur), plus explicit interactions (country mismatch x
  velocity, night x amount-ratio, mismatch x new device x risky merchant),
  a latent per-customer risk (never a feature), and Gaussian noise.
  No single exposed feature determines the label.
- **Delayed labels:** `label_available_at` = event time + 1-14 days. Training
  on recent data must respect label delay; drift/performance monitoring
  treats recent performance as unknown until labels arrive.
- **Imbalance:** fraud prevalence is a few percent (exact figure in
  `eval/evaluation_report.json` for the committed run).
- **Features at transaction time only:** rolling velocity / since-prev /
  amount-vs-customer-median / customer age are computed chronologically
  from prior transactions of the same customer — no future information.
- **Drift scenario:** `generate(drift=True)` shifts amounts (x1.6) and the
  merchant/channel/device mix. Used by `scripts/simulate_drift.py`.
- **License:** synthetic, generated for this project, MIT with the code.
- **Limits:** synthetic data proves engineering behaviour (training,
  registry, serving, drift plumbing). It proves nothing about real-world
  fraud-detection effectiveness.
