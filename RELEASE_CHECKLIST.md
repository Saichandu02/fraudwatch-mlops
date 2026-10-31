# Release checklist

- [x] Tests pass locally (7 passed — docs/verification.md)
- [x] Training reproduced from a clean registry (scripts/run_training.py)
- [x] Evaluation + drift + HTTP demo artifacts regenerated and committed in eval/
- [x] No secrets, model binaries, or data CSV in git (.gitignore verified)
- [x] README metrics match eval/evaluation_report.json
- [ ] CI green on GitHub (not yet pushed/run — do not claim)
- [ ] Public demo deployed + smoke-tested over HTTPS (pending account)
