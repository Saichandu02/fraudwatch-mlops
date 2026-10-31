#!/usr/bin/env python3
"""CLI-only model administration: promote / rollback / status.
Usage:
  .venv/bin/python scripts/admin.py status
  .venv/bin/python scripts/admin.py promote --version 2
  .venv/bin/python scripts/admin.py rollback
After promote/rollback, tell the running server to reload:
  curl -X POST localhost:8000/admin/reload -H "x-admin-token: $FRAUDWATCH_ADMIN_TOKEN"
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudwatch import registry


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote"); p.add_argument("--version", type=int, required=True)
    sub.add_parser("rollback"); sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps({"champion": registry.champion(),
                          "versions": [{"version": v["version"], "kind": v["kind"],
                                        "threshold": v["threshold"]} for v in registry.list_versions()]}, indent=2))
    elif a.cmd == "promote":
        print(json.dumps(registry.promote(a.version), indent=2))
    else:
        print(json.dumps(registry.rollback(), indent=2))


if __name__ == "__main__":
    main()
