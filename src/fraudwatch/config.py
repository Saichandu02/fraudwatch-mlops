import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("FRAUDWATCH_ARTIFACTS", ROOT / "artifacts"))
REGISTRY_DB = ARTIFACTS / "registry.db"
MODELS_DIR = ARTIFACTS / "models"
DATA_DIR = Path(os.environ.get("FRAUDWATCH_DATA", ROOT / "data"))
ADMIN_TOKEN = os.environ.get("FRAUDWATCH_ADMIN_TOKEN", "local-dev-token")
SEED = 42
