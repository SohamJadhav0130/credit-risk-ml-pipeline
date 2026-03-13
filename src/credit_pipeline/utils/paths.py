from pathlib import Path

# This resolves to project root regardless of where you run from
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "logs" / "training" / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
OPTUNA_DB_PATH = LOGS_DIR / "optuna" / "optuna_results.db"
if __name__ == "__main__":
    print(PROJECT_ROOT)
