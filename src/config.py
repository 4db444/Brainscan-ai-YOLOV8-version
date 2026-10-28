from pathlib import Path
from os import environ

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TMP_DIR = PROJECT_ROOT / "tmp"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FINAL_MODEL_PATH = MODELS_DIR / "final_model.h5"
CLASS_NAMES_PATH = PROCESSED_DATA_DIR / "class_names.json"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png"}

IMAGE_SIZE = 224
NUM_CLASSES = 4

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3

TEST_SIZE = 0.2
RANDOM_SEED = int(environ["RANDOM_SEED"])
