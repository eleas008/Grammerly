import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = Path(__file__).resolve().parent

BART_NAME = "facebook/bart-large-cnn"
GEC_MODEL_PATH = BASE_DIR / "Flan-t5"
PARAPHRASING_MODEL_PATH = BASE_DIR / "t5-small_paraphrase"

PORT = 5000
DEBUG = True