import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL = "google/flan-t5-base"
ADAPTER_PATH = str(BASE_DIR.parent / "flan_t5_lora_multitask" / "best_lora_weights")

PORT = 5000
DEBUG = True