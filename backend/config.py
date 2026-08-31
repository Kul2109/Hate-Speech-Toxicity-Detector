import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "models" / "final_model"))
MODEL_NAME = os.getenv("MODEL_NAME", "roberta-base")
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.50"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

LABELS = [
    "toxicity",
    "insult",
    "threat",
    "identity_attack",
]
