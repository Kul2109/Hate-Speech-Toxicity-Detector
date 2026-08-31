import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Trained DistilBERT model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "final_distilbert_toxicity_model"
)

print("Loading model from:", MODEL_PATH)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)