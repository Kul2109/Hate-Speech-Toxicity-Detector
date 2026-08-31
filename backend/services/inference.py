from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.services.explanations import attention_token_weights


# ============================================================
# MODEL CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "final_distilbert_toxicity_model"

MAX_LENGTH = 256
PREDICTION_THRESHOLD = 0.50

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


# ============================================================
# HATE SPEECH / TOXICITY PREDICTOR
# ============================================================

class HateSpeechPredictor:

    def __init__(self):

        print(f"Loading model from: {MODEL_PATH}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at:\n{MODEL_PATH}"
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH)
        )

        # Load fine-tuned DistilBERT model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH),
            output_attentions=True
        )

        self.model.eval()

        # Use GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        self.labels = LABELS

        print("Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Labels: {self.labels}")


    # ========================================================
    # PREDICTION
    # ========================================================

    @torch.no_grad()
    def predict(self, text: str):

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        # Move tensors to device
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # DistilBERT does not use token_type_ids
        inputs.pop("token_type_ids", None)

        # Model prediction
        outputs = self.model(
            **inputs,
            output_attentions=True
        )

        # Convert logits to probabilities
        probabilities = torch.sigmoid(
            outputs.logits
        )[0].detach().cpu().numpy()

        # ====================================================
        # BUILD SIX CATEGORY RESULTS
        # ====================================================

        labels = {}

        for label, score in zip(
            self.labels,
            probabilities
        ):

            score = float(score)

            labels[label] = {
                "flagged": bool(
                    score >= PREDICTION_THRESHOLD
                ),
                "score": round(score, 4)
            }

        # ====================================================
        # OVERALL TOXICITY
        # ====================================================

        overall_toxic = bool(
            probabilities[0] >= PREDICTION_THRESHOLD
        )

        # ====================================================
        # ATTENTION EXPLANATION
        # ====================================================

        try:

            attention = attention_token_weights(
                outputs.attentions,
                inputs["input_ids"],
                self.tokenizer
            )

        except Exception as exc:

            print(
                f"Warning: attention calculation failed: {exc}"
            )

            attention = []

        # ====================================================
        # FINAL API RESPONSE
        # ====================================================

        return {
            "text": text,
            "overall_toxic": overall_toxic,
            "labels": labels,
            "threshold": PREDICTION_THRESHOLD,
            "attention": attention
        }