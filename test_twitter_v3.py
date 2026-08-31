import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# HATE SPEECH TOXICITY DETECTOR - TWITTER RoBERTa V3 TEST
# ============================================================

MODEL_PATH = "./models/twitter_roberta_v3"

LABELS = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Hate Speech Toxicity Detector - TWITTER RoBERTa V3 TEST")
print("=" * 70)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Loading RoBERTa V3 model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Model loaded successfully.")
print("Model path:", MODEL_PATH)


# ============================================================
# TEST SENTENCES
# ============================================================

tests = [
    "You are stupid",
    "You are an idiot",
    "I will hurt you",
    "I hate this person because of their race",
    "I hate you",
    "This is disgusting",
    "You are a terrible person",
    "I love everyone",
    "Have a wonderful day"
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = F.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_class = int(
        torch.argmax(probabilities)
    )

    label = LABELS[predicted_class]

    return (
        label,
        probabilities.tolist()
    )


# ============================================================
# RUN TESTS
# ============================================================

print("\n")
print("=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)

for text in tests:

    label, probabilities = predict(text)

    print("\nTEXT:", text)

    print(
        "PREDICTION:",
        label
    )

    print(
        "Hate Speech        :",
        f"{probabilities[0] * 100:.2f}%"
    )

    print(
        "Offensive Language :",
        f"{probabilities[1] * 100:.2f}%"
    )

    print(
        "Neither            :",
        f"{probabilities[2] * 100:.2f}%"
    )

    print(
        "Confidence         :",
        f"{max(probabilities) * 100:.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("V3 TEST COMPLETED")
print("=" * 70)