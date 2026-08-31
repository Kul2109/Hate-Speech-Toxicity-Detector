import pandas as pd

REQUIRED_TEXT = "text"
LABELS = ["toxicity", "insult", "threat", "identity_attack"]

def validate_training_dataframe(df: pd.DataFrame) -> None:
    missing = [c for c in [REQUIRED_TEXT, *LABELS] if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    if df["text"].isna().any():
        raise ValueError("The text column contains missing values.")

    for label in LABELS:
        bad = ~df[label].isin([0, 1])
        if bad.any():
            raise ValueError(
                f"Column '{label}' must contain only 0/1 labels."
            )

def validate_prediction_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    text = text.strip()
    if not text:
        raise ValueError("text cannot be empty.")
    if len(text) > 10000:
        raise ValueError("text is too long.")
    return text
