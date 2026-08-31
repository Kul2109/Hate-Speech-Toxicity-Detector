import torch

from model_loader import model, tokenizer, device


# Labels used during training
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


def predict_toxicity(text):
    """
    Predict toxicity categories for a given text.

    Returns:
        dict: Overall toxicity status and probability
              for each toxicity category.
    """

    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Move tensors to the same device as the model.
    # DistilBERT does not use token_type_ids.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key != "token_type_ids"
    }

    # Run model in evaluation mode
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.sigmoid(
        outputs.logits
    )[0].cpu().numpy()

    # Store category results
    categories = {}

    for label, probability in zip(LABELS, probabilities):

        probability_percentage = float(
            probability * 100
        )

        categories[label] = {
            "probability": round(
                probability_percentage,
                2
            ),
            "detected": bool(
                probability >= 0.5
            )
        }

    # Overall prediction
    overall_toxic = any(
        category["detected"]
        for category in categories.values()
    )

    # Return JSON-friendly Python values
    return {
        "text": str(text),
        "overall_toxic": bool(overall_toxic),
        "categories": categories
    }


# Optional command-line test
if __name__ == "__main__":

    test_text = "You are stupid and useless"

    result = predict_toxicity(test_text)

    print("\nPrediction Result:")
    print(result)