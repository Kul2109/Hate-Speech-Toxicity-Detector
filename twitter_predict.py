import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# HATE SPEECH TOXICITY DETECTOR - TWITTER RoBERTa PREDICTION
# ============================================================

MODEL_PATH = "models/twitter_roberta_v2"


# ============================================================
# LABEL MAPPING
# ============================================================

LABELS = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither",
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Hate Speech Toxicity Detector - TWITTER RoBERTa PREDICTION")
print("=" * 70)

print("\nLoading Twitter RoBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully.")
print("Model path:", MODEL_PATH)
print("Device:", device)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_tweet(text):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if text is None or not str(text).strip():

        return {
            "text": text,
            "class_id": None,
            "label": "Invalid",
            "confidence": 0.0,

            "hate_score": 0.0,
            "offensive_score": 0.0,
            "neither_score": 0.0,

            "probabilities": {
                "Hate Speech": 0.0,
                "Offensive Language": 0.0,
                "Neither": 0.0,
            }
        }

    text = str(text).strip()

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=96,
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # --------------------------------------------------------
    # Extract individual class probabilities
    # --------------------------------------------------------

    hate_score = probabilities[0].item()

    offensive_score = probabilities[1].item()

    neither_score = probabilities[2].item()

    # --------------------------------------------------------
    # Find predicted class
    # --------------------------------------------------------

    predicted_class = torch.argmax(
        probabilities
    ).item()

    confidence_value = probabilities[
        predicted_class
    ].item()

    label = LABELS.get(
        predicted_class,
        "Unknown"
    )

    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {

        "text": text,

        "class_id": predicted_class,

        "label": label,

        "confidence": confidence_value,

        # Individual probabilities
        "hate_score": hate_score,

        "offensive_score": offensive_score,

        "neither_score": neither_score,

        # Percentage values
        "hate_percentage": hate_score * 100,

        "offensive_percentage": offensive_score * 100,

        "neither_percentage": neither_score * 100,

        # Complete probability dictionary
        "probabilities": {

            "Hate Speech":
                hate_score,

            "Offensive Language":
                offensive_score,

            "Neither":
                neither_score,
        }
    }


# ============================================================
# TEST MODEL
# ============================================================

def run_test_predictions():

    test_tweets = [

        # ----------------------------------------------------
        # NEITHER
        # ----------------------------------------------------

        "Have a wonderful day everyone!",

        "I really enjoyed watching the match today.",

        "The weather is beautiful today.",

        # ----------------------------------------------------
        # OFFENSIVE LANGUAGE
        # ----------------------------------------------------

        "You are such an idiot.",

        "That was a stupid thing to do.",

        "Stop being so annoying.",

        # ----------------------------------------------------
        # HATE SPEECH
        # ----------------------------------------------------

        "People from that group are inferior.",

        "That group should not be accepted in society.",

        "People of that ethnicity should not be allowed here.",

    ]

    print("\n")
    print("=" * 70)
    print("MODEL PREDICTIONS")
    print("=" * 70)

    for number, tweet in enumerate(
        test_tweets,
        start=1
    ):

        result = predict_tweet(tweet)

        print(f"\nTest {number}")
        print("-" * 70)

        print("Tweet:")
        print(tweet)

        print(
            "\nPredicted class:",
            result["class_id"]
        )

        print(
            "Prediction:",
            result["label"]
        )

        print(
            "Confidence:",
            f"{result['confidence'] * 100:.2f}%"
        )

        print("\nClass probabilities:")

        print(
            "Hate Speech:",
            f"{result['hate_percentage']:.2f}%"
        )

        print(
            "Offensive Language:",
            f"{result['offensive_percentage']:.2f}%"
        )

        print(
            "Neither:",
            f"{result['neither_percentage']:.2f}%"
        )


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode():

    print("\n")
    print("=" * 70)
    print("INTERACTIVE TWITTER ANALYSIS")
    print("=" * 70)

    print("\nEnter a tweet to analyze.")
    print("Type 'exit' to stop.\n")

    while True:

        text = input("Tweet: ").strip()

        if text.lower() == "exit":

            print(
                "\nTwitter prediction test completed."
            )

            break

        if not text:

            print(
                "Please enter some text."
            )

            continue

        result = predict_tweet(text)

        print(
            "\nPrediction :",
            result["label"]
        )

        print(
            "Confidence :",
            f"{result['confidence'] * 100:.2f}%"
        )

        print("\nClass probabilities:")

        print(
            "Hate Speech :",
            f"{result['hate_percentage']:.2f}%"
        )

        print(
            "Offensive Language :",
            f"{result['offensive_percentage']:.2f}%"
        )

        print(
            "Neither :",
            f"{result['neither_percentage']:.2f}%"
        )

        print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_test_predictions()

    interactive_mode()