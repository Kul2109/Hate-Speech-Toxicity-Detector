import os
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# HATE SPEECH TOXICITY DETECTOR - TWITTER RoBERTa V3 MODEL SERVICE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "twitter_roberta_v3"
)


# ============================================================
# TWITTER LABELS
# ============================================================

LABELS = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}


# ============================================================
# CLASSIFICATION IDS
# ============================================================

CLASSIFICATION_IDS = {
    0: "01",
    1: "02",
    2: "03"
}


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MAX_LENGTH = 64


# ============================================================
# SAFE FLOAT HELPER
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return float(default)

        result = float(value)

        if not torch.isfinite(
            torch.tensor(result)
        ):

            return float(default)

        return result

    except Exception:

        return float(default)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Remove Twitter/X mentions
    # --------------------------------------------------------

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# EXPLICIT THREAT DETECTION
# ============================================================

def contains_explicit_threat(text):

    threat_patterns = [

        r"\bi\s+will\s+hurt\s+you\b",

        r"\bi(?:'ll| will)\s+hurt\s+you\b",

        r"\bi\s+am\s+going\s+to\s+hurt\s+you\b",

        r"\bi(?:'m| am)\s+going\s+to\s+hurt\s+you\b",

        r"\bi\s+will\s+kill\s+you\b",

        r"\bi(?:'ll| will)\s+kill\s+you\b",

        r"\bi\s+am\s+going\s+to\s+kill\s+you\b",

        r"\bi(?:'m| am)\s+going\s+to\s+kill\s+you\b",

        r"\byou\s+will\s+be\s+hurt\b",

        r"\byou\s+are\s+going\s+to\s+die\b",

        r"\byou're\s+going\s+to\s+die\b"

    ]

    text_lower = text.lower()

    for pattern in threat_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return True

    return False


# ============================================================
# IDENTITY / GROUP TARGETING DETECTION
# ============================================================

def contains_identity_targeting(text):

    identity_words = [

        "race",
        "racial",
        "racist",
        "racism",
        "ethnicity",
        "ethnic",
        "religion",
        "religious",
        "nationality",
        "national",
        "immigrant",
        "immigrants",
        "gender",
        "sexuality",
        "disability",
        "disabled"

    ]

    hate_verbs = [

        "hate",
        "hates",
        "hated",
        "despise",
        "despises",
        "despised",
        "attack",
        "attacking",
        "target",
        "targeting"

    ]

    text_lower = text.lower()

    has_identity_word = any(

        re.search(
            r"\b" + re.escape(word) + r"\b",
            text_lower
        )

        for word in identity_words

    )

    has_hate_word = any(

        re.search(
            r"\b" + re.escape(word) + r"\b",
            text_lower
        )

        for word in hate_verbs

    )

    return (
        has_identity_word
        and
        has_hate_word
    )


# ============================================================
# PERSONAL INSULT DETECTION
# ============================================================

def contains_personal_insult(text):

    insult_patterns = [

        # ----------------------------------------------------
        # You are ...
        # ----------------------------------------------------

        r"\byou\s+are\s+stupid\b",

        r"\byou\s+are\s+an?\s+idiot\b",

        r"\byou\s+are\s+an?\s+moron\b",

        r"\byou\s+are\s+an?\s+asshole\b",

        r"\byou\s+are\s+an?\s+jerk\b",

        r"\byou\s+are\s+pathetic\b",

        r"\byou\s+are\s+terrible\b",

        r"\byou\s+are\s+disgusting\b",

        r"\byou\s+are\s+useless\b",

        r"\byou\s+are\s+awful\b",

        # ----------------------------------------------------
        # You're ...
        # ----------------------------------------------------

        r"\byou're\s+stupid\b",

        r"\byou're\s+an?\s+idiot\b",

        r"\byou're\s+an?\s+moron\b",

        r"\byou're\s+an?\s+asshole\b",

        r"\byou're\s+an?\s+jerk\b",

        r"\byou're\s+pathetic\b",

        r"\byou're\s+terrible\b",

        r"\byou're\s+disgusting\b",

        r"\byou're\s+useless\b",

        r"\byou're\s+awful\b",

        # ----------------------------------------------------
        # Direct insult words
        # ----------------------------------------------------

        r"\byou\s+(?:are|you're|'re)\s+stupid\b",

        r"\byou\s+(?:are|you're|'re)\s+an?\s+idiot\b",

        r"\byou\s+(?:are|you're|'re)\s+an?\s+moron\b"

    ]

    text_lower = text.lower()

    for pattern in insult_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return True

    return False


# ============================================================
# DIRECT PERSONAL HATE DETECTION
# ============================================================

def contains_direct_hate(text):

    hate_patterns = [

        r"\bi\s+hate\s+you\b",

        r"\bi\s+hate\s+him\b",

        r"\bi\s+hate\s+her\b",

        r"\bi\s+hate\s+them\b",

        r"\bi\s+despise\s+you\b",

        r"\bi\s+despise\s+him\b",

        r"\bi\s+despise\s+her\b",

        r"\bi\s+despise\s+them\b"

    ]

    text_lower = text.lower()

    for pattern in hate_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return True

    return False


# ============================================================
# GENERAL OFFENSIVE EXPRESSIONS
# ============================================================

def contains_common_offensive(text):

    offensive_patterns = [

        r"\bdisgusting\b",

        r"\bshut\s+up\b",

        r"\bidiot\b",

        r"\bidiots\b",

        r"\bmoron\b",

        r"\bmorons\b",

        r"\bstupid\b",

        r"\bpathetic\b",

        r"\bjerk\b",

        r"\bawful\b",

        r"\buseless\b",

        r"\bterrible\b",

        r"\basshole\b"

    ]

    text_lower = text.lower()

    for pattern in offensive_patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return True

    return False


# ============================================================
# TWITTER PREDICTOR
# ============================================================

class TwitterPredictor:

    def __init__(self):

        self.model = None

        self.tokenizer = None

        self.device = torch.device(
            "cpu"
        )

        self.available = False

        self.load_model()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        try:

            if not os.path.exists(
                MODEL_PATH
            ):

                print(
                    f"[Twitter] MODEL NOT FOUND: {MODEL_PATH}"
                )

                print(
                    "[Twitter] Please make sure twitter_roberta_v3 "
                    "has been trained."
                )

                return


            print("=" * 70)

            print(
                "Hate Speech Toxicity Detector - TWITTER RoBERTa V3"
            )

            print("=" * 70)


            # ------------------------------------------------
            # TOKENIZER
            # ------------------------------------------------

            print(
                "[Twitter] Loading V3 tokenizer..."
            )

            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    MODEL_PATH
                )
            )


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            print(
                "[Twitter] Loading V3 RoBERTa model..."
            )

            self.model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    MODEL_PATH
                )
            )


            # ------------------------------------------------
            # CPU
            # ------------------------------------------------

            self.model.to(
                self.device
            )

            self.model.eval()

            self.available = True


            # ------------------------------------------------
            # INFORMATION
            # ------------------------------------------------

            print(
                "[Twitter] RoBERTa V3 model loaded successfully."
            )

            print(
                "[Twitter] Model path:",
                MODEL_PATH
            )

            print(
                "[Twitter] Device:",
                self.device
            )

            print(
                "[Twitter] Max length:",
                MAX_LENGTH
            )

            print(
                "[Twitter] Labels:",
                LABELS
            )

            print("=" * 70)


        except Exception as e:

            self.available = False

            print(
                "[Twitter] MODEL LOADING ERROR:",
                str(e)
            )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    def _model_predict(self, text):

        inputs = self.tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            max_length=MAX_LENGTH,

            padding=True

        )


        inputs = {

            key: value.to(
                self.device
            )

            for key, value in inputs.items()

        }


        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]


        return probabilities


    # ========================================================
    # CLASSIFICATION RULES
    # ========================================================

    def _apply_classification_rules(
        self,
        text,
        probabilities,
        predicted_class
    ):

        # ----------------------------------------------------
        # ALWAYS obtain actual model probabilities
        # ----------------------------------------------------

        hate_score = safe_float(
            probabilities[0].item()
        )

        offensive_score = safe_float(
            probabilities[1].item()
        )

        neither_score = safe_float(
            probabilities[2].item()
        )


        normalized = normalize_text(
            text
        )


        final_class = int(
            predicted_class
        )

        classification_method = (
            "Fine-tuned RoBERTa"
        )

        refinement_applied = False


        # ====================================================
        # RULE 1 - IDENTITY TARGETING
        # ====================================================

        if contains_identity_targeting(
            normalized
        ):

            final_class = 0

            classification_method = (
                "Identity-targeting linguistic rule"
            )

            refinement_applied = (
                predicted_class != 0
            )

            print(
                "[Twitter] Identity-targeting pattern "
                "detected -> Hate Speech"
            )


        # ====================================================
        # RULE 2 - EXPLICIT THREAT
        # ====================================================

        elif contains_explicit_threat(
            normalized
        ):

            # ------------------------------------------------
            # The Twitter dataset has only:
            #
            # 0 = Hate Speech
            # 1 = Offensive Language
            # 2 = Neither
            #
            # Therefore threats are placed under the
            # Offensive Language category.
            # ------------------------------------------------

            final_class = 1

            classification_method = (
                "Threat linguistic rule"
            )

            refinement_applied = (
                predicted_class != 1
            )

            print(
                "[Twitter] Explicit threat pattern "
                "detected -> Offensive Language"
            )


        # ====================================================
        # RULE 3 - PERSONAL INSULT
        # ====================================================

        elif contains_personal_insult(
            normalized
        ):

            final_class = 1

            classification_method = (
                "Personal-insult linguistic rule"
            )

            refinement_applied = (
                predicted_class != 1
            )

            print(
                "[Twitter] Personal insult pattern "
                "detected -> Offensive Language"
            )


        # ====================================================
        # RULE 4 - DIRECT PERSONAL HATE
        # ====================================================

        elif contains_direct_hate(
            normalized
        ):

            final_class = 1

            classification_method = (
                "Direct-hate linguistic rule"
            )

            refinement_applied = (
                predicted_class != 1
            )

            print(
                "[Twitter] Direct personal hate expression "
                "detected -> Offensive Language"
            )


        # ====================================================
        # RULE 5 - COMMON OFFENSIVE EXPRESSION
        # ====================================================

        elif contains_common_offensive(
            normalized
        ):

            final_class = 1

            classification_method = (
                "Offensive-language linguistic rule"
            )

            refinement_applied = (
                predicted_class != 1
            )

            print(
                "[Twitter] Common offensive expression "
                "detected -> Offensive Language"
            )


        return {

            "predicted_class":
                int(predicted_class),

            "final_class":
                int(final_class),

            "hate_score":
                hate_score,

            "offensive_score":
                offensive_score,

            "neither_score":
                neither_score,

            "classification_method":
                classification_method,

            "refinement_applied":
                refinement_applied

        }


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, text):

        if not self.available:

            return {

                "success": False,

                "error":
                    "Twitter RoBERTa V3 model is not available."

            }


        if (
            text is None
            or not str(text).strip()
        ):

            return {

                "success": False,

                "error":
                    "Please enter some text."

            }


        try:

            # ------------------------------------------------
            # ORIGINAL TEXT
            # ------------------------------------------------

            original_text = str(
                text
            ).strip()


            # ------------------------------------------------
            # NORMALIZED TEXT
            # ------------------------------------------------

            normalized_text = normalize_text(
                original_text
            )


            if not normalized_text:

                return {

                    "success": False,

                    "error":
                        "Please enter some valid text."

                }


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            probabilities = self._model_predict(
                normalized_text
            )


            # ------------------------------------------------
            # SAFETY CHECK
            # ------------------------------------------------

            if probabilities is None:

                return {

                    "success": False,

                    "error":
                        "Twitter model returned no prediction."

                }


            if len(probabilities) < 3:

                return {

                    "success": False,

                    "error":
                        "Twitter model returned an invalid "
                        "number of classes."

                }


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            model_predicted_class = int(
                torch.argmax(
                    probabilities
                ).item()
            )


            if model_predicted_class not in LABELS:

                model_predicted_class = 2


            # ------------------------------------------------
            # APPLY RULES
            # ------------------------------------------------

            rule_result = (
                self._apply_classification_rules(
                    normalized_text,
                    probabilities,
                    model_predicted_class
                )
            )


            predicted_class = int(
                rule_result.get(
                    "predicted_class",
                    model_predicted_class
                )
            )

            final_class = int(
                rule_result.get(
                    "final_class",
                    predicted_class
                )
            )


            # ------------------------------------------------
            # SAFETY
            # ------------------------------------------------

            if predicted_class not in LABELS:

                predicted_class = 2


            if final_class not in LABELS:

                final_class = predicted_class


            # ------------------------------------------------
            # SCORES
            # ------------------------------------------------

            hate_score = safe_float(
                rule_result.get(
                    "hate_score",
                    0.0
                )
            )

            offensive_score = safe_float(
                rule_result.get(
                    "offensive_score",
                    0.0
                )
            )

            neither_score = safe_float(
                rule_result.get(
                    "neither_score",
                    0.0
                )
            )


            # ------------------------------------------------
            # NORMALIZE SCORES
            #
            # Softmax should already total 1.0, but this makes
            # the result robust against unexpected values.
            # ------------------------------------------------

            total_score = (
                hate_score
                +
                offensive_score
                +
                neither_score
            )


            if total_score > 0:

                hate_score = (
                    hate_score /
                    total_score
                )

                offensive_score = (
                    offensive_score /
                    total_score
                )

                neither_score = (
                    neither_score /
                    total_score
                )

            else:

                hate_score = 0.0

                offensive_score = 0.0

                neither_score = 1.0


            # ------------------------------------------------
            # LABELS
            # ------------------------------------------------

            model_label = LABELS.get(
                predicted_class,
                "Neither"
            )

            final_label = LABELS.get(
                final_class,
                "Neither"
            )


            # ------------------------------------------------
            # CLASSIFICATION ID
            # ------------------------------------------------

            model_classification_id = (
                CLASSIFICATION_IDS.get(
                    predicted_class,
                    "03"
                )
            )

            classification_id = (
                CLASSIFICATION_IDS.get(
                    final_class,
                    "03"
                )
            )


            # ------------------------------------------------
            # CONFIDENCE
            #
            # IMPORTANT:
            #
            # If the final result is the model prediction,
            # confidence = actual model probability.
            #
            # If a linguistic rule changes the classification,
            # confidence is calculated from the rule decision
            # without pretending that the model gave a high
            # probability to that category.
            #
            # For a refined classification we use:
            #
            #   max(final model probability, 0.50)
            #
            # This provides a clear positive classification
            # while retaining the original model scores below.
            # ------------------------------------------------

            final_probability = safe_float(
                probabilities[
                    final_class
                ].item()
            )


            model_probability = safe_float(
                probabilities[
                    predicted_class
                ].item()
            )


            refinement_applied = bool(
                rule_result.get(
                    "refinement_applied",
                    False
                )
            )


            if not refinement_applied:

                confidence = model_probability

            else:

                confidence = max(
                    final_probability,
                    0.50
                )


            confidence = min(
                max(
                    safe_float(
                        confidence,
                        0.0
                    ),
                    0.0
                ),
                1.0
            )


            # ------------------------------------------------
            # TOXICITY
            # ------------------------------------------------

            if final_class in [0, 1]:

                toxicity = "Toxic"

            else:

                toxicity = "Non-Toxic"


            # ------------------------------------------------
            # FINAL METHOD
            # ------------------------------------------------

            classification_method = (
                rule_result.get(
                    "classification_method",
                    "Fine-tuned RoBERTa"
                )
            )

            if not classification_method:

                classification_method = (
                    "Fine-tuned RoBERTa"
                )


            # ------------------------------------------------
            # RETURN RESULT
            # ------------------------------------------------

            return {

                "success": True,

                # ==================================================
                # TEXT
                # ==================================================

                "text":
                    original_text,

                "cleaned_text":
                    normalized_text,


                # ==================================================
                # MODEL CLASSIFICATION
                # ==================================================

                "model_prediction":
                    model_label,

                "model_predicted_class":
                    predicted_class,

                "model_classification_id":
                    model_classification_id,


                # ==================================================
                # FINAL CLASSIFICATION
                # ==================================================

                "final_classification":
                    final_label,

                "final_class":
                    final_class,

                "classification_id":
                    classification_id,

                "class_id":
                    final_class,

                "label":
                    final_label,

                "prediction":
                    final_label,

                "category":
                    final_label,


                # ==================================================
                # TOXICITY
                # ==================================================

                "toxicity":
                    toxicity,

                "toxicity_status":
                    toxicity,


                # ==================================================
                # CONFIDENCE
                # ==================================================

                "confidence":
                    round(
                        confidence,
                        6
                    ),

                "confidence_percentage":
                    round(
                        confidence * 100,
                        2
                    ),

                "twitter_confidence":
                    round(
                        confidence,
                        6
                    ),


                # ==================================================
                # MODEL CONFIDENCE
                # ==================================================

                "model_confidence":
                    round(
                        model_probability,
                        6
                    ),

                "model_confidence_percentage":
                    round(
                        model_probability * 100,
                        2
                    ),


                # ==================================================
                # RAW MODEL SCORES
                # ==================================================

                "hate_score":
                    round(
                        safe_float(
                            hate_score
                        ),
                        6
                    ),

                "offensive_score":
                    round(
                        safe_float(
                            offensive_score
                        ),
                        6
                    ),

                "neither_score":
                    round(
                        safe_float(
                            neither_score
                        ),
                        6
                    ),


                # ==================================================
                # PERCENTAGE SCORES
                # ==================================================

                "hate_percentage":
                    round(
                        safe_float(
                            hate_score
                        ) * 100,
                        2
                    ),

                "offensive_percentage":
                    round(
                        safe_float(
                            offensive_score
                        ) * 100,
                        2
                    ),

                "neither_percentage":
                    round(
                        safe_float(
                            neither_score
                        ) * 100,
                        2
                    ),


                # ==================================================
                # COMPLETE PROBABILITIES
                # ==================================================

                "probabilities": {

                    "Hate Speech":
                        round(
                            safe_float(
                                hate_score
                            ),
                            6
                        ),

                    "Offensive Language":
                        round(
                            safe_float(
                                offensive_score
                            ),
                            6
                        ),

                    "Neither":
                        round(
                            safe_float(
                                neither_score
                            ),
                            6
                        )

                },


                # ==================================================
                # RULE INFORMATION
                # ==================================================

                "refinement_applied":
                    refinement_applied,

                "classification_refined":
                    refinement_applied,

                "classification_method":
                    classification_method,

                "method":
                    classification_method,


                # ==================================================
                # MODEL INFORMATION
                # ==================================================

                "model":
                    "Fine-tuned RoBERTa V3",

                "model_name":
                    "Fine-tuned RoBERTa V3",

                "source":
                    "Twitter / X"

            }


        except Exception as e:

            print(
                "[Twitter] Prediction error:",
                str(e)
            )

            return {

                "success": False,

                "error":
                    str(e)

            }


# ============================================================
# SINGLETON
# ============================================================

twitter_predictor = TwitterPredictor()


# ============================================================
# SIMPLE FUNCTION
# ============================================================

def predict_twitter_text(text):

    return twitter_predictor.predict(
        text
    )