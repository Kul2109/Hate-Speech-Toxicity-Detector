# ============================================================
# HATE SPEECH TOXICITY DETECTOR - TWITTER/X MODULE
# Twitter/X classification separated from app.py
# ============================================================

import os
import traceback

from flask import (
    render_template,
    request,
    flash,
    session
)


# ============================================================
# TWITTER/X CONFIGURATION
# ============================================================

TWITTER_HATE_LABELS = [
    "identity_hate",
    "threat",
    "severe_toxic"
]

TWITTER_OFFENSIVE_LABELS = [
    "toxic",
    "obscene",
    "insult"
]

TWITTER_THRESHOLD = float(
    os.getenv(
        "TWITTER_THRESHOLD",
        "0.50"
    )
)


# ============================================================
# TWITTER/X CLASSIFICATION
# ============================================================

def classify_twitter_result(
    text,
    result,
    LABELS
):
    """
    Twitter/X 3-class classification.

    01 = Hate Speech
    02 = Offensive Language
    03 = Neither

    The existing predict_text() result remains
    authoritative for hate-speech detection.
    """

    labels = result.get(
        "labels",
        {}
    ) or {}

    # --------------------------------------------------------
    # NORMALIZE MODEL LABELS
    # --------------------------------------------------------

    cleaned_labels = {}

    for label, score in labels.items():

        try:

            normalized_label = (
                str(label)
                .lower()
                .strip()
            )

            cleaned_labels[
                normalized_label
            ] = float(score)

        except Exception:
            continue

    # --------------------------------------------------------
    # ENSURE ALL MODEL LABELS EXIST
    # --------------------------------------------------------

    for label in LABELS:

        if label not in cleaned_labels:

            cleaned_labels[label] = 0.0

    # --------------------------------------------------------
    # INDIVIDUAL SCORES
    # --------------------------------------------------------

    identity_hate_score = float(
        cleaned_labels.get(
            "identity_hate",
            0.0
        )
    )

    threat_score = float(
        cleaned_labels.get(
            "threat",
            0.0
        )
    )

    severe_toxic_score = float(
        cleaned_labels.get(
            "severe_toxic",
            0.0
        )
    )

    toxic_score = float(
        cleaned_labels.get(
            "toxic",
            0.0
        )
    )

    insult_score = float(
        cleaned_labels.get(
            "insult",
            0.0
        )
    )

    obscene_score = float(
        cleaned_labels.get(
            "obscene",
            0.0
        )
    )

    # --------------------------------------------------------
    # HATE SPEECH SCORE
    # --------------------------------------------------------

    hate_scores = {

        "identity_hate":
            identity_hate_score,

        "threat":
            threat_score,

        "severe_toxic":
            severe_toxic_score
    }

    hate_label = max(
        hate_scores,
        key=hate_scores.get
    )

    hate_score = float(
        hate_scores[hate_label]
    )

    # --------------------------------------------------------
    # OFFENSIVE LANGUAGE SCORE
    # --------------------------------------------------------

    offensive_scores = {

        "toxic":
            toxic_score,

        "insult":
            insult_score,

        "obscene":
            obscene_score
    }

    offensive_label = max(
        offensive_scores,
        key=offensive_scores.get
    )

    offensive_score = float(
        offensive_scores[offensive_label]
    )

    # --------------------------------------------------------
    # THRESHOLDS
    # --------------------------------------------------------

    hate_threshold = float(
        os.getenv(
            "TWITTER_HATE_THRESHOLD",
            "0.30"
        )
    )

    offensive_threshold = float(
        os.getenv(
            "TWITTER_OFFENSIVE_THRESHOLD",
            "0.50"
        )
    )

    # --------------------------------------------------------
    # MAIN predict_text() RESULT
    # --------------------------------------------------------

    base_prediction = str(
        result.get(
            "prediction",
            ""
        )
    ).strip()

    base_hate_speech = bool(
        result.get(
            "hate_speech",
            False
        )
    )

    base_hate_category = result.get(
        "hate_category",
        None
    )

    # --------------------------------------------------------
    # FINAL TWITTER CLASSIFICATION
    # --------------------------------------------------------

    if (
        base_prediction.lower()
        == "hate speech"
        or
        base_hate_speech
    ):

        category = "Hate Speech"

        classification_id = "01"

        # Use hate category returned by
        # predict_text() when available.

        if base_hate_category:

            hate_label = str(
                base_hate_category
            ).lower().strip()

            if hate_label in hate_scores:

                hate_score = float(
                    hate_scores[hate_label]
                )

        base_confidence = float(
            result.get(
                "confidence",
                0.0
            ) or 0.0
        )

        overall_confidence = max(
            hate_score,
            base_confidence
        )

    elif (
        offensive_score
        >= offensive_threshold
    ):

        category = "Offensive Language"

        classification_id = "02"

        overall_confidence = (
            offensive_score
        )

    else:

        category = "Neither"

        classification_id = "03"

        highest_score = max(
            hate_score,
            offensive_score,
            0.0
        )

        overall_confidence = max(
            0.0,
            1.0 - highest_score
        )

    # --------------------------------------------------------
    # NEITHER SCORE
    # --------------------------------------------------------

    neither_score = max(
        0.0,
        1.0 - max(
            hate_score,
            offensive_score
        )
    )

    # --------------------------------------------------------
    # FINAL TWITTER RESULT
    # --------------------------------------------------------

    twitter_result = dict(
        result
    )

    twitter_result.update({

        "analyzed_text":
            text,

        "category":
            category,

        "classification_id":
            classification_id,

        "twitter_confidence":
            round(
                float(
                    overall_confidence
                ),
                4
            ),

        "hate_speech":
            category == "Hate Speech",

        "offensive_language":
            category == "Offensive Language",

        "neither":
            category == "Neither",

        "hate_score":
            round(
                float(hate_score),
                4
            ),

        "offensive_score":
            round(
                float(offensive_score),
                4
            ),

        "neither_score":
            round(
                float(neither_score),
                4
            ),

        "hate_label":
            hate_label,

        "offensive_label":
            offensive_label,

        "identity_hate_score":
            round(
                identity_hate_score,
                4
            ),

        "threat_score":
            round(
                threat_score,
                4
            ),

        "severe_toxic_score":
            round(
                severe_toxic_score,
                4
            ),

        "twitter_category_scores": {

            "Hate Speech":
                round(
                    float(hate_score),
                    4
                ),

            "Offensive Language":
                round(
                    float(offensive_score),
                    4
                ),

            "Neither":
                round(
                    float(neither_score),
                    4
                )
        }
    })

    # --------------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("[TWITTER/X CLASSIFICATION]")
    print("Input:", text)
    print()

    print(
        "Base Prediction:",
        base_prediction
    )

    print(
        "Base Hate Speech:",
        base_hate_speech
    )

    print(
        "Base Hate Category:",
        base_hate_category
    )

    print()

    print(
        "Identity Hate:",
        identity_hate_score
    )

    print(
        "Threat:",
        threat_score
    )

    print(
        "Severe Toxic:",
        severe_toxic_score
    )

    print(
        "Toxic:",
        toxic_score
    )

    print(
        "Insult:",
        insult_score
    )

    print(
        "Obscene:",
        obscene_score
    )

    print()

    print(
        "Hate Score:",
        hate_score
    )

    print(
        "Hate Label:",
        hate_label
    )

    print(
        "Offensive Score:",
        offensive_score
    )

    print(
        "Offensive Label:",
        offensive_label
    )

    print(
        "Hate Threshold:",
        hate_threshold
    )

    print(
        "Offensive Threshold:",
        offensive_threshold
    )

    print()

    print(
        "FINAL CATEGORY:",
        category
    )

    print(
        "CLASSIFICATION ID:",
        classification_id
    )

    print(
        "CONFIDENCE:",
        overall_confidence
    )

    print("=" * 70)
    print()

    return twitter_result


# ============================================================
# REGISTER TWITTER/X ROUTE
# ============================================================

def register_twitter_routes(
    app,
    login_required,
    clean_text,
    predict_text,
    save_prediction_history,
    MODEL_AVAILABLE,
    MODEL_LABELS,
    LABELS
):

    @app.route(
        "/twitter",
        methods=["GET", "POST"]
    )
    @login_required
    def twitter_analysis():

        result = None

        if request.method == "POST":

            text = clean_text(
                request.form.get(
                    "text",
                    ""
                )
            )

            if not text:

                text = clean_text(
                    request.form.get(
                        "tweet_text",
                        ""
                    )
                )

            if not text:

                text = clean_text(
                    request.form.get(
                        "tweet",
                        ""
                    )
                )

            if not text:

                flash(
                    "Please enter some text to analyze.",
                    "warning"
                )

                return render_template(
                    "twitter.html",
                    result=None
                )

            try:

                print()
                print("=" * 70)

                print(
                    "[TWITTER] STARTING TWITTER/X ANALYSIS"
                )

                print(
                    "[TWITTER] Input:",
                    text
                )

                print(
                    "[TWITTER] Model available:",
                    MODEL_AVAILABLE
                )

                print(
                    "[TWITTER] Model labels:",
                    MODEL_LABELS
                )

                print("=" * 70)

                # ------------------------------------------------
                # USE EXISTING WORKING MODEL
                # ------------------------------------------------

                base_result = predict_text(
                    text
                )

                print(
                    "[TWITTER] Base prediction:",
                    base_result.get(
                        "prediction"
                    )
                )

                print(
                    "[TWITTER] Base confidence:",
                    base_result.get(
                        "confidence"
                    )
                )

                print(
                    "[TWITTER] Base hate speech:",
                    base_result.get(
                        "hate_speech"
                    )
                )

                print(
                    "[TWITTER] Base hate category:",
                    base_result.get(
                        "hate_category"
                    )
                )

                print(
                    "[TWITTER] Base labels:",
                    base_result.get(
                        "labels"
                    )
                )

                # ------------------------------------------------
                # MODEL ERROR
                # ------------------------------------------------

                if base_result.get(
                    "prediction"
                ) in [
                    "Error",
                    "Model Unavailable"
                ]:

                    result = {

                        **base_result,

                        "analyzed_text":
                            text,

                        "category":
                            "Unknown",

                        "classification_id":
                            "—",

                        "twitter_confidence":
                            0.0,

                        "hate_speech":
                            False,

                        "offensive_language":
                            False,

                        "neither":
                            False,

                        "hate_score":
                            0.0,

                        "offensive_score":
                            0.0,

                        "neither_score":
                            0.0
                    }

                    flash(
                        "Twitter/X analysis could not be completed.",
                        "danger"
                    )

                    return render_template(
                        "twitter.html",
                        result=result
                    )

                # ------------------------------------------------
                # TWITTER CLASSIFICATION
                # ------------------------------------------------

                result = classify_twitter_result(
                    text,
                    base_result,
                    LABELS
                )

                # ------------------------------------------------
                # SAVE HISTORY
                # ------------------------------------------------

                save_prediction_history(
                    session["user_id"],
                    text,
                    base_result,
                    "twitter"
                )

                # ------------------------------------------------
                # DEBUG
                # ------------------------------------------------

                print()
                print(
                    "[TWITTER] FINAL CATEGORY:",
                    result.get(
                        "category"
                    )
                )

                print(
                    "[TWITTER] CLASSIFICATION ID:",
                    result.get(
                        "classification_id"
                    )
                )

                print(
                    "[TWITTER] TWITTER CONFIDENCE:",
                    result.get(
                        "twitter_confidence"
                    )
                )

                print(
                    "[TWITTER] HATE SCORE:",
                    result.get(
                        "hate_score"
                    )
                )

                print(
                    "[TWITTER] HATE LABEL:",
                    result.get(
                        "hate_label"
                    )
                )

                print(
                    "[TWITTER] OFFENSIVE SCORE:",
                    result.get(
                        "offensive_score"
                    )
                )

                print(
                    "[TWITTER] OFFENSIVE LABEL:",
                    result.get(
                        "offensive_label"
                    )
                )

                print(
                    "[TWITTER] NEITHER SCORE:",
                    result.get(
                        "neither_score"
                    )
                )

                print("=" * 70)
                print()

                flash(
                    "Twitter/X text analyzed successfully.",
                    "success"
                )

            except Exception as e:

                print()
                print(
                    "[TWITTER ERROR]"
                )

                print(
                    str(e)
                )

                traceback.print_exc()

                result = {

                    "prediction":
                        "Error",

                    "confidence":
                        0.0,

                    "twitter_confidence":
                        0.0,

                    "labels":
                        {},

                    "language":
                        "en",

                    "translated_text":
                        text,

                    "analyzed_text":
                        text,

                    "category":
                        "Unknown",

                    "classification_id":
                        "—",

                    "hate_speech":
                        False,

                    "offensive_language":
                        False,

                    "neither":
                        False,

                    "hate_score":
                        0.0,

                    "offensive_score":
                        0.0,

                    "neither_score":
                        0.0,

                    "error":
                        str(e)
                }

                flash(
                    "Twitter/X analysis failed.",
                    "danger"
                )

        return render_template(
            "twitter.html",
            result=result
        )

    return twitter_analysis