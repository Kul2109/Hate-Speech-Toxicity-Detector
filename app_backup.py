# ============================================================
# HATE SPEECH TOXICITY DETECTOR - COMPLETE FLASK APPLICATION
# Multilingual Hate Speech & Toxicity Detection
#
# Twitter/X classification fixed.
#
# IMPORTANT:
# ------------------------------------------------------------
# Login / Registration
# Dashboard
# Text Analysis
# CSV Analysis
# YouTube Analysis
# History
# Profile
# Database
# Model loading
#
# are preserved.
#
# Twitter/X uses a separate classification layer on top of
# the existing 6-label toxicity model.
# ============================================================

import os
import re
import json
import sqlite3
import traceback
import time
import httplib2

from datetime import datetime
from functools import wraps
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import torch

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    flash
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from dotenv import load_dotenv


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoConfig
    )
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoConfig = None


try:
    from langdetect import detect
except ImportError:
    detect = None


try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None


try:
    from googleapiclient.discovery import build
except ImportError:
    build = None


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# FLASK CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

app = Flask(
    __name__,
    template_folder=os.path.join(
        BASE_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        BASE_DIR,
        "static"
    )
)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "toxiguard-development-secret-key-change-this"
)

app.config["MAX_CONTENT_LENGTH"] = (
    20 * 1024 * 1024
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_DIR = os.path.join(
    BASE_DIR,
    "data"
)

os.makedirs(
    DATABASE_DIR,
    exist_ok=True
)

DATABASE = os.path.join(
    DATABASE_DIR,
    "toxiguard.db"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(
        BASE_DIR,
        "final_distilbert_toxicity_model"
    )
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("Hate Speech Toxicity Detector")
print("=" * 70)
print("Base directory :", BASE_DIR)
print("Database       :", DATABASE)
print("Model path     :", MODEL_PATH)
print("Device         :", DEVICE)
print("=" * 70)


# ============================================================
# MODEL LABELS
# ============================================================

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


LABEL_DISPLAY_NAMES = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate"
}


# ============================================================
# TWITTER/X CLASSIFICATION CONFIGURATION
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
# DATABASE HELPERS
# ============================================================

def get_db():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    return conn


get_db_connection = get_db


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            language TEXT,
            translated_text TEXT,
            prediction TEXT NOT NULL,
            confidence REAL DEFAULT 0,
            labels_json TEXT,
            source TEXT DEFAULT 'text',
            created_at TEXT NOT NULL,

            FOREIGN KEY(user_id)
            REFERENCES users(id)
            ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            analysis_type TEXT NOT NULL,
            source TEXT,
            total_items INTEGER DEFAULT 0,
            toxic_items INTEGER DEFAULT 0,
            non_toxic_items INTEGER DEFAULT 0,
            toxic_percentage REAL DEFAULT 0,
            result_json TEXT,
            created_at TEXT NOT NULL,

            FOREIGN KEY(user_id)
            REFERENCES users(id)
            ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS youtube_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            video_id TEXT NOT NULL,
            comment_id TEXT,
            author TEXT,
            comment_text TEXT,
            prediction TEXT,
            confidence REAL DEFAULT 0,
            labels_json TEXT,
            created_at TEXT NOT NULL,

            FOREIGN KEY(user_id)
            REFERENCES users(id)
            ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()

    print(
        "[DATABASE] Database initialized successfully."
    )


init_database()


# ============================================================
# MODEL LOADING
# ============================================================

tokenizer = None
model = None
MODEL_AVAILABLE = False
MODEL_LABELS = list(LABELS)


def load_model():

    global tokenizer
    global model
    global MODEL_AVAILABLE
    global MODEL_LABELS

    if (
        AutoTokenizer is None
        or AutoModelForSequenceClassification is None
    ):

        print(
            "[MODEL] transformers is not installed."
        )

        return

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "[MODEL] Model directory not found:"
        )

        print(
            MODEL_PATH
        )

        return

    try:

        print(
            "[MODEL] Loading tokenizer..."
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH
        )

        print(
            "[MODEL] Loading model..."
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        model.to(
            DEVICE
        )

        model.eval()

        try:

            config = model.config

            id2label = getattr(
                config,
                "id2label",
                {}
            )

            if id2label:

                detected_labels = []

                for index in range(
                    len(id2label)
                ):

                    label = id2label.get(
                        index,
                        id2label.get(
                            str(index),
                            ""
                        )
                    )

                    if label:

                        label = str(
                            label
                        ).lower().strip()

                        detected_labels.append(
                            label
                        )

                if len(detected_labels) == len(
                    LABELS
                ):

                    MODEL_LABELS = (
                        detected_labels
                    )

                    print(
                        "[MODEL] Labels loaded from config:"
                    )

                    print(
                        MODEL_LABELS
                    )

        except Exception as label_error:

            print(
                "[MODEL] Could not read config labels:",
                label_error
            )

            MODEL_LABELS = list(
                LABELS
            )

        MODEL_AVAILABLE = True

        print(
            "[MODEL] Model loaded successfully."
        )

        print(
            "[MODEL] Device:",
            DEVICE
        )

        print(
            "[MODEL] Final labels:",
            MODEL_LABELS
        )

    except Exception as e:

        MODEL_AVAILABLE = False

        print(
            "[MODEL] ERROR while loading model:"
        )

        print(
            str(e)
        )

        traceback.print_exc()


load_model()


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(view_function):

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please login to access this page.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        return view_function(
            *args,
            **kwargs
        )

    return wrapped_view


# ============================================================
# CURRENT USER
# ============================================================

def get_current_user():

    if "user_id" not in session:
        return None

    conn = get_db()

    user = conn.execute(
        """
        SELECT
            id,
            username,
            email,
            created_at
        FROM users
        WHERE id = ?
        """,
        (
            session["user_id"],
        )
    ).fetchone()

    conn.close()

    return user


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    if not text:
        return "en"

    text = text.strip()

    if len(text.split()) < 4:
        return "en"

    if detect is None:
        return "en"

    try:

        return detect(text)

    except Exception:

        return "en"


# ============================================================
# TRANSLATION
# ============================================================

def translate_to_english(
    text,
    language
):

    if not text:
        return ""

    if language == "en":
        return text

    if GoogleTranslator is None:
        return text

    try:

        translated = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(
            text
        )

        if translated:
            return translated

    except Exception as e:

        print(
            "[TRANSLATION] Failed:",
            str(e)
        )

    return text


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# MODEL PREDICTION
# ============================================================


def predict_text(text):

    text = clean_text(text)

    if not text:

        return {
            "prediction": "Non-Toxic",
            "confidence": 0.0,
            "labels": {},
            "language": "en",
            "translated_text": "",
            "hate_speech": False,
            "hate_category": None
        }

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    language = detect_language(text)

    translated_text = translate_to_english(
        text,
        language
    )

    # --------------------------------------------------------
    # NORMALIZED TEXT
    # --------------------------------------------------------

    normalized_text = re.sub(
        r"[^a-zA-Z0-9\s'-]",
        " ",
        translated_text.lower()
    )

    normalized_text = re.sub(
        r"\s+",
        " ",
        normalized_text
    ).strip()

    # --------------------------------------------------------
    # MODEL AVAILABILITY
    # --------------------------------------------------------

    if not MODEL_AVAILABLE:

        return {
            "prediction": "Model Unavailable",
            "confidence": 0.0,
            "labels": {},
            "language": language,
            "translated_text": translated_text,
            "hate_speech": False,
            "hate_category": None,
            "error": "Trained model could not be loaded."
        }

    try:

        # ====================================================
        # TRANSFORMER MODEL
        # ====================================================

        inputs = tokenizer(
            translated_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                **inputs
            )

            logits = outputs.logits

            probabilities = torch.sigmoid(
                logits
            )[0].detach().cpu().numpy()

        # ====================================================
        # BUILD MODEL LABEL SCORES
        # ====================================================

        labels_result = {}

        active_labels = MODEL_LABELS

        for index, score in enumerate(
            probabilities
        ):

            if index < len(active_labels):

                label = str(
                    active_labels[index]
                ).lower().strip()

                labels_result[label] = round(
                    float(score),
                    4
                )

        # ====================================================
        # INDIVIDUAL MODEL SCORES
        # ====================================================

        identity_hate_score = float(
            labels_result.get(
                "identity_hate",
                0.0
            )
        )

        threat_score = float(
            labels_result.get(
                "threat",
                0.0
            )
        )

        severe_toxic_score = float(
            labels_result.get(
                "severe_toxic",
                0.0
            )
        )

        toxic_score = float(
            labels_result.get(
                "toxic",
                0.0
            )
        )

        insult_score = float(
            labels_result.get(
                "insult",
                0.0
            )
        )

        obscene_score = float(
            labels_result.get(
                "obscene",
                0.0
            )
        )

        # ====================================================
        # HATE SPEECH MODEL SCORE
        # ====================================================

        hate_scores = {

            "identity_hate":
                identity_hate_score,

            "threat":
                threat_score,

            "severe_toxic":
                severe_toxic_score
        }

        strongest_hate_label = max(
            hate_scores,
            key=hate_scores.get
        )

        strongest_hate_score = float(
            hate_scores[
                strongest_hate_label
            ]
        )

        # ====================================================
        # OFFENSIVE MODEL SCORE
        # ====================================================

        offensive_scores = {

            "toxic":
                toxic_score,

            "insult":
                insult_score,

            "obscene":
                obscene_score
        }

        strongest_offensive_label = max(
            offensive_scores,
            key=offensive_scores.get
        )

        strongest_offensive_score = float(
            offensive_scores[
                strongest_offensive_label
            ]
        )

        # ====================================================
        # THRESHOLDS
        # ====================================================

        HATE_THRESHOLD = float(
            os.getenv(
                "TEXT_HATE_THRESHOLD",
                "0.30"
            )
        )

        OFFENSIVE_THRESHOLD = float(
            os.getenv(
                "TEXT_OFFENSIVE_THRESHOLD",
                "0.50"
            )
        )

        # ====================================================
        # MODEL DETECTION
        # ====================================================

        identity_hate_detected = (
            identity_hate_score
            >= HATE_THRESHOLD
        )

        threat_detected = (
            threat_score
            >= HATE_THRESHOLD
        )

        severe_toxic_detected = (
            severe_toxic_score
            >= HATE_THRESHOLD
        )

        offensive_detected = (
            strongest_offensive_score
            >= OFFENSIVE_THRESHOLD
        )

        # ====================================================
        # RULE-BASED HATE SPEECH DETECTION
        #
        # The trained model can miss indirect hate speech.
        #
        # This layer detects statements that:
        #
        #   1. Refer to a protected/group identity
        #   2. Advocate exclusion, removal, banning,
        #      segregation, or denying residence/access
        #
        # It does NOT classify ordinary mentions of groups
        # as hate speech by themselves.
        # ====================================================

        group_patterns = [

            # Generic protected-group references
            r"\bethnic\s+group\b",
            r"\bethnic\s+groups\b",
            r"\brace\b",
            r"\bracial\s+group\b",
            r"\bracial\s+groups\b",
            r"\bnationality\b",
            r"\bnationalities\b",
            r"\breligious\s+group\b",
            r"\breligious\s+groups\b",
            r"\breligion\b",
            r"\bimmigrants?\b",
            r"\bminority\s+group\b",
            r"\bminority\s+groups\b",

            # Group-oriented wording
            r"\bpeople\s+from\s+that\s+ethnic\s+group\b",
            r"\bpeople\s+of\s+that\s+race\b",
            r"\bpeople\s+of\s+that\s+religion\b",
            r"\bpeople\s+from\s+that\s+group\b"
        ]

        exclusion_patterns = [

            # Explicit exclusion
            r"\bshould\s+not\s+be\s+allowed\b",
            r"\bmust\s+not\s+be\s+allowed\b",
            r"\bshouldn't\s+be\s+allowed\b",
            r"\bmustn't\s+be\s+allowed\b",

            # Removal / banning
            r"\bshould\s+be\s+banned\b",
            r"\bmust\s+be\s+banned\b",
            r"\bshould\s+be\s+removed\b",
            r"\bmust\s+be\s+removed\b",
            r"\bshould\s+be\s+excluded\b",
            r"\bmust\s+be\s+excluded\b",

            # Residence / community exclusion
            r"\bnot\s+allowed\s+to\s+live\b",
            r"\bnot\s+allowed\s+to\s+stay\b",
            r"\bnot\s+allowed\s+to\s+enter\b",
            r"\bnot\s+allowed\s+to\s+join\b",
            r"\bshould\s+not\s+live\b",
            r"\bshouldn't\s+live\b",
            r"\bshould\s+leave\s+our\s+community\b",
            r"\bmust\s+leave\s+our\s+community\b",

            # Segregation / exclusion
            r"\bkeep\s+them\s+out\b",
            r"\bkeep\s+that\s+group\s+out\b",
            r"\bkeep\s+those\s+people\s+out\b",
            r"\bexclude\s+them\b",
            r"\bexclude\s+that\s+group\b"
        ]

        has_group_reference = any(
            re.search(
                pattern,
                normalized_text,
                re.IGNORECASE
            )
            for pattern in group_patterns
        )

        has_exclusion_language = any(
            re.search(
                pattern,
                normalized_text,
                re.IGNORECASE
            )
            for pattern in exclusion_patterns
        )

        # ====================================================
        # RULE-BASED IDENTITY HATE
        # ====================================================

        rule_based_identity_hate = (
            has_group_reference
            and
            has_exclusion_language
        )

        # ====================================================
        # RULE-BASED HATE SCORE
        # ====================================================

        if rule_based_identity_hate:

            rule_hate_score = 0.95

        else:

            rule_hate_score = 0.0

        # ====================================================
        # COMBINE MODEL + RULE SCORE
        # ====================================================

        combined_identity_hate_score = max(
            identity_hate_score,
            rule_hate_score
        )

        # ====================================================
        # FINAL HATE DETECTION
        # ====================================================

        if (
            combined_identity_hate_score
            >= HATE_THRESHOLD
        ):

            identity_hate_detected = True

        # ====================================================
        # FINAL PREDICTION
        #
        # Hate speech has priority over ordinary toxicity.
        # ====================================================

        if identity_hate_detected:

            prediction = "Hate Speech"

            confidence = (
                combined_identity_hate_score
            )

            final_hate_category = (
                "identity_hate"
            )

        elif threat_detected:

            prediction = "Hate Speech"

            confidence = threat_score

            final_hate_category = (
                "threat"
            )

        elif severe_toxic_detected:

            prediction = "Hate Speech"

            confidence = severe_toxic_score

            final_hate_category = (
                "severe_toxic"
            )

        elif offensive_detected:

            prediction = "Toxic"

            confidence = (
                strongest_offensive_score
            )

            final_hate_category = None

        else:

            prediction = "Non-Toxic"

            max_score = (
                max(
                    labels_result.values()
                )
                if labels_result
                else 0.0
            )

            confidence = (
                1.0 - max_score
            )

            final_hate_category = None

        # ====================================================
        # UPDATE IDENTITY-HATE SCORE
        # ====================================================

        if rule_based_identity_hate:

            labels_result[
                "identity_hate"
            ] = round(
                combined_identity_hate_score,
                4
            )

        # ====================================================
        # EXTRA SCORES FOR FRONTEND
        # ====================================================

        labels_result[
            "hate_speech_score"
        ] = round(
            float(
                max(
                    combined_identity_hate_score,
                    threat_score,
                    severe_toxic_score
                )
            ),
            4
        )

        labels_result[
            "offensive_score"
        ] = round(
            float(
                strongest_offensive_score
            ),
            4
        )

        labels_result[
            "hate_category"
        ] = final_hate_category

        # ====================================================
        # RESULT
        # ====================================================

        return {

            "prediction":
                prediction,

            "confidence":
                round(
                    float(confidence),
                    4
                ),

            "labels":
                labels_result,

            "language":
                language,

            "translated_text":
                translated_text,

            "hate_speech":
                prediction == "Hate Speech",

            "hate_category":
                final_hate_category
        }

    except Exception as e:

        print(
            "[PREDICTION] ERROR:",
            str(e)
        )

        traceback.print_exc()

        return {

            "prediction":
                "Error",

            "confidence":
                0.0,

            "labels":
                {},

            "language":
                language,

            "translated_text":
                translated_text,

            "hate_speech":
                False,

            "hate_category":
                None,

            "error":
                str(e)
        }

# ============================================================
# SAVE PREDICTION HISTORY
# ============================================================

def save_prediction_history(
    user_id,
    input_text,
    result,
    source="text"
):

    conn = get_db()

    conn.execute(
        """
        INSERT INTO prediction_history
        (
            user_id,
            input_text,
            language,
            translated_text,
            prediction,
            confidence,
            labels_json,
            source,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            input_text,
            result.get(
                "language",
                "en"
            ),
            result.get(
                "translated_text",
                ""
            ),
            result.get(
                "prediction",
                "Unknown"
            ),
            float(
                result.get(
                    "confidence",
                    0
                )
            ),
            json.dumps(
                result.get(
                    "labels",
                    {}
                )
            ),
            source,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "index.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )

    if request.method == "POST":

        username = clean_text(
            request.form.get(
                "username",
                ""
            )
        )

        email = clean_text(
            request.form.get(
                "email",
                ""
            )
        ).lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not username:

            flash(
                "Username is required.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if not email:

            flash(
                "Email is required.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if not re.match(
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            email
        ):

            flash(
                "Please enter a valid email address.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        try:

            conn = get_db()

            existing = conn.execute(
                """
                SELECT id
                FROM users
                WHERE LOWER(username) = LOWER(?)
                   OR LOWER(email) = LOWER(?)
                """,
                (
                    username,
                    email
                )
            ).fetchone()

            if existing:

                conn.close()

                flash(
                    "Username or email already exists.",
                    "danger"
                )

                return render_template(
                    "register.html"
                )

            hashed_password = (
                generate_password_hash(
                    password
                )
            )

            conn.execute(
                """
                INSERT INTO users
                (
                    username,
                    email,
                    password,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    hashed_password,
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            )

            conn.commit()

            conn.close()

            flash(
                "Registration successful. Please login.",
                "success"
            )

            return redirect(
                url_for("login")
            )

        except sqlite3.IntegrityError:

            flash(
                "Username or email already exists.",
                "danger"
            )

        except Exception as e:

            print(
                "[REGISTER ERROR]",
                str(e)
            )

            traceback.print_exc()

            flash(
                "Registration failed because of a database error.",
                "danger"
            )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )

    if request.method == "POST":

        login_value = clean_text(
            request.form.get(
                "email",
                ""
            )
        ).lower()

        password = request.form.get(
            "password",
            ""
        )

        if (
            not login_value
            or not password
        ):

            flash(
                "Email/username and password are required.",
                "danger"
            )

            return render_template(
                "login.html"
            )

        try:

            conn = get_db()

            user = conn.execute(
                """
                SELECT *
                FROM users
                WHERE LOWER(email) = LOWER(?)
                   OR LOWER(username) = LOWER(?)
                """,
                (
                    login_value,
                    login_value
                )
            ).fetchone()

            conn.close()

            if user is None:

                flash(
                    "Invalid username/email or password.",
                    "danger"
                )

                return render_template(
                    "login.html"
                )

            if not check_password_hash(
                user["password"],
                password
            ):

                flash(
                    "Invalid username/email or password.",
                    "danger"
                )

                return render_template(
                    "login.html"
                )

            session.clear()

            session["user_id"] = (
                user["id"]
            )

            session["username"] = (
                user["username"]
            )

            session["email"] = (
                user["email"]
            )

            flash(
                "Login successful.",
                "success"
            )

            return redirect(
                url_for("dashboard")
            )

        except Exception as e:

            print(
                "[LOGIN ERROR]",
                str(e)
            )

            traceback.print_exc()

            flash(
                "Login failed because of a database error.",
                "danger"
            )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    user_id = session["user_id"]

    conn = get_db()

    try:

        user = conn.execute(
            """
            SELECT
                id,
                username,
                email,
                created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        ).fetchone()

        total_predictions = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
            """,
            (user_id,)
        ).fetchone()[0]

        toxic_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND LOWER(prediction) = 'toxic'
            """,
            (user_id,)
        ).fetchone()[0]

        non_toxic_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND LOWER(prediction)
                  IN ('non-toxic', 'non toxic')
            """,
            (user_id,)
        ).fetchone()[0]

        today_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND DATE(created_at) =
                  DATE('now', 'localtime')
            """,
            (user_id,)
        ).fetchone()[0]

        toxicity_rate = (
            round(
                (
                    toxic_count /
                    total_predictions
                ) * 100,
                1
            )
            if total_predictions > 0
            else 0
        )

        total_analyses = conn.execute(
            """
            SELECT COUNT(*)
            FROM analysis_history
            WHERE user_id = ?
            """,
            (user_id,)
        ).fetchone()[0]

        youtube_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM analysis_history
            WHERE user_id = ?
              AND LOWER(analysis_type) = 'youtube'
            """,
            (user_id,)
        ).fetchone()[0]

        recent_rows = conn.execute(
            """
            SELECT
                id,
                input_text,
                prediction,
                confidence,
                source,
                language,
                created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (user_id,)
        ).fetchall()

        recent_history = []

        for row in recent_rows:

            recent_history.append({

                "id":
                    row["id"],

                "input_text":
                    row["input_text"],

                "prediction":
                    row["prediction"],

                "confidence":
                    row["confidence"],

                "source":
                    row["source"] or "text",

                "language":
                    row["language"] or "en",

                "created_at":
                    row["created_at"]
            })

        activity_rows = conn.execute(
            """
            SELECT
                DATE(created_at) AS analysis_date,
                COUNT(*) AS total
            FROM prediction_history
            WHERE user_id = ?
            GROUP BY DATE(created_at)
            ORDER BY analysis_date ASC
            LIMIT 30
            """,
            (user_id,)
        ).fetchall()

        chart_labels = [
            row["analysis_date"]
            for row in activity_rows
        ]

        chart_values = [
            row["total"]
            for row in activity_rows
        ]

        source_rows = conn.execute(
            """
            SELECT
                COALESCE(
                    NULLIF(source, ''),
                    'text'
                ) AS source,
                COUNT(*) AS count
            FROM prediction_history
            WHERE user_id = ?
            GROUP BY
                COALESCE(
                    NULLIF(source, ''),
                    'text'
                )
            ORDER BY count DESC
            """,
            (user_id,)
        ).fetchall()

        source_distribution = []

        for row in source_rows:

            source_distribution.append({

                "source":
                    row["source"].upper(),

                "count":
                    row["count"]
            })

        if not source_distribution:

            source_distribution = [
                {
                    "source": "TEXT",
                    "count": 0
                }
            ]

        label_counts = {
            label: 0
            for label in LABELS
        }

        prediction_rows = conn.execute(
            """
            SELECT labels_json
            FROM prediction_history
            WHERE user_id = ?
            """,
            (user_id,)
        ).fetchall()

        for row in prediction_rows:

            try:

                labels_data = json.loads(
                    row["labels_json"] or "{}"
                )

                for label, score in labels_data.items():

                    if (
                        label in label_counts
                        and float(score) >= 0.5
                    ):

                        label_counts[label] += 1

            except Exception:

                continue

        category_labels = [
            "Toxic",
            "Severe Toxic",
            "Obscene",
            "Threat",
            "Insult",
            "Identity Hate"
        ]

        category_values = [

            label_counts["toxic"],
            label_counts["severe_toxic"],
            label_counts["obscene"],
            label_counts["threat"],
            label_counts["insult"],
            label_counts["identity_hate"]
        ]

        stats = {

            "prediction_count":
                total_predictions,

            "toxic_count":
                toxic_count,

            "non_toxic_count":
                non_toxic_count,

            "youtube_count":
                youtube_count,

            "today_count":
                today_count,

            "total_analyses":
                total_analyses,

            "toxic_percentage":
                toxicity_rate
        }

        model_available_value = bool(
            MODEL_AVAILABLE
        )

        conn.close()

        return render_template(

            "dashboard.html",

            user=user,

            stats=stats,

            total_predictions=total_predictions,

            toxic_count=toxic_count,

            non_toxic_count=non_toxic_count,

            today_count=today_count,

            toxicity_rate=toxicity_rate,

            total_analyses=total_analyses,

            youtube_count=youtube_count,

            recent_history=recent_history,

            chart_labels=chart_labels,

            chart_values=chart_values,

            labels=LABELS,

            label_display_names=LABEL_DISPLAY_NAMES,

            label_counts=label_counts,

            category_labels=category_labels,

            category_values=category_values,

            source_distribution=source_distribution,

            model_available=model_available_value
        )

    except Exception as e:

        try:
            conn.close()
        except Exception:
            pass

        print(
            "\n[DASHBOARD ERROR]"
        )

        print(
            str(e)
        )

        traceback.print_exc()

        return render_template(

            "dashboard.html",

            user=get_current_user(),

            stats={

                "prediction_count": 0,
                "toxic_count": 0,
                "non_toxic_count": 0,
                "youtube_count": 0,
                "today_count": 0,
                "total_analyses": 0,
                "toxic_percentage": 0
            },

            total_predictions=0,
            toxic_count=0,
            non_toxic_count=0,
            today_count=0,
            toxicity_rate=0,
            total_analyses=0,
            youtube_count=0,
            recent_history=[],
            chart_labels=[],
            chart_values=[],
            labels=LABELS,
            label_display_names=LABEL_DISPLAY_NAMES,

            label_counts={
                label: 0
                for label in LABELS
            },

            category_labels=[
                "Toxic",
                "Severe Toxic",
                "Obscene",
                "Threat",
                "Insult",
                "Identity Hate"
            ],

            category_values=[
                0, 0, 0, 0, 0, 0
            ],

            source_distribution=[
                {
                    "source": "TEXT",
                    "count": 0
                }
            ],

            model_available=False
        )


# ============================================================
# TEXT ANALYSIS
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
@login_required
def predict():

    result = None

    if request.method == "POST":

        text = clean_text(
            request.form.get(
                "text",
                ""
            )
        )

        if not text:

            flash(
                "Please enter some text.",
                "warning"
            )

            return render_template(
                "predict.html",
                result=None
            )

        result = predict_text(
            text
        )

        if result.get("prediction") not in [
            "Error",
            "Model Unavailable"
        ]:

            save_prediction_history(
                session["user_id"],
                text,
                result,
                "text"
            )

    return render_template(
        "predict.html",
        result=result
    )


# ============================================================
# API - PREDICT
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
@login_required
def api_predict():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        text = clean_text(
            data.get(
                "text",
                ""
            )
        )

        if not text:

            return jsonify({
                "success": False,
                "error": "Text is required."
            }), 400

        result = predict_text(
            text
        )

        if result.get("prediction") not in [
            "Error",
            "Model Unavailable"
        ]:

            save_prediction_history(
                session["user_id"],
                text,
                result,
                "api"
            )

        return jsonify({

            "success":
                True,

            "result":
                result
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# CSV ANALYSIS
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
@login_required
def upload():

    results = []

    if request.method == "POST":

        file = request.files.get(
            "file"
        )

        if not file:

            flash(
                "Please select a CSV file.",
                "warning"
            )

            return render_template(
                "upload.html",
                results=[]
            )

        if file.filename == "":

            flash(
                "No file selected.",
                "warning"
            )

            return render_template(
                "upload.html",
                results=[]
            )

        if not file.filename.lower().endswith(
            ".csv"
        ):

            flash(
                "Only CSV files are supported.",
                "danger"
            )

            return render_template(
                "upload.html",
                results=[]
            )

        try:

            df = pd.read_csv(
                file
            )

            if df.empty:

                flash(
                    "The CSV file is empty.",
                    "warning"
                )

                return render_template(
                    "upload.html",
                    results=[]
                )

            possible_columns = [

                "text",
                "comment_text",
                "comment",
                "Text",
                "Comment",
                "commentText"
            ]

            text_column = None

            for column in possible_columns:

                if column in df.columns:

                    text_column = column

                    break

            if text_column is None:

                text_column = df.columns[0]

            for index, row in df.iterrows():

                text = clean_text(
                    row[text_column]
                )

                if not text:
                    continue

                result = predict_text(
                    text
                )

                record = {

                    "text":
                        text,

                    "language":
                        result.get(
                            "language",
                            "en"
                        ),

                    "translated_text":
                        result.get(
                            "translated_text",
                            ""
                        ),

                    "prediction":
                        result.get(
                            "prediction",
                            "Unknown"
                        ),

                    "confidence":
                        result.get(
                            "confidence",
                            0
                        ),

                    "labels":
                        result.get(
                            "labels",
                            {}
                        )
                }

                results.append(
                    record
                )

                if result.get(
                    "prediction"
                ) not in [
                    "Error",
                    "Model Unavailable"
                ]:

                    save_prediction_history(

                        session["user_id"],
                        text,
                        result,
                        "csv"
                    )

            flash(
                f"Successfully analyzed {len(results)} rows.",
                "success"
            )

        except Exception as e:

            print(
                "[CSV ERROR]",
                str(e)
            )

            traceback.print_exc()

            flash(
                "Unable to process the CSV file.",
                "danger"
            )

    return render_template(
        "upload.html",
        results=results
    )


# ============================================================
# YOUTUBE VIDEO ID EXTRACTION
# ============================================================

def extract_youtube_video_id(url):

    if not url:
        return None

    url = url.strip()

    if re.match(
        r"^[A-Za-z0-9_-]{11}$",
        url
    ):

        return url

    try:

        parsed = urlparse(
            url
        )

        hostname = (
            parsed.hostname or ""
        ).lower()

        if (
            "youtube.com" in hostname
            or "youtube-nocookie.com" in hostname
        ):

            query = parse_qs(
                parsed.query
            )

            if "v" in query:

                return query["v"][0]

            match = re.search(
                r"/shorts/([A-Za-z0-9_-]{11})",
                parsed.path
            )

            if match:

                return match.group(1)

            match = re.search(
                r"/embed/([A-Za-z0-9_-]{11})",
                parsed.path
            )

            if match:

                return match.group(1)

        if "youtu.be" in hostname:

            video_id = (
                parsed.path
                .strip("/")
                .split("/")[0]
            )

            if re.match(
                r"^[A-Za-z0-9_-]{11}$",
                video_id
            ):

                return video_id

    except Exception:

        pass

    return None


# ============================================================
# YOUTUBE API CLIENT
# ============================================================

def get_youtube_service():

    api_key = os.getenv(
        "YOUTUBE_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "YOUTUBE_API_KEY is not configured."
        )

    if build is None:

        raise ImportError(
            "google-api-python-client is not installed."
        )

    http = httplib2.Http(
        timeout=60
    )

    return build(
        "youtube",
        "v3",
        developerKey=api_key,
        http=http,
        cache_discovery=False
    )


# ============================================================
# GET VIDEO INFORMATION
# ============================================================

def get_video_information(
    youtube,
    video_id
):

    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    items = response.get(
        "items",
        []
    )

    if not items:
        return None

    item = items[0]

    snippet = item.get(
        "snippet",
        {}
    )

    statistics = item.get(
        "statistics",
        {}
    )

    return {

        "video_id":
            video_id,

        "title":
            snippet.get(
                "title",
                ""
            ),

        "channel":
            snippet.get(
                "channelTitle",
                ""
            ),

        "published_at":
            snippet.get(
                "publishedAt",
                ""
            ),

        "view_count":
            int(
                statistics.get(
                    "viewCount",
                    0
                )
            ),

        "like_count":
            int(
                statistics.get(
                    "likeCount",
                    0
                )
            ),

        "comment_count":
            int(
                statistics.get(
                    "commentCount",
                    0
                )
            )
    }


# ============================================================
# GET ALL ACCESSIBLE YOUTUBE COMMENTS + REPLIES
# ============================================================

def get_all_youtube_comments(
    youtube,
    video_id
):

    comments = []

    next_page_token = None

    max_retries = 3

    while True:

        response = None

        for attempt in range(
            max_retries
        ):

            try:

                print(
                    f"[YOUTUBE] Fetching comments page "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )

                request_obj = (
                    youtube.commentThreads().list(
                        part="snippet,replies",
                        videoId=video_id,
                        maxResults=100,
                        pageToken=next_page_token,
                        textFormat="plainText"
                    )
                )

                response = (
                    request_obj.execute()
                )

                print(
                    "[YOUTUBE] Comment page received."
                )

                break

            except Exception as e:

                print(
                    f"[YOUTUBE] Comment request failed "
                    f"(attempt {attempt + 1}/{max_retries}): "
                    f"{str(e)}"
                )

                if attempt < (
                    max_retries - 1
                ):

                    print(
                        "[YOUTUBE] Retrying..."
                    )

                    time.sleep(
                        2
                    )

                else:

                    raise

        if response is None:

            raise RuntimeError(
                "Unable to retrieve YouTube comments."
            )

        for item in response.get(
            "items",
            []
        ):

            thread_snippet = item.get(
                "snippet",
                {}
            )

            top_level_comment = (
                thread_snippet.get(
                    "topLevelComment",
                    {}
                )
            )

            top_comment_snippet = (
                top_level_comment.get(
                    "snippet",
                    {}
                )
            )

            top_comment_id = (
                top_level_comment.get(
                    "id"
                )
            )

            comments.append({

                "comment_id":
                    top_comment_id,

                "parent_id":
                    None,

                "author":
                    top_comment_snippet.get(
                        "authorDisplayName",
                        "Unknown"
                    ),

                "text":
                    top_comment_snippet.get(
                        "textDisplay",
                        ""
                    ),

                "is_reply":
                    False
            })

            replies = (
                item.get(
                    "replies",
                    {}
                ).get(
                    "comments",
                    []
                )
            )

            for reply in replies:

                reply_snippet = reply.get(
                    "snippet",
                    {}
                )

                comments.append({

                    "comment_id":
                        reply.get(
                            "id"
                        ),

                    "parent_id":
                        reply_snippet.get(
                            "parentId"
                        ),

                    "author":
                        reply_snippet.get(
                            "authorDisplayName",
                            "Unknown"
                        ),

                    "text":
                        reply_snippet.get(
                            "textDisplay",
                            ""
                        ),

                    "is_reply":
                        True
                })

        next_page_token = (
            response.get(
                "nextPageToken"
            )
        )

        if not next_page_token:

            break

    print(
        f"[YOUTUBE] Total comments/replies fetched: "
        f"{len(comments)}"
    )

    return comments


# ============================================================
# ANALYZE YOUTUBE COMMENTS
# ============================================================

def analyze_youtube_comments(
    comments,
    user_id,
    video_id
):

    analyzed = []

    category_counts = {
        label: 0
        for label in LABELS
    }

    toxic_comments = 0

    non_toxic_comments = 0

    conn = get_db()

    try:

        for comment in comments:

            text = clean_text(
                comment.get(
                    "text",
                    ""
                )
            )

            if not text:

                continue

            result = predict_text(
                text
            )

            prediction = result.get(
                "prediction",
                "Unknown"
            )

            confidence = float(
                result.get(
                    "confidence",
                    0
                )
            )

            labels = result.get(
                "labels",
                {}
            )

            if prediction == "Toxic":

                toxic_comments += 1

            elif prediction == "Non-Toxic":

                non_toxic_comments += 1

            for label, score in labels.items():

                if (
                    score >= 0.5
                    and label in category_counts
                ):

                    category_counts[
                        label
                    ] += 1

            conn.execute(
                """
                INSERT INTO youtube_comments
                (
                    user_id,
                    video_id,
                    comment_id,
                    author,
                    comment_text,
                    prediction,
                    confidence,
                    labels_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    video_id,
                    comment.get(
                        "comment_id"
                    ),
                    comment.get(
                        "author",
                        "Unknown"
                    ),
                    text,
                    prediction,
                    confidence,
                    json.dumps(
                        labels
                    ),
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            )

            analyzed.append({

                "comment_id":
                    comment.get(
                        "comment_id"
                    ),

                "parent_id":
                    comment.get(
                        "parent_id"
                    ),

                "author":
                    comment.get(
                        "author",
                        "Unknown"
                    ),

                "text":
                    text,

                "is_reply":
                    comment.get(
                        "is_reply",
                        False
                    ),

                "prediction":
                    prediction,

                "confidence":
                    confidence,

                "labels":
                    labels
            })

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()

    total_comments = (
        toxic_comments +
        non_toxic_comments
    )

    toxic_percentage = (
        round(
            (
                toxic_comments /
                total_comments
            ) * 100,
            2
        )
        if total_comments > 0
        else 0
    )

    non_toxic_percentage = (
        round(
            (
                non_toxic_comments /
                total_comments
            ) * 100,
            2
        )
        if total_comments > 0
        else 0
    )

    return {

        "comments":
            analyzed,

        "total_comments":
            total_comments,

        "toxic_comments":
            toxic_comments,

        "non_toxic_comments":
            non_toxic_comments,

        "toxic_percentage":
            toxic_percentage,

        "non_toxic_percentage":
            non_toxic_percentage,

        "category_counts":
            category_counts
    }


# ============================================================
# YOUTUBE PAGE
# ============================================================

@app.route(
    "/youtube",
    methods=["GET", "POST"]
)
@login_required
def youtube_analysis():

    result = None

    video = None

    if request.method == "POST":

        video_url = clean_text(
            request.form.get(
                "url",
                ""
            )
        )

        video_id = extract_youtube_video_id(
            video_url
        )

        if not video_id:

            flash(
                "Invalid YouTube URL.",
                "danger"
            )

            return render_template(
                "youtube.html",
                result=None,
                video=None
            )

        try:

            print(
                f"[YOUTUBE] Starting analysis "
                f"for video: {video_id}"
            )

            youtube = (
                get_youtube_service()
            )

            video = (
                get_video_information(
                    youtube,
                    video_id
                )
            )

            if video is None:

                flash(
                    "YouTube video not found.",
                    "danger"
                )

                return render_template(
                    "youtube.html",
                    result=None,
                    video=None
                )

            comments = (
                get_all_youtube_comments(
                    youtube,
                    video_id
                )
            )

            result = (
                analyze_youtube_comments(
                    comments,
                    session["user_id"],
                    video_id
                )
            )

            conn = get_db()

            conn.execute(
                """
                INSERT INTO analysis_history
                (
                    user_id,
                    analysis_type,
                    source,
                    total_items,
                    toxic_items,
                    non_toxic_items,
                    toxic_percentage,
                    result_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session["user_id"],
                    "youtube",
                    video_id,
                    result["total_comments"],
                    result["toxic_comments"],
                    result["non_toxic_comments"],
                    result["toxic_percentage"],
                    json.dumps({
                        "category_counts":
                            result["category_counts"]
                    }),
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            )

            conn.commit()

            conn.close()

            flash(
                f"Analyzed "
                f"{result['total_comments']} "
                f"accessible comments and replies.",
                "success"
            )

        except Exception as e:

            print(
                "[YOUTUBE ERROR]"
            )

            print(
                str(e)
            )

            traceback.print_exc()

            flash(
                f"YouTube analysis failed: "
                f"{str(e)}",
                "danger"
            )

    return render_template(
        "youtube.html",
        result=result,
        video=video
    )


# ============================================================
# API - YOUTUBE SUMMARY
# ============================================================

@app.route(
    "/api/youtube/summary",
    methods=["GET", "POST"]
)
@login_required
def api_youtube_summary():

    try:

        if request.method == "POST":

            data = request.get_json(
                silent=True
            ) or {}

            video_url = clean_text(
                data.get(
                    "url",
                    ""
                )
            )

        else:

            video_url = clean_text(
                request.args.get(
                    "url",
                    ""
                )
            )

        video_id = extract_youtube_video_id(
            video_url
        )

        if not video_id:

            return jsonify({

                "success":
                    False,

                "error":
                    "Invalid YouTube URL."

            }), 400

        youtube = (
            get_youtube_service()
        )

        video = (
            get_video_information(
                youtube,
                video_id
            )
        )

        if video is None:

            return jsonify({

                "success":
                    False,

                "error":
                    "Video not found."

            }), 404

        comments = (
            get_all_youtube_comments(
                youtube,
                video_id
            )
        )

        result = (
            analyze_youtube_comments(
                comments,
                session["user_id"],
                video_id
            )
        )

        response = {

            "success":
                True,

            "video":
                video,

            "total_comments":
                result["total_comments"],

            "toxic_comments":
                result["toxic_comments"],

            "non_toxic_comments":
                result["non_toxic_comments"],

            "toxic_percentage":
                result["toxic_percentage"],

            "non_toxic_percentage":
                result["non_toxic_percentage"],

            "category_counts":
                result["category_counts"],

            "comments":
                result["comments"]
        }

        return jsonify(
            response
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# TWITTER/X CLASSIFICATION
# FIXED HATE SPEECH DETECTION
# ============================================================

def classify_twitter_result(text, result):
    """
    Twitter/X 3-class classification.

    01 = Hate Speech
    02 = Offensive Language
    03 = Neither

    IMPORTANT:
    The main predict_text() function is the authoritative
    toxicity/hate-speech detector.

    If predict_text() already identifies Hate Speech, this
    function preserves that result instead of recalculating
    it with a second threshold.
    """

    labels = result.get("labels", {}) or {}

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

            cleaned_labels[normalized_label] = float(score)

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
    # CHECK MAIN predict_text() RESULT FIRST
    # --------------------------------------------------------
    #
    # This is the important fix.
    #
    # Your working predict_text() can return:
    #
    # prediction = "Hate Speech"
    # hate_speech = True
    # hate_category = "identity_hate"
    #
    # Twitter must preserve that result.
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

        # Use the actual hate category returned by
        # predict_text() when available.
        if base_hate_category:

            hate_label = str(
                base_hate_category
            ).lower().strip()

            # If the returned category is valid,
            # use its actual score.
            if hate_label in hate_scores:

                hate_score = float(
                    hate_scores[hate_label]
                )

        # If the main detector says hate speech but the
        # model score available here is unexpectedly low,
        # use the confidence returned by predict_text().
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
# TWITTER / X ANALYSIS PAGE
# ============================================================

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
            # USE THE SAME WORKING MODEL PREDICTION
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
                base_result
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


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():

    user_id = session[
        "user_id"
    ]

    conn = get_db()

    rows = conn.execute(
        """
        SELECT *
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (
            user_id,
        )
    ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        history=rows
    )


# ============================================================
# DELETE HISTORY
# ============================================================

@app.route(
    "/history/delete/<int:history_id>",
    methods=["POST"]
)
@login_required
def delete_history(
    history_id
):

    conn = get_db()

    conn.execute(
        """
        DELETE FROM prediction_history
        WHERE id = ?
          AND user_id = ?
        """,
        (
            history_id,
            session["user_id"]
        )
    )

    conn.commit()

    conn.close()

    flash(
        "History item deleted.",
        "success"
    )

    return redirect(
        url_for(
            "history"
        )
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/history/clear",
    methods=["POST"]
)
@login_required
def clear_history():

    conn = get_db()

    conn.execute(
        """
        DELETE FROM prediction_history
        WHERE user_id = ?
        """,
        (
            session["user_id"],
        )
    )

    conn.commit()

    conn.close()

    flash(
        "Prediction history cleared.",
        "success"
    )

    return redirect(
        url_for(
            "history"
        )
    )


# ============================================================
# API - HISTORY
# ============================================================

@app.route(
    "/api/history",
    methods=["GET"]
)
@login_required
def api_history():

    conn = get_db()

    rows = conn.execute(
        """
        SELECT *
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (
            session["user_id"],
        )
    ).fetchall()

    conn.close()

    history_data = []

    for row in rows:

        try:

            labels_data = json.loads(
                row["labels_json"]
                or "{}"
            )

        except Exception:

            labels_data = {}

        history_data.append({

            "id":
                row["id"],

            "input_text":
                row["input_text"],

            "language":
                row["language"],

            "translated_text":
                row["translated_text"],

            "prediction":
                row["prediction"],

            "confidence":
                row["confidence"],

            "labels":
                labels_data,

            "source":
                row["source"],

            "created_at":
                row["created_at"]
        })

    return jsonify({

        "success":
            True,

        "history":
            history_data
    })


# ============================================================
# DASHBOARD CHART DATA API
# ============================================================

@app.route(
    "/api/dashboard/stats",
    methods=["GET"]
)
@login_required
def dashboard_stats():

    user_id = session[
        "user_id"
    ]

    conn = get_db()

    try:

        toxic = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND LOWER(prediction) = 'toxic'
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        non_toxic = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND LOWER(prediction)
                  IN ('non-toxic', 'non toxic')
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        daily_rows = conn.execute(
            """
            SELECT
                DATE(created_at) AS date,
                COUNT(*) AS count
            FROM prediction_history
            WHERE user_id = ?
            GROUP BY DATE(created_at)
            ORDER BY date ASC
            """,
            (
                user_id,
            )
        ).fetchall()

        source_rows = conn.execute(
            """
            SELECT
                COALESCE(
                    NULLIF(source, ''),
                    'text'
                ) AS source,
                COUNT(*) AS count
            FROM prediction_history
            WHERE user_id = ?
            GROUP BY
                COALESCE(
                    NULLIF(source, ''),
                    'text'
                )
            """,
            (
                user_id,
            )
        ).fetchall()

        conn.close()

        return jsonify({

            "success":
                True,

            "prediction_distribution": {

                "toxic":
                    toxic,

                "non_toxic":
                    non_toxic
            },

            "daily_activity": [

                {
                    "date":
                        row["date"],

                    "count":
                        row["count"]
                }

                for row in daily_rows
            ],

            "source_distribution": [

                {
                    "source":
                        row["source"],

                    "count":
                        row["count"]
                }

                for row in source_rows
            ]
        })

    except Exception as e:

        try:
            conn.close()
        except Exception:
            pass

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
@login_required
def profile():

    user_id = session[
        "user_id"
    ]

    conn = get_db()

    try:

        user = conn.execute(
            """
            SELECT
                id,
                username,
                email,
                created_at
            FROM users
            WHERE id = ?
            """,
            (
                user_id,
            )
        ).fetchone()

        total_predictions = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        toxic_predictions = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND LOWER(prediction) = 'toxic'
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        non_toxic_predictions = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND LOWER(prediction)
                  IN ('non-toxic', 'non toxic')
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        toxicity_rate = (

            round(

                (
                    toxic_predictions /
                    total_predictions
                ) * 100,

                1
            )

            if total_predictions > 0

            else 0
        )

        today_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND DATE(created_at) =
                  DATE('now', 'localtime')
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        total_analyses = conn.execute(
            """
            SELECT COUNT(*)
            FROM analysis_history
            WHERE user_id = ?
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        youtube_analyses = conn.execute(
            """
            SELECT COUNT(*)
            FROM analysis_history
            WHERE user_id = ?
              AND LOWER(analysis_type) = 'youtube'
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        csv_predictions = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND LOWER(source) = 'csv'
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        text_predictions = conn.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND LOWER(source) IN
                  ('text', 'api')
            """,
            (
                user_id,
            )
        ).fetchone()[0]

        recent_history = conn.execute(
            """
            SELECT
                id,
                input_text,
                prediction,
                confidence,
                source,
                language,
                created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (
                user_id,
            )
        ).fetchall()

        source_rows = conn.execute(
            """
            SELECT
                COALESCE(
                    NULLIF(source, ''),
                    'text'
                ) AS source,
                COUNT(*) AS count
            FROM prediction_history
            WHERE user_id = ?
            GROUP BY
                COALESCE(
                    NULLIF(source, ''),
                    'text'
                )
            ORDER BY count DESC
            """,
            (
                user_id,
            )
        ).fetchall()

        source_distribution = []

        for row in source_rows:

            source_distribution.append({

                "source":
                    row["source"].upper(),

                "count":
                    row["count"]
            })

        if not source_distribution:

            source_distribution = [

                {
                    "source":
                        "TEXT",

                    "count":
                        0
                }
            ]

        model_status = bool(
            MODEL_AVAILABLE
        )

        created_at = (

            user["created_at"]

            if user

            else ""
        )

        conn.close()

        return render_template(

            "profile.html",

            user=user,

            total_predictions=
                total_predictions,

            toxic_predictions=
                toxic_predictions,

            non_toxic_predictions=
                non_toxic_predictions,

            toxicity_rate=
                toxicity_rate,

            today_count=
                today_count,

            total_analyses=
                total_analyses,

            youtube_analyses=
                youtube_analyses,

            csv_predictions=
                csv_predictions,

            text_predictions=
                text_predictions,

            recent_history=
                recent_history,

            source_distribution=
                source_distribution,

            created_at=
                created_at,

            model_available=
                model_status
        )

    except Exception as e:

        try:
            conn.close()
        except Exception:
            pass

        print(
            "[PROFILE ERROR]",
            str(e)
        )

        traceback.print_exc()

        flash(
            "Unable to load your profile.",
            "danger"
        )

        return redirect(
            url_for(
                "dashboard"
            )
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "ok",

        "application":
            "Hate Speech Toxicity Detector",

        "model_available":
            MODEL_AVAILABLE,

        "device":
            str(DEVICE),

        "database":
            os.path.exists(
                DATABASE
            )
    })


# ============================================================
# MODEL STATUS
# ============================================================

@app.route(
    "/api/model-status"
)
def model_status():

    return jsonify({

        "model_available":
            MODEL_AVAILABLE,

        "model_path":
            MODEL_PATH,

        "device":
            str(DEVICE),

        "labels":
            MODEL_LABELS
    })


# ============================================================
# 404
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    if request.path.startswith(
        "/api/"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "Endpoint not found"

        }), 404

    return render_template(
        "404.html"
    ), 404


# ============================================================
# 413
# ============================================================

@app.errorhandler(413)
def request_too_large(error):

    if request.path.startswith(
        "/api/"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "Uploaded file is too large."

        }), 413

    flash(
        "Uploaded file is too large.",
        "danger"
    )

    return redirect(
        url_for(
            "upload"
        )
    )


# ============================================================
# 500
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    traceback.print_exc()

    if request.path.startswith(
        "/api/"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "Internal server error."

        }), 500

    return render_template(
        "500.html"
    ), 500


# ============================================================
# TEMPLATE GLOBALS
# ============================================================

@app.context_processor
def inject_globals():

    return {

        "current_user":
            get_current_user(),

        "model_available":
            MODEL_AVAILABLE,

        "app_name":
            "Hate Speech Toxicity Detector"
    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 70)

    print(
        "Hate Speech Toxicity Detector SERVER"
    )

    print("=" * 70)

    print(
        "Dashboard : "
        "http://127.0.0.1:5000/dashboard"
    )

    print(
        "Login     : "
        "http://127.0.0.1:5000/login"
    )

    print(
        "Register  : "
        "http://127.0.0.1:5000/register"
    )

    print(
        "Predict   : "
        "http://127.0.0.1:5000/predict"
    )

    print(
        "Upload    : "
        "http://127.0.0.1:5000/upload"
    )

    print(
        "YouTube   : "
        "http://127.0.0.1:5000/youtube"
    )

    print(
        "Twitter   : "
        "http://127.0.0.1:5000/twitter"
    )

    print(
        "History   : "
        "http://127.0.0.1:5000/history"
    )

    print(
        "Profile   : "
        "http://127.0.0.1:5000/profile"
    )

    print(
        "Health    : "
        "http://127.0.0.1:5000/health"
    )

    print("=" * 70)

    print()

    print(
        "[TWITTER] Hate labels:",
        TWITTER_HATE_LABELS
    )

    print(
        "[TWITTER] Offensive labels:",
        TWITTER_OFFENSIVE_LABELS
    )

    print(
        "[TWITTER] Hate Threshold:",
        os.getenv(
            "TWITTER_HATE_THRESHOLD",
            "0.40"
        )
    )

    print(
        "[TWITTER] Offensive Threshold:",
        os.getenv(
            "TWITTER_OFFENSIVE_THRESHOLD",
            "0.50"
        )
    )

    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )