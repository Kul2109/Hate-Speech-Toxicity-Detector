# ============================================================
# HATE SPEECH TOXICITY DETECTOR  - COMPLETE FLASK APPLICATION
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

from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv


# ============================================================
# TWITTER / X - REAL RoBERTa V3 SERVICE
# ============================================================

try:
    from twitter_service import predict_twitter_text

    TWITTER_MODEL_AVAILABLE = True

    print("[TWITTER] twitter_service imported successfully.")

except Exception as e:

    predict_twitter_text = None
    TWITTER_MODEL_AVAILABLE = False

    print("[TWITTER] ERROR importing twitter_service:", str(e))
    traceback.print_exc()


# ============================================================
# TRANSFORMERS
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


# ============================================================
# LANGUAGE DETECTION
# ============================================================

try:

    from langdetect import detect

except ImportError:

    detect = None


# ============================================================
# TRANSLATION
# ============================================================

try:

    from deep_translator import GoogleTranslator

except ImportError:

    GoogleTranslator = None


# ============================================================
# YOUTUBE API
# ============================================================

try:

    from googleapiclient.discovery import build

except ImportError:

    build = None


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# APPLICATION SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "hate-speech-toxicity-detector-development-secret-key-change-this"
)

app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# ============================================================
# DATABASE
# ============================================================

DATABASE_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATABASE_DIR, exist_ok=True)

DATABASE = os.path.join(
    DATABASE_DIR,
    "hate_speech_toxicity_detector.db"
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
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# NORMAL MODEL LABELS
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
# TWITTER LABEL GROUPS
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


# ============================================================
# STARTUP INFORMATION
# ============================================================

print("=" * 70)
print("Hate Speech Toxicity Detector")
print("=" * 70)

print("Base directory :", BASE_DIR)
print("Database       :", DATABASE)
print("Model path     :", MODEL_PATH)
print("Device         :", DEVICE)

print("=" * 70)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


get_db_connection = get_db


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():

    conn = get_db()

    cursor = conn.cursor()


    # --------------------------------------------------------
    # USERS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT NOT NULL UNIQUE,

            email TEXT NOT NULL UNIQUE,

            password TEXT NOT NULL,

            created_at TEXT NOT NULL

        )
    """)


    # --------------------------------------------------------
    # PREDICTION HISTORY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ANALYSIS HISTORY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # YOUTUBE COMMENTS
    # --------------------------------------------------------

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
# NORMAL MODEL
# ============================================================

tokenizer = None
model = None

MODEL_AVAILABLE = False


def load_model():

    global tokenizer
    global model
    global MODEL_AVAILABLE


    if (
        AutoTokenizer is None
        or AutoModelForSequenceClassification is None
    ):

        print(
            "[MODEL] transformers is not installed."
        )

        return


    if not os.path.exists(MODEL_PATH):

        print(
            "[MODEL] Model directory not found:",
            MODEL_PATH
        )

        return


    try:

        print("[MODEL] Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH
        )


        print("[MODEL] Loading model...")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )


        model.to(DEVICE)

        model.eval()

        MODEL_AVAILABLE = True


        print(
            "[MODEL] Model loaded successfully."
        )

        print(
            "[MODEL] Device:",
            DEVICE
        )


        if AutoConfig is not None:

            try:

                config = AutoConfig.from_pretrained(
                    MODEL_PATH
                )

                print(
                    "[MODEL] id2label:",
                    getattr(
                        config,
                        "id2label",
                        None
                    )
                )

            except Exception:

                pass


    except Exception as e:

        MODEL_AVAILABLE = False

        print(
            "[MODEL] ERROR while loading model:",
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

            # API requests should receive JSON, not the login HTML page
            if request.path.startswith("/api/"):
                return jsonify({
                    "success": False,
                    "error": "Please login to access this feature."
                }), 401

            # Normal website pages continue to redirect to login
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


    user = conn.execute("""
        SELECT
            id,
            username,
            email,
            created_at
        FROM users
        WHERE id = ?
    """, (
        session["user_id"],
    )).fetchone()


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

    if (
        not text
        or language == "en"
        or GoogleTranslator is None
    ):

        return text or ""


    try:

        translated = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)


        return (
            translated
            if translated
            else text
        )


    except Exception as e:

        print(
            "[TRANSLATION] Failed:",
            str(e)
        )

        return text


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if text is None:

        return ""


    text = str(text).strip()


    return re.sub(
        r"\s+",
        " ",
        text
    )


# ============================================================
# NORMAL TEXT PREDICTION
# ============================================================

def predict_text(text):

    text = clean_text(text)


    if not text:

        return {

            "prediction": "Non-Toxic",

            "confidence": 0.0,

            "labels": {},

            "language": "en",

            "translated_text": ""

        }


    language = detect_language(text)


    translated_text = translate_to_english(
        text,
        language
    )


    if (
        not MODEL_AVAILABLE
        or tokenizer is None
        or model is None
    ):

        return {

            "prediction": "Model Unavailable",

            "confidence": 0.0,

            "labels": {},

            "language": language,

            "translated_text": translated_text,

            "error":
                "Trained model could not be loaded."

        }


    try:

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

            outputs = model(**inputs)

            probabilities = (
                torch.sigmoid(
                    outputs.logits
                )[0]
                .detach()
                .cpu()
                .numpy()
            )


        # ----------------------------------------------------
        # MODEL LABEL ORDER
        # ----------------------------------------------------

        model_labels = LABELS[:]


        if AutoConfig is not None:

            try:

                cfg = AutoConfig.from_pretrained(
                    MODEL_PATH
                )


                id2label = (
                    getattr(
                        cfg,
                        "id2label",
                        {}
                    )
                    or {}
                )


                configured = []


                for i in range(
                    len(probabilities)
                ):

                    raw = id2label.get(
                        i,
                        id2label.get(
                            str(i),
                            LABELS[i]
                            if i < len(LABELS)
                            else f"label_{i}"
                        )
                    )


                    raw = str(
                        raw
                    ).lower().strip()


                    configured.append(
                        raw
                    )


                if len(configured) == len(
                    probabilities
                ):

                    model_labels = configured


            except Exception:

                pass


        # ----------------------------------------------------
        # LABEL SCORES
        # ----------------------------------------------------

        labels_result = {}


        for index, score in enumerate(
            probabilities
        ):

            if index < len(model_labels):

                labels_result[
                    model_labels[index]
                ] = round(
                    float(score),
                    4
                )


        # ----------------------------------------------------
        # NORMALIZE LABELS
        # ----------------------------------------------------

        normalized = {}


        aliases = {

            "identity-hate":
                "identity_hate",

            "identity hate":
                "identity_hate",

            "severe-toxic":
                "severe_toxic",

            "severe toxic":
                "severe_toxic"

        }


        for label, score in labels_result.items():

            normalized[
                aliases.get(
                    label,
                    label
                )
            ] = score


        labels_result = normalized


        # ----------------------------------------------------
        # FINAL PREDICTION
        # ----------------------------------------------------

        toxic_scores = [

            score

            for score in labels_result.values()

            if score >= 0.5

        ]


        if toxic_scores:

            prediction = "Toxic"

            confidence = max(
                toxic_scores
            )

        else:

            prediction = "Non-Toxic"

            max_score = (
                max(
                    labels_result.values()
                )
                if labels_result
                else 0.0
            )

            confidence = 1.0 - max_score


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
                translated_text

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

            "error":
                str(e)

        }


# ============================================================
# TWITTER / X CLASSIFICATION
# ============================================================

def classify_twitter_result(
    text,
    result
):

    labels = result.get(
        "labels"
    ) or {}


    cleaned = {}


    for label, score in labels.items():

        key = (
            str(label)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )


        try:

            cleaned[key] = float(
                score
            )

        except Exception:

            cleaned[key] = 0.0


    hate_score = max(

        cleaned.get(
            "identity_hate",
            0.0
        ),

        cleaned.get(
            "threat",
            0.0
        ),

        cleaned.get(
            "severe_toxic",
            0.0
        )

    )


    offensive_score = max(

        cleaned.get(
            "toxic",
            0.0
        ),

        cleaned.get(
            "obscene",
            0.0
        ),

        cleaned.get(
            "insult",
            0.0
        )

    )


    if (
        hate_score >= 0.5
        and hate_score >= offensive_score
    ):

        category = "Hate Speech"

        classification_id = "01"

        twitter_confidence = hate_score


    elif offensive_score >= 0.5:

        category = "Offensive Language"

        classification_id = "02"

        twitter_confidence = offensive_score


    else:

        category = "Neither"

        classification_id = "03"

        highest = (
            max(cleaned.values())
            if cleaned
            else 0.0
        )

        twitter_confidence = max(
            0.0,
            1.0 - highest
        )


    hate_detected = (
        hate_score >= 0.5
    )


    offensive_detected = (
        offensive_score >= 0.5
    )


    neither_detected = (
        not hate_detected
        and not offensive_detected
    )


    neither_score = max(
        0.0,
        1.0 - max(
            hate_score,
            offensive_score
        )
    )


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
                    twitter_confidence
                ),
                4
            ),

        "hate_speech":
            hate_detected,

        "offensive_language":
            offensive_detected,

        "neither":
            neither_detected,

        "hate_score":
            round(
                hate_score,
                4
            ),

        "offensive_score":
            round(
                offensive_score,
                4
            ),

        "neither_score":
            round(
                neither_score,
                4
            ),

        "twitter_category_scores": {

            "Hate Speech":
                round(
                    hate_score,
                    4
                ),

            "Offensive Language":
                round(
                    offensive_score,
                    4
                ),

            "Neither":
                round(
                    neither_score,
                    4
                )

        }

    })


    return twitter_result


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


    conn.execute("""
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
    """, (

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

    ))


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


            existing = conn.execute("""
                SELECT id
                FROM users
                WHERE LOWER(username)=LOWER(?)
                   OR LOWER(email)=LOWER(?)
            """, (
                username,
                email
            )).fetchone()


            if existing:

                conn.close()

                flash(
                    "Username or email already exists.",
                    "danger"
                )

                return render_template(
                    "register.html"
                )


            conn.execute("""
                INSERT INTO users
                (
                    username,
                    email,
                    password,
                    created_at
                )
                VALUES (?, ?, ?, ?)
            """, (

                username,

                email,

                generate_password_hash(
                    password
                ),

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            ))


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
                request.form.get(
                    "username",
                    ""
                )
            )
        ).lower()


        password = request.form.get(
            "password",
            ""
        )


        if not login_value or not password:

            flash(
                "Email/username and password are required.",
                "danger"
            )

            return render_template(
                "login.html"
            )


        try:

            conn = get_db()


            user = conn.execute("""
                SELECT *
                FROM users
                WHERE LOWER(email)=LOWER(?)
                   OR LOWER(username)=LOWER(?)
            """, (
                login_value,
                login_value
            )).fetchone()


            conn.close()


            if (
                user is None
                or not check_password_hash(
                    user["password"],
                    password
                )
            ):

                flash(
                    "Invalid username/email or password.",
                    "danger"
                )

                return render_template(
                    "login.html"
                )


            session.clear()

            session["user_id"] = user["id"]

            session["username"] = user["username"]

            session["email"] = user["email"]


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

        user = conn.execute("""
            SELECT
                id,
                username,
                email,
                created_at
            FROM users
            WHERE id=?
        """, (
            user_id,
        )).fetchone()


        # ----------------------------------------------------
        # TOTAL PREDICTIONS
        # ----------------------------------------------------

        total_predictions = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
        """, (
            user_id,
        )).fetchone()[0]


        # ----------------------------------------------------
        # TOXIC COUNT
        # ----------------------------------------------------

        toxic_count = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND LOWER(prediction)='toxic'
        """, (
            user_id,
        )).fetchone()[0]


        # ----------------------------------------------------
        # NON TOXIC COUNT
        # ----------------------------------------------------

        non_toxic_count = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND LOWER(prediction)
                  IN ('non-toxic','non toxic')
        """, (
            user_id,
        )).fetchone()[0]


        # ----------------------------------------------------
        # TODAY COUNT
        # ----------------------------------------------------

        today_count = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND DATE(created_at)
                  = DATE('now','localtime')
        """, (
            user_id,
        )).fetchone()[0]


        # ----------------------------------------------------
        # TOTAL ANALYSES
        # ----------------------------------------------------

        total_analyses = conn.execute("""
            SELECT COUNT(*)
            FROM analysis_history
            WHERE user_id=?
        """, (
            user_id,
        )).fetchone()[0]


        # ----------------------------------------------------
        # YOUTUBE ANALYSES
        # ----------------------------------------------------

        youtube_count = conn.execute("""
            SELECT COUNT(*)
            FROM analysis_history
            WHERE user_id=?
              AND LOWER(analysis_type)='youtube'
        """, (
            user_id,
        )).fetchone()[0]


        # ====================================================
        # TWITTER / X ANALYSES
        # ====================================================
        #
        # IMPORTANT:
        # Twitter/X results are saved in prediction_history
        # using source='twitter'.
        #
        # This count therefore includes ONLY Twitter/X
        # analyses and does not mix them with Text or CSV.
        # ====================================================

        twitter_count = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id = ?
              AND LOWER(
                    TRIM(
                        COALESCE(source, '')
                    )
                  ) = 'twitter'
        """, (
            user_id,
        )).fetchone()[0]


        # Explicit variable for dashboard.html
        twitter_analyses = twitter_count


        # ----------------------------------------------------
        # TOXICITY RATE
        # ----------------------------------------------------

        toxicity_rate = (
            round(
                toxic_count
                / total_predictions
                * 100,
                1
            )
            if total_predictions
            else 0
        )


        # ----------------------------------------------------
        # RECENT HISTORY
        # ----------------------------------------------------

        recent_rows = conn.execute("""
            SELECT
                id,
                input_text,
                prediction,
                confidence,
                source,
                language,
                created_at
            FROM prediction_history
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 10
        """, (
            user_id,
        )).fetchall()


        recent_history = [
            dict(row)
            for row in recent_rows
        ]


        # ----------------------------------------------------
        # ACTIVITY CHART
        # ----------------------------------------------------

        activity_rows = conn.execute("""
            SELECT
                DATE(created_at) AS analysis_date,
                COUNT(*) AS total
            FROM prediction_history
            WHERE user_id=?
            GROUP BY DATE(created_at)
            ORDER BY analysis_date ASC
            LIMIT 30
        """, (
            user_id,
        )).fetchall()


        chart_labels = [
            r["analysis_date"]
            for r in activity_rows
        ]


        chart_values = [
            r["total"]
            for r in activity_rows
        ]


        # ----------------------------------------------------
        # SOURCE DISTRIBUTION
        # ----------------------------------------------------

        source_rows = conn.execute("""
            SELECT
                COALESCE(
                    NULLIF(source,''),
                    'text'
                ) AS source,
                COUNT(*) AS count
            FROM prediction_history
            WHERE user_id=?
            GROUP BY
                COALESCE(
                    NULLIF(source,''),
                    'text'
                )
            ORDER BY count DESC
        """, (
            user_id,
        )).fetchall()


        source_distribution = [

            {
                "source":
                    r["source"].upper(),

                "count":
                    r["count"]
            }

            for r in source_rows

        ]


        if not source_distribution:

            source_distribution = [

                {
                    "source": "TEXT",
                    "count": 0
                }

            ]


        # ----------------------------------------------------
        # LABEL COUNTS
        # ----------------------------------------------------

        label_counts = {
            label: 0
            for label in LABELS
        }


        prediction_rows = conn.execute(
            """
            SELECT labels_json
            FROM prediction_history
            WHERE user_id=?
            """,
            (
                user_id,
            )
        ).fetchall()


        for row in prediction_rows:

            try:

                data = json.loads(
                    row["labels_json"]
                    or "{}"
                )


                for label, score in data.items():

                    if (
                        label in label_counts
                        and float(score) >= 0.5
                    ):

                        label_counts[
                            label
                        ] += 1


            except Exception:

                pass


        # ----------------------------------------------------
        # CATEGORY CHART
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # STATS
        # ----------------------------------------------------

        stats = {

            "prediction_count":
                total_predictions,

            "toxic_count":
                toxic_count,

            "non_toxic_count":
                non_toxic_count,

            "youtube_count":
                youtube_count,

            "twitter_count":
                twitter_count,

            "twitter_analyses":
                twitter_analyses,

            "today_count":
                today_count,

            "total_analyses":
                total_analyses,

            "toxic_percentage":
                toxicity_rate

        }


        conn.close()


        # ----------------------------------------------------
        # DASHBOARD TEMPLATE
        # ----------------------------------------------------

        return render_template(

            "dashboard.html",

            user=user,

            stats=stats,

            total_predictions=
                total_predictions,

            toxic_count=
                toxic_count,

            non_toxic_count=
                non_toxic_count,

            today_count=
                today_count,

            toxicity_rate=
                toxicity_rate,

            total_analyses=
                total_analyses,

            youtube_count=
                youtube_count,

            twitter_count=
                twitter_count,

            # NEW EXPLICIT TWITTER VARIABLE
            twitter_analyses=
                twitter_analyses,

            recent_history=
                recent_history,

            chart_labels=
                chart_labels,

            chart_values=
                chart_values,

            labels=
                LABELS,

            label_display_names=
                LABEL_DISPLAY_NAMES,

            label_counts=
                label_counts,

            category_labels=
                category_labels,

            category_values=
                category_values,

            source_distribution=
                source_distribution,

            model_available=
                MODEL_AVAILABLE

        )


    except Exception as e:

        try:

            conn.close()

        except Exception:

            pass


        print(
            "[DASHBOARD ERROR]",
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

                "twitter_count": 0,

                "twitter_analyses": 0,

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

            twitter_count=0,

            twitter_analyses=0,

            recent_history=[],

            chart_labels=[],

            chart_values=[],

            labels=LABELS,

            label_display_names=
                LABEL_DISPLAY_NAMES,

            label_counts={
                x: 0
                for x in LABELS
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
                "text"
            )


    return render_template(
        "predict.html",
        result=result
    )


# ============================================================
# API TEXT PREDICTION
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

                "error":
                    "Text is required."

            }), 400


        result = predict_text(
            text
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
                "api"
            )


        return jsonify({

            "success": True,

            "result":
                result

        })


    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "success": False,

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


            possible = [

                "text",

                "comment_text",

                "comment",

                "Text",

                "Comment",

                "commentText"

            ]


            text_column = next(

                (
                    c
                    for c in possible
                    if c in df.columns
                ),

                df.columns[0]

            )


            for _, row in df.iterrows():

                text = clean_text(
                    row[text_column]
                )


                if not text:

                    continue


                result = predict_text(
                    text
                )


                results.append({

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

                })


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
# YOUTUBE VIDEO ID
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
            parsed.hostname
            or ""
        ).lower()


        if (
            "youtube.com"
            in hostname
            or
            "youtube-nocookie.com"
            in hostname
        ):

            query = parse_qs(
                parsed.query
            )


            if "v" in query:

                return query["v"][0]


            m = re.search(
                r"/shorts/([A-Za-z0-9_-]{11})",
                parsed.path
            )


            if m:

                return m.group(1)


            m = re.search(
                r"/embed/([A-Za-z0-9_-]{11})",
                parsed.path
            )


            if m:

                return m.group(1)


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
# YOUTUBE SERVICE
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


    return build(

        "youtube",

        "v3",

        developerKey=api_key,

        http=httplib2.Http(
            timeout=60
        ),

        cache_discovery=False

    )


# ============================================================
# YOUTUBE VIDEO INFORMATION
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

    stats = item.get(
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
                stats.get(
                    "viewCount",
                    0
                )
            ),

        "like_count":
            int(
                stats.get(
                    "likeCount",
                    0
                )
            ),

        "comment_count":
            int(
                stats.get(
                    "commentCount",
                    0
                )
            )

    }


# ============================================================
# GET ALL YOUTUBE COMMENTS
# ============================================================

def get_all_youtube_comments(
    youtube,
    video_id
):

    comments = []

    next_page_token = None


    while True:

        response = None


        for attempt in range(3):

            try:

                print(
                    f"[YOUTUBE] Fetching comments page "
                    f"(attempt {attempt+1}/3)..."
                )


                response = youtube.commentThreads().list(

                    part="snippet,replies",

                    videoId=video_id,

                    maxResults=100,

                    pageToken=next_page_token,

                    textFormat="plainText"

                ).execute()


                break


            except Exception as e:

                print(
                    f"[YOUTUBE] Comment request failed: {e}"
                )


                if attempt < 2:

                    time.sleep(2)

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

            ts = item.get(
                "snippet",
                {}
            ).get(
                "topLevelComment",
                {}
            )


            tss = ts.get(
                "snippet",
                {}
            )


            comments.append({

                "comment_id":
                    ts.get("id"),

                "parent_id":
                    None,

                "author":
                    tss.get(
                        "authorDisplayName",
                        "Unknown"
                    ),

                "text":
                    tss.get(
                        "textDisplay",
                        ""
                    ),

                "is_reply":
                    False

            })


            for reply in item.get(
                "replies",
                {}
            ).get(
                "comments",
                []
            ):

                rs = reply.get(
                    "snippet",
                    {}
                )


                comments.append({

                    "comment_id":
                        reply.get("id"),

                    "parent_id":
                        rs.get(
                            "parentId"
                        ),

                    "author":
                        rs.get(
                            "authorDisplayName",
                            "Unknown"
                        ),

                    "text":
                        rs.get(
                            "textDisplay",
                            ""
                        ),

                    "is_reply":
                        True

                })


        next_page_token = response.get(
            "nextPageToken"
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
                    label in category_counts
                    and float(score) >= 0.5
                ):

                    category_counts[
                        label
                    ] += 1


            conn.execute("""
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
            """, (

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

            ))


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


    total = (
        toxic_comments
        + non_toxic_comments
    )


    return {

        "comments":
            analyzed,

        "total_comments":
            total,

        "toxic_comments":
            toxic_comments,

        "non_toxic_comments":
            non_toxic_comments,

        "toxic_percentage":
            round(
                toxic_comments
                / total
                * 100,
                2
            )
            if total
            else 0,

        "non_toxic_percentage":
            round(
                non_toxic_comments
                / total
                * 100,
                2
            )
            if total
            else 0,

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

            youtube = get_youtube_service()


            video = get_video_information(
                youtube,
                video_id
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


            comments = get_all_youtube_comments(
                youtube,
                video_id
            )


            result = analyze_youtube_comments(
                comments,
                session["user_id"],
                video_id
            )


            conn = get_db()


            conn.execute("""
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
            """, (

                session["user_id"],

                "youtube",

                video_id,

                result[
                    "total_comments"
                ],

                result[
                    "toxic_comments"
                ],

                result[
                    "non_toxic_comments"
                ],

                result[
                    "toxic_percentage"
                ],

                json.dumps({

                    "category_counts":
                        result[
                            "category_counts"
                        ]

                }),

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            ))


            conn.commit()

            conn.close()


            flash(
                f"Analyzed {result['total_comments']} "
                f"accessible comments and replies.",
                "success"
            )


        except Exception as e:

            print(
                "[YOUTUBE ERROR]",
                str(e)
            )

            traceback.print_exc()


            flash(
                f"YouTube analysis failed: {str(e)}",
                "danger"
            )


    return render_template(
        "youtube.html",
        result=result,
        video=video
    )


# ============================================================
# YOUTUBE API SUMMARY
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

                "success": False,

                "error":
                    "Invalid YouTube URL."

            }), 400


        youtube = get_youtube_service()


        video = get_video_information(
            youtube,
            video_id
        )


        if video is None:

            return jsonify({

                "success": False,

                "error":
                    "Video not found."

            }), 404


        result = analyze_youtube_comments(

            get_all_youtube_comments(
                youtube,
                video_id
            ),

            session["user_id"],

            video_id

        )


        return jsonify({

            "success":
                True,

            "video":
                video,

            "total_comments":
                result[
                    "total_comments"
                ],

            "toxic_comments":
                result[
                    "toxic_comments"
                ],

            "non_toxic_comments":
                result[
                    "non_toxic_comments"
                ],

            "toxic_percentage":
                result[
                    "toxic_percentage"
                ],

            "non_toxic_percentage":
                result[
                    "non_toxic_percentage"
                ],

            "category_counts":
                result[
                    "category_counts"
                ],

            "comments":
                result[
                    "comments"
                ]

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

            text = clean_text(
                request.form.get(
                    "content",
                    ""
                )
            )


        # ----------------------------------------------------
        # JSON SUPPORT
        # ----------------------------------------------------

        if (
            not text
            and request.is_json
        ):

            try:

                data = request.get_json(
                    silent=True
                ) or {}


                text = clean_text(

                    data.get("text")

                    or data.get("tweet_text")

                    or data.get("tweet")

                    or data.get("content")

                    or ""

                )


            except Exception:

                pass


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

            print("=" * 70)

            print(
                "[TWITTER] Starting Twitter/X analysis"
            )

            print(
                "[TWITTER] Input:",
                repr(text)
            )

            print(
                "[TWITTER] V3 model available:",
                TWITTER_MODEL_AVAILABLE
            )

            print(
                "[TWITTER] Model:",
                "models/twitter_roberta_v3"
            )

            print("=" * 70)


            # ------------------------------------------------
            # MODEL UNAVAILABLE
            # ------------------------------------------------

            if (
                not TWITTER_MODEL_AVAILABLE
                or predict_twitter_text is None
            ):

                result = {

                    "success":
                        False,

                    "text":
                        text,

                    "analyzed_text":
                        text,

                    "prediction":
                        "Model Unavailable",

                    "label":
                        "Model Unavailable",

                    "category":
                        "Model Unavailable",

                    "final_classification":
                        "Model Unavailable",

                    "classification_id":
                        "—",

                    "confidence":
                        0.0,

                    "confidence_percentage":
                        0.0,

                    "twitter_confidence":
                        0.0,

                    "hate_score":
                        0.0,

                    "offensive_score":
                        0.0,

                    "neither_score":
                        0.0,

                    "hate_percentage":
                        0.0,

                    "offensive_percentage":
                        0.0,

                    "neither_percentage":
                        0.0,

                    "error":
                        "Twitter RoBERTa V3 service could not be loaded."

                }


                flash(
                    "Twitter RoBERTa V3 model is unavailable. "
                    "Check the terminal.",
                    "danger"
                )


                return render_template(
                    "twitter.html",
                    result=result
                )


            # ------------------------------------------------
            # CALL REAL TWITTER MODEL
            # ------------------------------------------------

            twitter_result = predict_twitter_text(
                text
            )


            print(
                "[TWITTER] Raw V3 result:",
                twitter_result
            )


            if not twitter_result:

                raise RuntimeError(
                    "Twitter RoBERTa V3 returned no result."
                )


            if not twitter_result.get(
                "success",
                False
            ):

                result = dict(
                    twitter_result
                )


                result.setdefault(
                    "text",
                    text
                )


                result.setdefault(
                    "analyzed_text",
                    text
                )


                flash(

                    twitter_result.get(

                        "error",

                        "Twitter/X analysis "
                        "could not be completed."

                    ),

                    "danger"

                )


                return render_template(
                    "twitter.html",
                    result=result
                )


            # ------------------------------------------------
            # FINAL RESULT
            # ------------------------------------------------

            result = dict(
                twitter_result
            )


            final_class = int(

                result.get(

                    "final_class",

                    result.get(
                        "class_id",
                        2
                    )

                )

            )


            final_label = result.get(

                "final_classification",

                result.get(
                    "prediction",
                    "Neither"
                )

            )


            confidence = float(

                result.get(

                    "confidence",

                    result.get(
                        "twitter_confidence",
                        0.0
                    )

                )

            )


            hate_score = float(
                result.get(
                    "hate_score",
                    0.0
                )
            )


            offensive_score = float(
                result.get(
                    "offensive_score",
                    0.0
                )
            )


            neither_score = float(
                result.get(
                    "neither_score",
                    0.0
                )
            )


            result.update({

                "success":
                    True,

                "text":
                    text,

                "analyzed_text":
                    text,

                "prediction":
                    final_label,

                "label":
                    final_label,

                "category":
                    final_label,

                "final_classification":
                    final_label,

                "final_class":
                    final_class,

                "class_id":
                    final_class,

                "classification_id":
                    result.get(

                        "classification_id",

                        {
                            0: "01",
                            1: "02",
                            2: "03"
                        }.get(
                            final_class,
                            "03"
                        )

                    ),

                "confidence":
                    confidence,

                "confidence_percentage":
                    round(
                        confidence * 100,
                        2
                    ),

                "twitter_confidence":
                    confidence,

                "hate_score":
                    hate_score,

                "offensive_score":
                    offensive_score,

                "neither_score":
                    neither_score,

                "hate_percentage":
                    round(
                        hate_score * 100,
                        2
                    ),

                "offensive_percentage":
                    round(
                        offensive_score * 100,
                        2
                    ),

                "neither_percentage":
                    round(
                        neither_score * 100,
                        2
                    ),

                "model":
                    result.get(
                        "model",
                        "Fine-tuned RoBERTa V3"
                    ),

                "model_name":
                    result.get(
                        "model_name",
                        "Fine-tuned RoBERTa V3"
                    ),

                "source":
                    "Twitter / X"

            })


            # ------------------------------------------------
            # MODEL CONFIDENCE
            # ------------------------------------------------

            model_predicted_class = result.get(
                "model_predicted_class"
            )


            if model_predicted_class is not None:

                model_scores = {

                    0:
                        hate_score,

                    1:
                        offensive_score,

                    2:
                        neither_score

                }


                model_confidence = float(

                    model_scores.get(

                        int(
                            model_predicted_class
                        ),

                        0.0

                    )

                )


                result[
                    "model_confidence"
                ] = model_confidence


                result[
                    "model_confidence_percentage"
                ] = round(

                    model_confidence * 100,

                    2

                )


            # ------------------------------------------------
            # TOXICITY
            # ------------------------------------------------

            toxicity = (

                "Toxic"

                if final_class in [0, 1]

                else "Non-Toxic"

            )


            result[
                "toxicity"
            ] = toxicity


            result[
                "toxicity_status"
            ] = toxicity


            # ------------------------------------------------
            # SAVE TWITTER HISTORY
            # ------------------------------------------------

            history_result = {

                "prediction":
                    final_label,

                "confidence":
                    confidence,

                "labels": {

                    "twitter_hate_speech":
                        hate_score,

                    "twitter_offensive_language":
                        offensive_score,

                    "twitter_neither":
                        neither_score

                },

                "language":
                    "en",

                "translated_text":
                    text

            }


            save_prediction_history(

                session["user_id"],

                text,

                history_result,

                "twitter"

            )


            print(
                "[TWITTER] Saved to prediction_history"
            )

            print(
                "[TWITTER] User ID:",
                session["user_id"]
            )

            print(
                "[TWITTER] Final classification:",
                final_label
            )

            print(
                "[TWITTER] Classification ID:",
                result["classification_id"]
            )

            print(
                "[TWITTER] Confidence:",
                f"{confidence * 100:.2f}%"
            )

            print(
                "[TWITTER] Hate Speech:",
                f"{hate_score * 100:.2f}%"
            )

            print(
                "[TWITTER] Offensive Language:",
                f"{offensive_score * 100:.2f}%"
            )

            print(
                "[TWITTER] Neither:",
                f"{neither_score * 100:.2f}%"
            )

            print("=" * 70)


            flash(
                "Twitter/X text analyzed and saved successfully.",
                "success"
            )


        except Exception as e:

            print(
                "[TWITTER ERROR]",
                str(e)
            )

            traceback.print_exc()


            if (
                result
                and result.get("success")
            ):

                flash(
                    "Twitter/X was analyzed, "
                    "but the history could not be saved.",
                    "warning"
                )


            else:

                result = {

                    "success":
                        False,

                    "text":
                        text,

                    "analyzed_text":
                        text,

                    "prediction":
                        "Error",

                    "label":
                        "Error",

                    "category":
                        "Error",

                    "final_classification":
                        "Error",

                    "classification_id":
                        "—",

                    "confidence":
                        0.0,

                    "confidence_percentage":
                        0.0,

                    "twitter_confidence":
                        0.0,

                    "hate_score":
                        0.0,

                    "offensive_score":
                        0.0,

                    "neither_score":
                        0.0,

                    "hate_percentage":
                        0.0,

                    "offensive_percentage":
                        0.0,

                    "neither_percentage":
                        0.0,

                    "error":
                        str(e)

                }


                flash(
                    "Twitter/X analysis failed. "
                    "Check the terminal error.",
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

    conn = get_db()


    rows = conn.execute("""
        SELECT *
        FROM prediction_history
        WHERE user_id=?
        ORDER BY id DESC
    """, (
        session["user_id"],
    )).fetchall()


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


    conn.execute("""
        DELETE FROM prediction_history
        WHERE id=?
          AND user_id=?
    """, (
        history_id,
        session["user_id"]
    ))


    conn.commit()

    conn.close()


    flash(
        "History item deleted.",
        "success"
    )


    return redirect(
        url_for("history")
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


    conn.execute("""
        DELETE FROM prediction_history
        WHERE user_id=?
    """, (
        session["user_id"],
    ))


    conn.commit()

    conn.close()


    flash(
        "Prediction history cleared.",
        "success"
    )


    return redirect(
        url_for("history")
    )


# ============================================================
# HISTORY API
# ============================================================

@app.route(
    "/api/history",
    methods=["GET"]
)
@login_required
def api_history():

    conn = get_db()


    rows = conn.execute("""
        SELECT *
        FROM prediction_history
        WHERE user_id=?
        ORDER BY id DESC
    """, (
        session["user_id"],
    )).fetchall()


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
# DASHBOARD API STATS
# ============================================================

@app.route(
    "/api/dashboard/stats",
    methods=["GET"]
)
@login_required
def dashboard_stats():

    user_id = session["user_id"]

    conn = get_db()


    try:

        toxic = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND LOWER(prediction)='toxic'
        """, (
            user_id,
        )).fetchone()[0]


        non_toxic = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND LOWER(prediction)
                  IN ('non-toxic','non toxic')
        """, (
            user_id,
        )).fetchone()[0]


        daily = conn.execute("""
            SELECT
                DATE(created_at) AS date,
                COUNT(*) AS count
            FROM prediction_history
            WHERE user_id=?
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """, (
            user_id,
        )).fetchall()


        sources = conn.execute("""
            SELECT
                COALESCE(
                    NULLIF(source,''),
                    'text'
                ) AS source,
                COUNT(*) AS count
            FROM prediction_history
            WHERE user_id=?
            GROUP BY
                COALESCE(
                    NULLIF(source,''),
                    'text'
                )
        """, (
            user_id,
        )).fetchall()


        # ====================================================
        # TWITTER / X COUNT FOR API
        # ====================================================

        twitter_count = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND LOWER(
                    TRIM(
                        COALESCE(source,'')
                    )
                  )='twitter'
        """, (
            user_id,
        )).fetchone()[0]


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
                        r["date"],

                    "count":
                        r["count"]

                }

                for r in daily

            ],

            "source_distribution": [

                {

                    "source":
                        r["source"],

                    "count":
                        r["count"]

                }

                for r in sources

            ],

            # Explicit Twitter/X count
            "twitter_analyses":
                twitter_count

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

    user_id = session["user_id"]

    conn = get_db()


    try:

        user = conn.execute("""
            SELECT
                id,
                username,
                email,
                created_at
            FROM users
            WHERE id=?
        """, (
            user_id,
        )).fetchone()


        total = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
        """, (
            user_id,
        )).fetchone()[0]


        toxic = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND LOWER(prediction)='toxic'
        """, (
            user_id,
        )).fetchone()[0]


        non = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND LOWER(prediction)
                  IN ('non-toxic','non toxic')
        """, (
            user_id,
        )).fetchone()[0]


        rate = (

            round(
                toxic
                / total
                * 100,
                1
            )

            if total

            else 0

        )


        today = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND DATE(created_at)
                  = DATE('now','localtime')
        """, (
            user_id,
        )).fetchone()[0]


        analyses = conn.execute("""
            SELECT COUNT(*)
            FROM analysis_history
            WHERE user_id=?
        """, (
            user_id,
        )).fetchone()[0]


        youtube = conn.execute("""
            SELECT COUNT(*)
            FROM analysis_history
            WHERE user_id=?
              AND LOWER(analysis_type)='youtube'
        """, (
            user_id,
        )).fetchone()[0]


        csv_count = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND LOWER(source)='csv'
        """, (
            user_id,
        )).fetchone()[0]


        text_count = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_history
            WHERE user_id=?
              AND LOWER(source)
                  IN ('text','api')
        """, (
            user_id,
        )).fetchone()[0]


        recent = conn.execute("""
            SELECT
                id,
                input_text,
                prediction,
                confidence,
                source,
                language,
                created_at
            FROM prediction_history
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 10
        """, (
            user_id,
        )).fetchall()


        src = conn.execute("""
            SELECT
                COALESCE(
                    NULLIF(source,''),
                    'text'
                ) AS source,
                COUNT(*) AS count
            FROM prediction_history
            WHERE user_id=?
            GROUP BY
                COALESCE(
                    NULLIF(source,''),
                    'text'
                )
            ORDER BY count DESC
        """, (
            user_id,
        )).fetchall()


        source_distribution = [

            {

                "source":
                    r["source"].upper(),

                "count":
                    r["count"]

            }

            for r in src

        ]


        if not source_distribution:

            source_distribution = [

                {

                    "source":
                        "TEXT",

                    "count":
                        0

                }

            ]


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
                total,

            toxic_predictions=
                toxic,

            non_toxic_predictions=
                non,

            toxicity_rate=
                rate,

            today_count=
                today,

            total_analyses=
                analyses,

            youtube_analyses=
                youtube,

            csv_predictions=
                csv_count,

            text_predictions=
                text_count,

            recent_history=
                recent,

            source_distribution=
                source_distribution,

            created_at=
                created_at,

            model_available=
                MODEL_AVAILABLE

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
            url_for("dashboard")
        )


# ============================================================
# HEALTH
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

@app.route("/api/model-status")
def model_status():

    return jsonify({

        "model_available":
            MODEL_AVAILABLE,

        "model_path":
            MODEL_PATH,

        "device":
            str(DEVICE),

        "labels":
            LABELS

    })


# ============================================================
# 404 ERROR
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
# 413 ERROR
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
        url_for("upload")
    )


# ============================================================
# 500 ERROR
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
# GLOBAL TEMPLATE VARIABLES
# ============================================================

@app.context_processor
def inject_globals():

    return {

        "current_user":
            get_current_user(),

        "model_available":
            MODEL_AVAILABLE,

        "app_name":
            "Hate-Speech-Toxicity-Detector",

    }

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    import os

    print("=" * 70)
    print("Hate Speech Toxicity Detector Server")
    print("=" * 70)

    print("Dashboard : http://127.0.0.1:5000/dashboard")
    print("Login     : http://127.0.0.1:5000/login")
    print("Register  : http://127.0.0.1:5000/register")
    print("Predict   : http://127.0.0.1:5000/predict")
    print("Upload    : http://127.0.0.1:5000/upload")
    print("YouTube   : http://127.0.0.1:5000/youtube")
    print("Twitter   : http://127.0.0.1:5000/twitter")
    print("History   : http://127.0.0.1:5000/history")
    print("Profile   : http://127.0.0.1:5000/profile")
    print("Health    : http://127.0.0.1:5000/health")

    print("=" * 70)

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )