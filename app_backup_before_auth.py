from flask import Flask, render_template, jsonify, request
import os
import pandas as pd

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["JSON_SORT_KEYS"] = False


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")


# Create folders if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# POSSIBLE YOUTUBE CSV LOCATIONS
# ============================================================

YOUTUBE_CSV_PATHS = [

    # Main location used in your project
    os.path.join(
        RAW_DATA_DIR,
        "youtube_comments.csv"
    ),

    # Alternative location
    os.path.join(
        DATA_DIR,
        "youtube_comments.csv"
    ),

    # Alternative uploads location
    os.path.join(
        UPLOAD_DIR,
        "youtube_comments.csv"
    ),

]


# ============================================================
# FIND YOUTUBE CSV
# ============================================================

def find_youtube_csv():

    for path in YOUTUBE_CSV_PATHS:

        if os.path.exists(path):
            return path

    return None


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(df, possible_names):

    columns_lower = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:

        if name.lower() in columns_lower:
            return columns_lower[name.lower()]

    return None


# ============================================================
# CONVERT VALUE TO INTEGER
# ============================================================

def safe_int(value):

    try:
        return int(value)

    except (ValueError, TypeError):
        return 0


# ============================================================
# CONVERT VALUE TO FLOAT
# ============================================================

def safe_float(value):

    try:
        return float(value)

    except (ValueError, TypeError):
        return 0.0


# ============================================================
# FIND PREDICTION COLUMN
# ============================================================

def find_prediction_column(df):

    possible_columns = [

        "Prediction",
        "prediction",

        "Predicted",
        "predicted",

        "label",
        "Label",

        "toxicity",
        "Toxicity",

        "result",
        "Result",

        "classification",
        "Classification"

    ]

    return find_column(
        df,
        possible_columns
    )


# ============================================================
# FIND CATEGORY COLUMNS
# ============================================================

def find_category_column(df, category):

    category_variations = {

        "toxic": [
            "toxic",
            "toxicity"
        ],

        "obscene": [
            "obscene",
            "obscenity"
        ],

        "insult": [
            "insult"
        ],

        "threat": [
            "threat"
        ],

        "identity_hate": [
            "identity_hate",
            "identity hate",
            "identityhate"
        ],

        "severe_toxic": [
            "severe_toxic",
            "severe toxic",
            "severetoxic"
        ]

    }

    names = category_variations.get(
        category,
        []
    )

    return find_column(
        df,
        names
    )


# ============================================================
# VALUE IS POSITIVE
# ============================================================

def is_positive(value):

    if pd.isna(value):
        return False

    # Numeric values
    if isinstance(value, (int, float)):

        return value >= 0.5

    text = str(value).strip().lower()

    positive_values = {

        "toxic",
        "yes",
        "true",
        "1",
        "positive",
        "hate",
        "insult",
        "obscene",
        "threat"

    }

    return text in positive_values


# ============================================================
# CALCULATE CATEGORY COUNT
# ============================================================

def calculate_category_count(
    df,
    category
):

    column = find_category_column(
        df,
        category
    )

    if column is None:

        return 0

    return int(
        df[column]
        .apply(is_positive)
        .sum()
    )


# ============================================================
# CALCULATE YOUTUBE SUMMARY
# ============================================================

def calculate_youtube_summary():

    csv_path = find_youtube_csv()

    # --------------------------------------------------------
    # If CSV is not found
    # --------------------------------------------------------

    if csv_path is None:

        return {

            "source": "youtube",

            "total_comments": 0,

            "toxic_comments": 0,

            "non_toxic_comments": 0,

            "toxic_percentage": 0,

            "non_toxic_percentage": 0,

            "category_counts": {

                "toxic": 0,

                "obscene": 0,

                "insult": 0,

                "threat": 0,

                "identity_hate": 0,

                "severe_toxic": 0

            },

            "error":
                "YouTube CSV file was not found."

        }


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            csv_path,
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        df = pd.read_csv(
            csv_path,
            encoding="latin1"
        )

    except Exception as error:

        return {

            "source": "youtube",

            "total_comments": 0,

            "toxic_comments": 0,

            "non_toxic_comments": 0,

            "toxic_percentage": 0,

            "non_toxic_percentage": 0,

            "category_counts": {

                "toxic": 0,

                "obscene": 0,

                "insult": 0,

                "threat": 0,

                "identity_hate": 0,

                "severe_toxic": 0

            },

            "error": str(error)

        }


    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    df = df.dropna(
        how="all"
    )

    total_comments = len(df)


    # --------------------------------------------------------
    # Find prediction column
    # --------------------------------------------------------

    prediction_column = find_prediction_column(
        df
    )


    toxic_comments = 0


    # --------------------------------------------------------
    # Calculate toxic comments
    # --------------------------------------------------------

    if prediction_column is not None:

        toxic_comments = int(
            df[prediction_column]
            .apply(is_positive)
            .sum()
        )

    else:

        # ----------------------------------------------------
        # If no prediction column exists,
        # check the toxic category column
        # ----------------------------------------------------

        toxic_column = find_category_column(
            df,
            "toxic"
        )

        if toxic_column is not None:

            toxic_comments = int(
                df[toxic_column]
                .apply(is_positive)
                .sum()
            )


    # --------------------------------------------------------
    # Non-toxic comments
    # --------------------------------------------------------

    non_toxic_comments = (
        total_comments -
        toxic_comments
    )

    if non_toxic_comments < 0:
        non_toxic_comments = 0


    # --------------------------------------------------------
    # Percentages
    # --------------------------------------------------------

    if total_comments > 0:

        toxic_percentage = round(
            (toxic_comments / total_comments) * 100,
            1
        )

        non_toxic_percentage = round(
            (non_toxic_comments / total_comments) * 100,
            1
        )

    else:

        toxic_percentage = 0

        non_toxic_percentage = 0


    # --------------------------------------------------------
    # Category counts
    # --------------------------------------------------------

    category_counts = {

        "toxic": toxic_comments,

        "obscene":
            calculate_category_count(
                df,
                "obscene"
            ),

        "insult":
            calculate_category_count(
                df,
                "insult"
            ),

        "threat":
            calculate_category_count(
                df,
                "threat"
            ),

        "identity_hate":
            calculate_category_count(
                df,
                "identity_hate"
            ),

        "severe_toxic":
            calculate_category_count(
                df,
                "severe_toxic"
            )

    }


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "source": "youtube",

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
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# YOUTUBE SUMMARY API
# ============================================================

@app.route(
    "/api/youtube/summary",
    methods=["GET"]
)
def youtube_summary():

    summary = calculate_youtube_summary()

    return jsonify(summary)


# ============================================================
# YOUTUBE STATUS API
# ============================================================

@app.route(
    "/api/youtube/status",
    methods=["GET"]
)
def youtube_status():

    csv_path = find_youtube_csv()

    if csv_path is None:

        return jsonify({

            "status": "not_found",

            "source": "youtube",

            "message":
                "YouTube comments CSV was not found."

        })


    return jsonify({

        "status": "ready",

        "source": "youtube",

        "file": os.path.basename(
            csv_path
        ),

        "path": csv_path

    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "ok",

        "application":
            "Multilingual Hate Speech and Toxicity Detector",

        "backend":
            "Flask",

        "youtube_api":
            "/api/youtube/summary"

    })


# ============================================================
# DEBUG INFORMATION
# ============================================================

@app.route(
    "/api/debug/youtube",
    methods=["GET"]
)
def debug_youtube():

    csv_path = find_youtube_csv()

    if csv_path is None:

        return jsonify({

            "found": False,

            "searched_paths":
                YOUTUBE_CSV_PATHS

        })


    try:

        df = pd.read_csv(
            csv_path,
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        df = pd.read_csv(
            csv_path,
            encoding="latin1"
        )


    return jsonify({

        "found": True,

        "file":
            csv_path,

        "rows":
            len(df),

        "columns":
            list(df.columns),

        "first_rows":
            df.head(5)
            .fillna("")
            .to_dict(
                orient="records"
            )

    })


# ============================================================
# 404 ERROR
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({

        "error": "Endpoint not found",

        "message":
            "The requested URL does not exist."

    }), 404


# ============================================================
# 500 ERROR
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({

        "error": "Internal server error",

        "message":
            "Something went wrong on the Flask server."

    }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("  MULTILINGUAL HATE SPEECH & TOXICITY DETECTOR")
    print("=" * 60)
    print()

    print("Backend: Flask")

    youtube_csv = find_youtube_csv()

    if youtube_csv:

        print(
            "YouTube CSV:",
            youtube_csv
        )

    else:

        print(
            "WARNING: YouTube CSV not found."
        )

    print()

    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:5000/"
    )

    print()

    print(
        "YouTube Summary API:"
    )

    print(
        "http://127.0.0.1:5000/api/youtube/summary"
    )

    print()

    print("=" * 60)
    print()


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )