from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect,
    url_for,
    session,
    flash
)

import os
import re
import sqlite3
import pandas as pd

from functools import wraps
from urllib.parse import urlparse, parse_qs

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from dotenv import load_dotenv
from googleapiclient.discovery import build


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

print("==============================================")
print("FLASK APP FILE:", os.path.abspath(__file__))
print("TEMPLATE FOLDER:", app.template_folder)
print(
    "INDEX FILE:",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "templates",
        "dashboard.html"
    )
)
print("==============================================")


app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "toxiguard-final-year-project-secret-key"
)

app.config["JSON_SORT_KEYS"] = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RAW_DATA_DIR = os.path.join(
    DATA_DIR,
    "raw"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "users.db"
)

YOUTUBE_OUTPUT = os.path.join(
    UPLOAD_DIR,
    "youtube_live_analysis.csv"
)


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def get_db():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            username TEXT UNIQUE NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()

    print("Database initialized:", DATABASE_PATH)


# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            return redirect(
                url_for("login")
            )

        return function(
            *args,
            **kwargs
        )

    return decorated_function


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        if not name:

            flash(
                "Please enter your full name.",
                "error"
            )

            return render_template(
                "register.html"
            )


        if len(username) < 3:

            flash(
                "Username must contain at least 3 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )


        if not email:

            flash(
                "Please enter your email.",
                "error"
            )

            return render_template(
                "register.html"
            )


        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )


        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return render_template(
                "register.html"
            )


        connection = get_db()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE username = ?
            OR email = ?
            """,
            (
                username,
                email
            )
        )

        existing_user = cursor.fetchone()


        if existing_user:

            connection.close()

            flash(
                "Username or email already exists.",
                "error"
            )

            return render_template(
                "register.html"
            )


        hashed_password = generate_password_hash(
            password
        )


        cursor.execute(
            """
            INSERT INTO users
            (
                name,
                username,
                email,
                password
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                username,
                email,
                hashed_password
            )
        )


        connection.commit()

        connection.close()


        flash(
            "Account created successfully. Please login.",
            "success"
        )


        return redirect(
            url_for("login")
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

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        if not username or not password:

            flash(
                "Please enter username and password.",
                "error"
            )

            return render_template(
                "login.html"
            )


        connection = get_db()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE username = ?
            """,
            (username,)
        )

        user = cursor.fetchone()

        connection.close()


        if user is None:

            flash(
                "Invalid username or password.",
                "error"
            )

            return render_template(
                "login.html"
            )


        try:

            valid_password = check_password_hash(
                user["password"],
                password
            )

        except Exception:

            valid_password = False


        if not valid_password:

            flash(
                "Invalid username or password.",
                "error"
            )

            return render_template(
                "login.html"
            )


        session.clear()

        session["user_id"] = user["id"]

        session["username"] = user["username"]

        session["name"] = user["name"]

        session["email"] = user["email"]


        return redirect(
            url_for("dashboard")
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

    return redirect(
        url_for("login")
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.route("/")
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        username=session.get(
            "username",
            "User"
        ),
        name=session.get(
            "name",
            ""
        ),
        email=session.get(
            "email",
            ""
        )
    )


# ============================================================
# YOUTUBE PAGE
# ============================================================

@app.route("/youtube")
@login_required
def youtube_page():

    return render_template(
        "youtube.html",
        username=session.get(
            "username",
            "User"
        )
    )


# ============================================================
# ALL COMMENTS PAGE
# ============================================================

@app.route("/comments")
@login_required
def comments_page():

    return render_template(
        "comments.html",
        username=session.get(
            "username",
            "User"
        )
    )


# ============================================================
# SINGLE TEXT PAGE
# ============================================================

@app.route("/text-detection")
@login_required
def text_detection_page():

    return render_template(
        "text_detection.html",
        username=session.get(
            "username",
            "User"
        )
    )


# ============================================================
# CATEGORIES PAGE
# ============================================================

@app.route("/categories")
@login_required
def categories_page():

    return render_template(
        "categories.html",
        username=session.get(
            "username",
            "User"
        )
    )


# ============================================================
# EXTRACT YOUTUBE VIDEO ID
# ============================================================

def extract_video_id(url):

    if not url:
        return None

    url = url.strip()

    parsed = urlparse(url)

    query = parse_qs(
        parsed.query
    )


    if "v" in query:

        video_id = query["v"][0]

        if re.match(
            r"^[A-Za-z0-9_-]{11}$",
            video_id
        ):

            return video_id


    if "youtu.be" in parsed.netloc:

        video_id = parsed.path.strip(
            "/"
        ).split("/")[0]

        if re.match(
            r"^[A-Za-z0-9_-]{11}$",
            video_id
        ):

            return video_id


    match = re.search(
        r"/shorts/([A-Za-z0-9_-]{11})",
        url
    )

    if match:

        return match.group(1)


    match = re.search(
        r"/embed/([A-Za-z0-9_-]{11})",
        url
    )

    if match:

        return match.group(1)


    return None


# ============================================================
# YOUTUBE CLIENT
# ============================================================

def get_youtube_client():

    api_key = os.getenv(
        "YOUTUBE_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "YOUTUBE_API_KEY is not configured."
        )

    return build(
        "youtube",
        "v3",
        developerKey=api_key
    )


# ============================================================
# VIDEO INFORMATION
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

        raise RuntimeError(
            "YouTube video was not found or is unavailable."
        )


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
            )
    }


# ============================================================
# COLLECT YOUTUBE COMMENTS
# ============================================================

def collect_all_youtube_comments(
    youtube,
    video_id
):

    comments = []

    next_page_token = None


    while True:

        response = youtube.commentThreads().list(

            part="snippet,replies",

            videoId=video_id,

            maxResults=100,

            pageToken=next_page_token,

            textFormat="plainText"

        ).execute()


        for item in response.get(
            "items",
            []
        ):

            snippet = (
                item
                .get("snippet", {})
                .get("topLevelComment", {})
                .get("snippet", {})
            )


            text = snippet.get(
                "textDisplay",
                ""
            ).strip()


            if text:

                comments.append({

                    "text":
                        text,

                    "source":
                        "youtube",

                    "video_id":
                        video_id,

                    "comment_type":
                        "top_level"
                })


            replies = (
                item
                .get("replies", {})
                .get("comments", [])
            )


            for reply in replies:

                reply_snippet = reply.get(
                    "snippet",
                    {}
                )


                reply_text = reply_snippet.get(
                    "textDisplay",
                    ""
                ).strip()


                if reply_text:

                    comments.append({

                        "text":
                            reply_text,

                        "source":
                            "youtube",

                        "video_id":
                            video_id,

                        "comment_type":
                            "reply"
                    })


        next_page_token = response.get(
            "nextPageToken"
        )


        if not next_page_token:

            break


    return comments


# ============================================================
# MODEL LOADER
# ============================================================

_predictor = None


def get_predictor():

    global _predictor


    if _predictor is not None:

        return _predictor


    try:

        from backend.services.inference import (
            HateSpeechPredictor
        )


        _predictor = HateSpeechPredictor()

        return _predictor


    except Exception as exc:

        print(
            "MODEL LOADING ERROR:",
            repr(exc)
        )

        raise RuntimeError(
            "ML model could not be loaded: "
            + str(exc)
        )


# ============================================================
# PREDICT ONE COMMENT
# ============================================================

def predict_comment(
    predictor,
    text
):

    result = predictor.predict(
        text
    )


    labels = result.get(
        "labels",
        {}
    )


    toxic_score = float(
        labels
        .get("toxic", {})
        .get("score", 0)
    )


    toxic_confidence = round(
        toxic_score * 100,
        2
    )


    if result.get(
        "overall_toxic",
        False
    ):

        prediction = "Toxic"

        confidence = toxic_confidence

    else:

        prediction = "Non-Toxic"

        confidence = round(
            (1 - toxic_score) * 100,
            2
        )


    detected_categories = []


    for category, data in labels.items():

        if data.get(
            "flagged",
            False
        ):

            detected_categories.append(
                category
            )


    return {

        "text":
            text,

        "prediction":
            prediction,

        "confidence":
            confidence,

        "toxic_score":
            toxic_confidence,

        "categories":
            detected_categories,

        "labels":
            labels
    }


# ============================================================
# ANALYZE COMMENTS
# ============================================================

def analyze_comments(comments):

    predictor = get_predictor()


    results = []

    toxic_count = 0

    non_toxic_count = 0


    category_counts = {

        "toxic": 0,

        "severe_toxic": 0,

        "obscene": 0,

        "threat": 0,

        "insult": 0,

        "identity_hate": 0
    }


    for index, comment in enumerate(
        comments,
        start=1
    ):

        text = comment.get(
            "text",
            ""
        ).strip()


        if not text:

            continue


        try:

            prediction = predict_comment(
                predictor,
                text
            )


            if prediction["prediction"] == "Toxic":

                toxic_count += 1

            else:

                non_toxic_count += 1


            for category in prediction["categories"]:

                if category in category_counts:

                    category_counts[
                        category
                    ] += 1


            results.append({

                "text":
                    text,

                "prediction":
                    prediction[
                        "prediction"
                    ],

                "confidence":
                    prediction[
                        "confidence"
                    ],

                "toxic_score":
                    prediction[
                        "toxic_score"
                    ],

                "categories":
                    ", ".join(
                        prediction[
                            "categories"
                        ]
                    ),

                "source":
                    comment.get(
                        "source",
                        "youtube"
                    ),

                "video_id":
                    comment.get(
                        "video_id",
                        ""
                    ),

                "comment_type":
                    comment.get(
                        "comment_type",
                        ""
                    )
            })


        except Exception as exc:

            print(
                f"Prediction failed for comment {index}:",
                exc
            )


    analyzed_total = (
        toxic_count +
        non_toxic_count
    )


    if analyzed_total > 0:

        toxic_percentage = round(
            toxic_count /
            analyzed_total *
            100,
            2
        )

        non_toxic_percentage = round(
            non_toxic_count /
            analyzed_total *
            100,
            2
        )

    else:

        toxic_percentage = 0

        non_toxic_percentage = 0


    return {

        "total_comments":
            analyzed_total,

        "toxic_comments":
            toxic_count,

        "non_toxic_comments":
            non_toxic_count,

        "toxic_percentage":
            toxic_percentage,

        "non_toxic_percentage":
            non_toxic_percentage,

        "category_counts":
            category_counts,

        "comments":
            results
    }


# ============================================================
# YOUTUBE ANALYSIS API
# ============================================================

@app.route(
    "/api/youtube/analyze",
    methods=["POST"]
)
@login_required
def youtube_analyze():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        youtube_url = data.get(
            "url",
            ""
        ).strip()


        if not youtube_url:

            return jsonify({

                "success":
                    False,

                "error":
                    "Please provide a YouTube video URL."
            }), 400


        video_id = extract_video_id(
            youtube_url
        )


        if not video_id:

            return jsonify({

                "success":
                    False,

                "error":
                    "Invalid YouTube URL."
            }), 400


        youtube = get_youtube_client()


        video_info = get_video_information(
            youtube,
            video_id
        )


        comments = collect_all_youtube_comments(
            youtube,
            video_id
        )


        if not comments:

            return jsonify({

                "success":
                    False,

                "error":
                    "No comments were found for this video.",

                "video":
                    video_info
            }), 404


        analysis = analyze_comments(
            comments
        )


        # ----------------------------------------------------
        # SAVE ALL COMMENTS
        # ----------------------------------------------------

        pd.DataFrame(
            analysis["comments"]
        ).to_csv(
            YOUTUBE_OUTPUT,
            index=False,
            encoding="utf-8-sig"
        )


        # ----------------------------------------------------
        # STORE VIDEO INFORMATION IN SESSION
        # ----------------------------------------------------

        session["youtube_video"] = video_info

        session["youtube_url"] = youtube_url


        return jsonify({

            "success":
                True,

            "source":
                "youtube",

            "video":
                video_info,

            "youtube_url":
                youtube_url,

            "video_id":
                video_id,

            "total_comments":
                analysis[
                    "total_comments"
                ],

            "toxic_comments":
                analysis[
                    "toxic_comments"
                ],

            "non_toxic_comments":
                analysis[
                    "non_toxic_comments"
                ],

            "toxic_percentage":
                analysis[
                    "toxic_percentage"
                ],

            "non_toxic_percentage":
                analysis[
                    "non_toxic_percentage"
                ],

            "category_counts":
                analysis[
                    "category_counts"
                ]
        })


    except Exception as exc:

        print(
            "YOUTUBE ANALYSIS ERROR:",
            repr(exc)
        )


        return jsonify({

            "success":
                False,

            "error":
                str(exc)
        }), 500


# ============================================================
# YOUTUBE SUMMARY API
# ============================================================

@app.route(
    "/api/youtube/summary",
    methods=["GET"]
)
@login_required
def youtube_summary():

    if not os.path.exists(
        YOUTUBE_OUTPUT
    ):

        return jsonify({

            "success":
                False,

            "error":
                "No YouTube analysis has been performed yet."
        }), 404


    try:

        df = pd.read_csv(
            YOUTUBE_OUTPUT
        )


        total = len(df)


        toxic = int(
            (
                df["prediction"]
                .astype(str)
                .str.lower()
                == "toxic"
            ).sum()
        )


        non_toxic = total - toxic


        toxic_percentage = (
            round(
                toxic /
                total *
                100,
                2
            )
            if total > 0
            else 0
        )


        non_toxic_percentage = (
            round(
                non_toxic /
                total *
                100,
                2
            )
            if total > 0
            else 0
        )


        category_counts = {

            "toxic": 0,

            "severe_toxic": 0,

            "obscene": 0,

            "threat": 0,

            "insult": 0,

            "identity_hate": 0
        }


        if "categories" in df.columns:

            for categories in df[
                "categories"
            ].fillna(""):

                category_list = [

                    item.strip()

                    for item in str(
                        categories
                    ).split(",")

                    if item.strip()
                ]


                for category in category_list:

                    if category in category_counts:

                        category_counts[
                            category
                        ] += 1


        return jsonify({

            "success":
                True,

            "source":
                "youtube",

            "video":
                session.get(
                    "youtube_video",
                    {}
                ),

            "youtube_url":
                session.get(
                    "youtube_url",
                    ""
                ),

            "total_comments":
                total,

            "toxic_comments":
                toxic,

            "non_toxic_comments":
                non_toxic,

            "toxic_percentage":
                toxic_percentage,

            "non_toxic_percentage":
                non_toxic_percentage,

            "category_counts":
                category_counts
        })


    except Exception as exc:

        return jsonify({

            "success":
                False,

            "error":
                str(exc)
        }), 500


# ============================================================
# ALL COMMENTS API
# ============================================================

@app.route(
    "/api/youtube/comments",
    methods=["GET"]
)
@login_required
def youtube_comments():

    if not os.path.exists(
        YOUTUBE_OUTPUT
    ):

        return jsonify({

            "success":
                False,

            "error":
                "No YouTube analysis has been performed yet."
        }), 404


    try:

        df = pd.read_csv(
            YOUTUBE_OUTPUT
        )

        df = df.fillna("")


        comments = []


        for index, row in df.iterrows():

            comments.append({

                "id":
                    index + 1,

                "text":
                    str(
                        row.get(
                            "text",
                            ""
                        )
                    ),

                "prediction":
                    str(
                        row.get(
                            "prediction",
                            ""
                        )
                    ),

                "confidence":
                    float(
                        row.get(
                            "confidence",
                            0
                        )
                    ),

                "toxic_score":
                    float(
                        row.get(
                            "toxic_score",
                            0
                        )
                    ),

                "categories":
                    str(
                        row.get(
                            "categories",
                            ""
                        )
                    ),

                "comment_type":
                    str(
                        row.get(
                            "comment_type",
                            ""
                        )
                    )
            })


        return jsonify({

            "success":
                True,

            "total":
                len(comments),

            "comments":
                comments
        })


    except Exception as exc:

        return jsonify({

            "success":
                False,

            "error":
                str(exc)
        }), 500


# ============================================================
# SINGLE TEXT PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
@login_required
def predict_text():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        text = data.get(
            "text",
            ""
        ).strip()


        if not text:

            return jsonify({

                "success":
                    False,

                "error":
                    "Text is required."
            }), 400


        predictor = get_predictor()


        result = predict_comment(
            predictor,
            text
        )


        return jsonify({

            "success":
                True,

            "text":
                text,

            "prediction":
                result[
                    "prediction"
                ],

            "confidence":
                result[
                    "confidence"
                ],

            "toxic_score":
                result[
                    "toxic_score"
                ],

            "categories":
                result[
                    "categories"
                ],

            "labels":
                result[
                    "labels"
                ]
        })


    except Exception as exc:

        print(
            "PREDICTION ERROR:",
            repr(exc)
        )


        return jsonify({

            "success":
                False,

            "error":
                str(exc)
        }), 500


# ============================================================
# CATEGORIES API
# ============================================================

@app.route(
    "/api/categories",
    methods=["GET"]
)
@login_required
def categories_api():

    if not os.path.exists(
        YOUTUBE_OUTPUT
    ):

        return jsonify({

            "success":
                False,

            "error":
                "No YouTube analysis has been performed yet."
        }), 404


    try:

        df = pd.read_csv(
            YOUTUBE_OUTPUT
        )

        total = len(df)


        category_counts = {

            "toxic": 0,

            "severe_toxic": 0,

            "obscene": 0,

            "threat": 0,

            "insult": 0,

            "identity_hate": 0
        }


        if "categories" in df.columns:

            for categories in df[
                "categories"
            ].fillna(""):

                for category in str(
                    categories
                ).split(","):

                    category = category.strip()

                    if category in category_counts:

                        category_counts[
                            category
                        ] += 1


        return jsonify({

            "success":
                True,

            "total":
                total,

            "categories":
                category_counts
        })


    except Exception as exc:

        return jsonify({

            "success":
                False,

            "error":
                str(exc)
        }), 500


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "application":
            "Multilingual Hate Speech and Toxicity Detector",

        "backend":
            "Flask",

        "model":
            "Fine-tuned DistilBERT",

        "youtube_api":
            "configured"
            if os.getenv(
                "YOUTUBE_API_KEY"
            )
            else "not_configured"
    })


# ============================================================
# INITIALIZE DATABASE
# ============================================================

init_database()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "============================================================"
    )

    print(
        "TOXIGUARD - MULTI PAGE APPLICATION"
    )

    print(
        "============================================================"
    )

    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:5000/"
    )

    print(
        "YouTube:"
    )

    print(
        "http://127.0.0.1:5000/youtube"
    )

    print(
        "All Comments:"
    )

    print(
        "http://127.0.0.1:5000/comments"
    )

    print(
        "Text Detection:"
    )

    print(
        "http://127.0.0.1:5000/text-detection"
    )

    print(
        "Categories:"
    )

    print(
        "http://127.0.0.1:5000/categories"
    )

    print(
        "============================================================"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )