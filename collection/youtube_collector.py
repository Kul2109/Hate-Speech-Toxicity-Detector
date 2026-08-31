"""
Real-time YouTube comment collector.

Fetches top-level comments and replies using the
YouTube Data API.
"""

import os
import re
import pandas as pd

from googleapiclient.discovery import build


# ============================================================
# VIDEO ID EXTRACTION
# ============================================================

def extract_video_id(url_or_id):

    value = (url_or_id or "").strip()

    # Already a video ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    patterns = [
        r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            value
        )

        if match:
            return match.group(1)

    raise ValueError(
        "Invalid YouTube URL or video ID."
    )


# ============================================================
# API CLIENT
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
# FETCH REPLIES
# ============================================================

def fetch_replies(
    youtube,
    parent_comment_id
):

    replies = []
    page_token = None

    while True:

        request = youtube.comments().list(

            part="snippet",

            parentId=parent_comment_id,

            maxResults=100,

            pageToken=page_token,

            textFormat="plainText"

        )

        response = request.execute()

        for item in response.get(
            "items",
            []
        ):

            snippet = item["snippet"]

            replies.append({

                "text":
                    snippet.get(
                        "textDisplay",
                        ""
                    ),

                "source":
                    "youtube_reply"

            })

        page_token = response.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return replies


# ============================================================
# FETCH COMMENTS
# ============================================================

def fetch_youtube_comments(
    video_url,
    include_replies=True
):

    video_id = extract_video_id(
        video_url
    )

    youtube = get_youtube_client()

    comments = []

    page_token = None

    while True:

        request = youtube.commentThreads().list(

            part="snippet",

            videoId=video_id,

            maxResults=100,

            pageToken=page_token,

            textFormat="plainText"

        )

        response = request.execute()

        for item in response.get(
            "items",
            []
        ):

            top_comment = (
                item["snippet"]
                ["topLevelComment"]
                ["snippet"]
            )

            comment_id = (
                item["snippet"]
                ["topLevelComment"]
                ["id"]
            )

            comments.append({

                "text":
                    top_comment.get(
                        "textDisplay",
                        ""
                    ),

                "source":
                    "youtube",

                "video_id":
                    video_id

            })

            # ------------------------------------------------
            # FETCH REPLIES
            # ------------------------------------------------

            if include_replies:

                total_reply_count = item[
                    "snippet"
                ].get(
                    "totalReplyCount",
                    0
                )

                if total_reply_count > 0:

                    try:

                        replies = fetch_replies(
                            youtube,
                            comment_id
                        )

                        for reply in replies:

                            comments.append({

                                "text":
                                    reply["text"],

                                "source":
                                    "youtube_reply",

                                "video_id":
                                    video_id

                            })

                    except Exception as exc:

                        print(
                            "Reply fetch warning:",
                            exc
                        )

        page_token = response.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return pd.DataFrame(
        comments,
        columns=[
            "text",
            "source",
            "video_id"
        ]
    )


# ============================================================
# COLLECT AND SAVE
# ============================================================

def collect(
    video_url,
    output_path="uploads/youtube_comments_live.csv"
):

    df = fetch_youtube_comments(
        video_url
    )

    directory = os.path.dirname(
        output_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Fetched {len(df)} YouTube comments."
    )

    print(
        f"Saved to: {output_path}"
    )

    return df


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        required=True,
        help="YouTube URL"
    )

    parser.add_argument(
        "--output",
        default="uploads/youtube_comments_live.csv"
    )

    args = parser.parse_args()

    collect(
        args.url,
        args.output
    )