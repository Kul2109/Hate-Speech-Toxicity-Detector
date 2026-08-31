import os
import re
import html
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# HATE SPEECH TOXICITY DETECTOR - TWITTER DATASET PREPROCESSING
# ============================================================

DATASET_PATH = "datasets/twitter/labeled_data.csv"

TRAIN_PATH = "datasets/twitter/train.csv"
VAL_PATH = "datasets/twitter/validation.csv"
TEST_PATH = "datasets/twitter/test.csv"


# ============================================================
# CLEAN TWEET
# ============================================================

def clean_tweet(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # Decode HTML entities
    # Example:
    # &#128514; -> 😂
    # &amp;      -> &
    # &#8220;    -> "
    # --------------------------------------------------------

    text = html.unescape(text)

    # Decode HTML entities more than once if necessary
    text = html.unescape(text)

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
    # Remove RT prefixes
    #
    # Handles:
    # RT @username:
    # RT @username
    # RT: @username
    # RT : @username
    # RT :
    # --------------------------------------------------------

    text = re.sub(
        r"\bRT\s*:?\s*@?\w*\s*:?\s*",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Remove any remaining Twitter mentions
    # --------------------------------------------------------

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove any standalone RT that may remain
    #
    # This handles cases such as:
    # RT :
    # RT
    # RT: text
    # --------------------------------------------------------

    text = re.sub(
        r"\bRT\b\s*:?\s*",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Remove escaped newline / tab characters
    # --------------------------------------------------------

    text = text.replace("\\n", " ")
    text = text.replace("\\r", " ")
    text = text.replace("\\t", " ")

    # --------------------------------------------------------
    # Remove actual newline / carriage return / tab
    # --------------------------------------------------------

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

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
# LOAD DATASET
# ============================================================

def load_twitter_dataset():

    print("\nLoading original Twitter dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(
        "Original dataset size:",
        len(df)
    )

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    df = df[
        ["class", "tweet"]
    ].copy()

    # --------------------------------------------------------
    # Remove missing tweets
    # --------------------------------------------------------

    df = df.dropna(
        subset=["tweet"]
    )

    # --------------------------------------------------------
    # Clean tweets
    # --------------------------------------------------------

    print("Cleaning tweets...")

    df["tweet"] = df["tweet"].apply(
        clean_tweet
    )

    # --------------------------------------------------------
    # Remove empty tweets
    # --------------------------------------------------------

    df = df[
        df["tweet"].str.strip().str.len() > 0
    ]

    # --------------------------------------------------------
    # Convert labels to integer
    # --------------------------------------------------------

    df["class"] = df["class"].astype(int)

    # --------------------------------------------------------
    # Remove duplicate tweets BEFORE splitting
    #
    # This prevents the same tweet from appearing in
    # train / validation / test.
    # --------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["tweet"],
        keep="first"
    ).reset_index(drop=True)

    removed_duplicates = (
        before_duplicates - len(df)
    )

    print(
        "Duplicate tweets removed:",
        removed_duplicates
    )

    return df


# ============================================================
# CREATE TRAIN / VALIDATION / TEST SPLITS
# ============================================================

def create_splits(df):

    # --------------------------------------------------------
    # First split:
    # 80% training
    # 20% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(

        df,

        test_size=0.20,

        random_state=42,

        stratify=df["class"]
    )

    # --------------------------------------------------------
    # Second split:
    # Temporary -> 50% validation + 50% test
    #
    # Result:
    # 80% train
    # 10% validation
    # 10% test
    # --------------------------------------------------------

    val_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        random_state=42,

        stratify=temp_df["class"]
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "Hate Speech Toxicity Detector - TWITTER DATASET PREPROCESSING"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Load and clean
    # --------------------------------------------------------

    df = load_twitter_dataset()

    print("\nDataset after preprocessing:")

    print(
        "Total unique tweets:",
        len(df)
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    print(
        df["class"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    train_df, val_df, test_df = create_splits(df)

    print("\nDataset split:")

    print(
        "Training   :",
        len(train_df)
    )

    print(
        "Validation :",
        len(val_df)
    )

    print(
        "Testing    :",
        len(test_df)
    )

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    os.makedirs(
        "datasets/twitter",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    val_df.to_csv(
        VAL_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print("\nSaved files:")

    print(
        " ",
        TRAIN_PATH
    )

    print(
        " ",
        VAL_PATH
    )

    print(
        " ",
        TEST_PATH
    )

    # --------------------------------------------------------
    # Verify cleaning
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CLEANING VERIFICATION")
    print("=" * 60)

    combined = pd.concat(
        [
            train_df,
            val_df,
            test_df
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # URL check
    # --------------------------------------------------------

    url_count = combined["tweet"].str.contains(
        r"https?://\S+|www\.\S+",
        regex=True,
        case=False,
        na=False
    ).sum()

    # --------------------------------------------------------
    # Mention check
    # --------------------------------------------------------

    mention_count = combined["tweet"].str.contains(
        r"@\w+",
        regex=True,
        na=False
    ).sum()

    # --------------------------------------------------------
    # RT check
    # --------------------------------------------------------

    rt_count = combined["tweet"].str.contains(
        r"\bRT\b",
        regex=True,
        case=False,
        na=False
    ).sum()

    # --------------------------------------------------------
    # Empty check
    # --------------------------------------------------------

    empty_count = combined["tweet"].str.strip().eq("").sum()

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicate_count = combined["tweet"].duplicated().sum()

    print(
        "URLs remaining     :",
        url_count
    )

    print(
        "Mentions remaining :",
        mention_count
    )

    print(
        "RT remaining       :",
        rt_count
    )

    print(
        "Empty tweets       :",
        empty_count
    )

    print(
        "Duplicate tweets   :",
        duplicate_count
    )

    # --------------------------------------------------------
    # Cross split leakage verification
    # --------------------------------------------------------

    train_set = set(
        train_df["tweet"]
    )

    val_set = set(
        val_df["tweet"]
    )

    test_set = set(
        test_df["tweet"]
    )

    train_val_overlap = len(
        train_set & val_set
    )

    train_test_overlap = len(
        train_set & test_set
    )

    val_test_overlap = len(
        val_set & test_set
    )

    print("\nCross-split duplicate check:")

    print(
        "Train-Val overlap  :",
        train_val_overlap
    )

    print(
        "Train-Test overlap :",
        train_test_overlap
    )

    print(
        "Val-Test overlap   :",
        val_test_overlap
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print("\nSample cleaned tweets:")

    print(
        train_df.head(5).to_string(
            index=False
        )
    )

    print(
        "\nPreprocessing completed successfully."
    )