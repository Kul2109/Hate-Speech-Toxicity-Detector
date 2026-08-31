"""
Normalize an authorized Twitter/X export or other permitted source CSV.

Expected source columns can be adapted below. This script does NOT scrape
Twitter/X and does not bypass platform restrictions.
"""

import pandas as pd

INPUT = "data/raw/twitter_export.csv"
OUTPUT = "data/raw/twitter_normalized.csv"

# Change these mappings to match the columns in YOUR authorized export.
TEXT_SOURCE_COLUMN = "text"

# If labels already exist in your source, map them here.
LABEL_MAP = {
    "toxicity": "toxicity",
    "insult": "insult",
    "threat": "threat",
    "identity_attack": "identity_attack",
}

def main():
    df = pd.read_csv(INPUT)

    output = pd.DataFrame()
    output["text"] = df[TEXT_SOURCE_COLUMN].astype(str)

    for target, source in LABEL_MAP.items():
        if source not in df.columns:
            raise ValueError(
                f"Source label '{source}' is missing. "
                "Do not invent labels; use an authorized dataset with labels "
                "or perform a documented human annotation process."
            )
        output[target] = pd.to_numeric(df[source]).astype(int)

    output["source"] = "twitter"
    output.to_csv(OUTPUT, index=False)
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
