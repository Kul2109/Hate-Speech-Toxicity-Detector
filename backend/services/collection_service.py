import pandas as pd


def analyze_dataframe(df, predictor):
    """
    Run the existing toxicity model on a DataFrame.

    The DataFrame must contain a 'text' column.
    """

    if "text" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'text' column."
        )

    results = []

    for text in df["text"].fillna("").astype(str):

        if not text.strip():

            results.append({
                label: {
                    "score": 0.0,
                    "flagged": False
                }
                for label in predictor.labels
            })

        else:

            prediction = predictor.predict(text)

            results.append(
                prediction["labels"]
            )

    output = df.copy()

    for label in predictor.labels:

        output[f"{label}_score"] = [
            result[label]["score"]
            for result in results
        ]

        output[f"{label}_flagged"] = [
            int(result[label]["flagged"])
            for result in results
        ]

    return output