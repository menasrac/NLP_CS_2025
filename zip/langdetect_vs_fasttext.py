import json

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Load the datasets
df = pd.read_csv("train_submission_prediction.csv")


def get_top_language(detected_langs_str, threshold=0.9):
    try:
        if detected_langs_str == "unknown":
            return "unknown"
        detected_langs = json.loads(detected_langs_str)
        if detected_langs:
            top_lang, top_prob = sorted(
                detected_langs.items(), key=lambda item: item[1], reverse=True
            )[0]
            if top_prob >= threshold:
                return top_lang
        return "unknown"
    except Exception:
        return "unknown"


# Add fasttext_toplang and langdetect_toplang columns
df["fasttext_toplang"] = df["fasttext"].apply(lambda x: get_top_language(x, 0.5))
df["langdetect_toplang"] = df["langdetect"].apply(lambda x: get_top_language(x, 0.99))

print("Get common classes...")
# Filter the common classes
common_classes = set(df["fasttext_toplang"]).intersection(set(df["langdetect_toplang"]))
# Print number of common classes
print(f"Number of common classes: {len(common_classes)}")

print("Filter the merged dataset...")
filtered_df = df[
    df["fasttext_toplang"].isin(common_classes)
    & df["langdetect_toplang"].isin(common_classes)
]

# Calculate the number of agreements
agreement_results = {}

print("Calculating agreement results...")
for language in tqdm(common_classes, desc="Calculating agreement results"):
    subset_df = filtered_df[
        (filtered_df["fasttext_toplang"] == language)
        & (filtered_df["langdetect_toplang"] == language)
    ]
    agreement_count = len(subset_df)
    total_count = len(
        filtered_df[
            (filtered_df["fasttext_toplang"] == language)
            | (filtered_df["langdetect_toplang"] == language)
        ]
    )
    agreement_results[language] = (
        agreement_count / total_count if total_count > 0 else 0
    )


# Plot the agreement results
languages = list(agreement_results.keys())
agreements = list(agreement_results.values())

plt.figure(figsize=(12, 8))
plt.barh(languages, agreements, color="skyblue")
plt.xlabel("Agreement Ratio")
plt.ylabel("Languages")
plt.title("Agreement Ratio between FastText and LangDetect")
plt.grid(axis="x")

plt.show()

# Plot the accuracy class by class using Label for each language field
# Calculate accuracy for each class
fig, ax = plt.subplots(1, 2, figsize=(12, 16))

alphabet_to_label = {
    "CANADIAN": "iku",
    "GUJARATI": "guj",
    "GURMUKHI": "pan",
    "HANGUL": "kor",
    "HIRAGANA": "jpn",
    "KANNADA": "kan",
    "KATAKANA": "jpn",
    "KHMER": "khm",
    "LAO": "lao",
    "MALAYALAM": "mal",
    "OL": "sat",
    "SINHALA": "sin",
    "TAMIL": "tam",
    "THAANA": "div",
    "THAI": "tha",
}

# Filter out the languages that are in alphabet_to_label
filtered_common_classes = [
    lang for lang in common_classes if lang not in alphabet_to_label.values()
]

for idx, field in enumerate(["fasttext_toplang", "langdetect_toplang"]):
    accuracy_results = {}

    print(f"Calculating accuracy results for {field}...")
    for language in tqdm(
        filtered_common_classes, desc=f"Calculating accuracy results for {field}"
    ):
        correct_predictions = len(
            filtered_df[
                (filtered_df[field] == language) & (filtered_df["Label"] == language)
            ]
        )
        total_predictions = len(filtered_df[filtered_df[field] == language])
        accuracy_results[language] = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )

    # Sort the accuracy results
    sorted_accuracy_results = dict(
        sorted(accuracy_results.items(), key=lambda item: item[1], reverse=True)
    )

    # Plot the accuracy results
    languages = list(sorted_accuracy_results.keys())
    accuracies = list(sorted_accuracy_results.values())

    ax[idx].barh(languages, accuracies, color="lightgreen")
    ax[idx].set_xlabel("Accuracy")
    ax[idx].set_ylabel("Languages")
    ax[idx].set_title(f"Accuracy of {field} Predictions by Language")
    ax[idx].grid(axis="x")

plt.tight_layout()
plt.show()
