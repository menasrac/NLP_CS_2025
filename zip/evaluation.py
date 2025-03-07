import json

import matplotlib.pyplot as plt
import pandas as pd


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


def calculate_accuracy(output_csv, field, threshold=0.9):
    # Load the dataset with detected languages
    df = pd.read_csv(output_csv)

    # Apply the function to get the top language with threshold
    df["top_language"] = df[field].apply(lambda x: get_top_language(x, threshold))

    # Calculate accuracy
    correct_predictions = (df["Label"] == df["top_language"]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    print(f"Accuracy: {accuracy:.2%}")


def get_top_k_languages(detected_langs_str, k=2):
    try:
        if detected_langs_str == "unknown":
            return []
        detected_langs = json.loads(detected_langs_str)
        if detected_langs:
            sorted_langs = sorted(
                detected_langs.items(), key=lambda item: item[1], reverse=True
            )
            return [lang for lang, prob in sorted_langs[:k]]
        return []
    except Exception:
        return []


def calculate_accuracy_k_first(output_csv, field, k=2):
    # Load the dataset with detected languages
    df = pd.read_csv(output_csv)

    # Apply the function to get the top k languages
    df["top_k_languages"] = df[field].apply(lambda x: get_top_k_languages(x, k))

    # Calculate accuracy
    correct_predictions = df.apply(
        lambda row: row["Label"] in row["top_k_languages"], axis=1
    ).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions

    print(f"Accuracy with top {k} languages: {accuracy:.2%}")


def calculate_proportion_correct(output_csv, field, threshold=0.9):
    # Load the dataset with detected languages
    df = pd.read_csv(output_csv)

    # Apply the function to get the top language with threshold
    df["top_language"] = df[field].apply(lambda x: get_top_language(x, threshold))

    # Filter out rows where the predicted language is null
    df = df[df["top_language"] != "unknown"]

    # Calculate the proportion of correct predictions
    correct_predictions = (df["Label"] == df["top_language"]).sum()
    total_classified = len(df[df["top_language"] != "unknown"])
    proportion_correct = (
        correct_predictions / total_classified if total_classified > 0 else 0
    )

    print(f"Proportion of correct predictions: {proportion_correct:.2%}")


def proportion_classified(output_csv, field, threshold=0.9):
    # Load the dataset with detected languages
    df = pd.read_csv(output_csv)

    # Apply the function to check if the prediction is classified
    df["is_classified"] = df[field].apply(
        lambda x: get_top_language(x, threshold) != "unknown"
    )

    # Calculate the proportion of classified predictions
    classified_predictions = df[df["is_classified"]]
    total_predictions = len(df)
    proportion_classified = len(classified_predictions) / total_predictions

    print(f"Proportion of classified predictions: {proportion_classified:.2%}")


def plot_accuracy_by_class(output_csv, field, threshold=0.9):
    # Load the dataset with detected languages
    df = pd.read_csv(output_csv)

    # Apply the function to get the top language with threshold
    df["top_language"] = df[field].apply(lambda x: get_top_language(x, threshold))

    # Calculate accuracy by class
    class_accuracy = (
        df.groupby("top_language")
        .apply(lambda x: (x["Label"] == x["top_language"]).mean())
        .sort_values(ascending=False)
    )

    # Plot the accuracy by class
    class_accuracy.plot(kind="bar", figsize=(10, 6))
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Class")
    plt.xticks(rotation=45)
    plt.show()


# Calculate the accuracy of the predictions
output_csv = "train_submission_prediction.csv"
# print("Langdetect:")
# calculate_accuracy(output_csv, "langdetect", threshold=0.99)
# calculate_proportion_correct(output_csv, "langdetect", threshold=0.99)
# proportion_classified(output_csv, "langdetect", threshold=0.99)
# print("FastText:")
# calculate_accuracy(output_csv, "fasttext", threshold=0.5)
# calculate_proportion_correct(output_csv, "fasttext", threshold=0.5)
# proportion_classified(output_csv, "fasttext", threshold=0.5)

# plot_accuracy_by_class(output_csv, "langdetect", threshold=0.99)
# plot_accuracy_by_class(output_csv, "fasttext", threshold=0.9)

for k in range(1, 10):
    calculate_accuracy_k_first(output_csv, "langdetect", k=k)
