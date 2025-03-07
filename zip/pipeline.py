import unicodedata

import pandas as pd

# Load data
df = pd.read_csv("train_submission.csv")

# region Step 1 : Use exotic alphabet to identify 15 languages

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


def get_alphabet(text):
    alphabet_count = {}
    for char in text:
        if char.isalpha():
            script = unicodedata.name(char).split()[0]
            if script in alphabet_count:
                alphabet_count[script] += 1
            else:
                alphabet_count[script] = 1
    if alphabet_count:
        return max(alphabet_count, key=alphabet_count.get)
    return "other"


df["Alphabet"] = df["Text"].apply(get_alphabet)
df["Predicted_Label"] = df["Alphabet"].map(alphabet_to_label)

print(df["Predicted_Label"].value_counts())

# endregion

# region Step 2 : Use langdetect

# region Calculate accuracy

correct_predictions = (df["Label"] == df["Predicted_Label"]).sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy:.2%}")

# Calculate the proportion of correct predictions excluding null Predicted_Label
non_null_predictions = df["Predicted_Label"].notnull().sum()
correct_predictions = (df["Label"] == df["Predicted_Label"]).sum()
proportion_correct = correct_predictions / non_null_predictions
print(f"Proportion of correct predictions: {proportion_correct:.2%}")

# endregion
