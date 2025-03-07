import json

import pandas as pd
from langdetect import detect_langs
from tqdm import tqdm


def process_language_detection(input_csv, output_csv, iso_mapping_file):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Load the ISO 639-1 to ISO 639-3 correspondence
    with open(iso_mapping_file, "r") as f:
        iso_mapping = json.load(f)

    # Function to detect language and convert to ISO 639-3
    def detect_language(text):
        try:
            detected_langs = detect_langs(text)
            if detected_langs:
                # Store the entire result of detect_langs
                detected_langs_dict = {
                    iso_mapping.get(lang.lang, lang.lang): lang.prob
                    for lang in detected_langs
                }
                detected_langs_str = json.dumps(detected_langs_dict)
                return detected_langs_str
            return "unknown"
        except Exception:
            return "unknown"

    # Apply the function to the dataset with a progress bar
    tqdm.pandas()
    df["langdetect"] = df["Text"].progress_apply(detect_language)

    # Save the dataset
    df.to_csv(output_csv, index=False)


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


input_csv = "train_submission_prediction.csv"
output_csv = "train_submission_prediction.csv"
iso_mapping_file = "ISO_639-1_ISO_639-3_correspondance.json"
process_language_detection("train_submission.csv", output_csv, iso_mapping_file)
