import json

import fasttext
import pandas as pd
from tqdm import tqdm


def process_language_detection(input_csv, output_csv):
    # Load the pre-trained FastText model for language identification
    model = fasttext.load_model("lid.176.bin")

    def detect_language(text):
        predictions = model.predict(text, k=-1)  # Get all predictions
        lang_iso_639 = [pred.split("__label__")[1] for pred in predictions[0]]
        confidence = predictions[1]

        with open("ISO_639-1_ISO_639-3_correspondance.json", "r") as f:
            iso_639_1_to_3 = json.load(f)

        lang_iso_639_3 = [
            iso_639_1_to_3.get(lang, "unknown") if len(lang) == 2 else lang
            for lang in lang_iso_639
        ]

        return json.dumps(
            {lang: prob for lang, prob in zip(lang_iso_639_3, confidence)}
        )

    # Load the dataset
    df = pd.read_csv(input_csv)

    # Apply the function to the dataset with a progress bar
    tqdm.pandas()
    df["fasttext"] = df["Text"].progress_apply(detect_language)

    # Save the dataset
    df.to_csv(output_csv, index=False)


# Uncomment the following line to run the detection process
input_csv = "train_submission_prediction.csv"
output_csv = "train_submission_prediction.csv"
process_language_detection("train_submission_prediction.csv", output_csv)
