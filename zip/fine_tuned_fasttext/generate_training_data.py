import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv("train_submission.csv")

# Split the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Open files to write the training and test data
with open(
    "fine_tuned_fasttext/training_data.txt", "w", encoding="utf-8"
) as train_file, open(
    "fine_tuned_fasttext/test_data.txt", "w", encoding="utf-8"
) as test_file:

    for index, row in train_df.iterrows():
        text = row["Text"]
        label = row["Label"]
        train_file.write(f"__label__{label} {text}\n")

    for index, row in test_df.iterrows():
        text = row["Text"]
        label = row["Label"]
        test_file.write(f"__label__{label} {text}\n")

print("Training and test data generated successfully.")
