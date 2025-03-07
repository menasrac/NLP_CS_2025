import unicodedata

import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the CSV file
data = pd.read_csv("train_submission.csv")


def extraire_alphabet(phrase):
    alphabet_count = {}
    for char in phrase:
        if char.isalpha():  # Vérifie si c'est une lettre
            script = unicodedata.name(char).split()[
                0
            ]  # Extrait la première partie du nom Unicode
            if script in alphabet_count:
                alphabet_count[script] += 1
            else:
                alphabet_count[script] = 1
    if alphabet_count:
        return {max(alphabet_count, key=alphabet_count.get)}
    return set()


def categorize_language(text):
    alphabets = extraire_alphabet(text)
    if alphabets:
        return alphabets.pop()
    return "other"


# Apply the function to the Text column and create a new column for the language category
data["Alphabet"] = data["Text"].apply(categorize_language)

# Display the first few rows of the dataframe
print(data.head())

# Display the value counts for the Alphabet column
print(data["Alphabet"].value_counts())

# Check if there is a label which containing at least one example from one alphabet and another in another alphabet
multiple_alphabet_count = 0
single_alphabet_count = 0
for label in data["Label"].unique():
    label_data = data[data["Label"] == label]
    alphabets = label_data["Alphabet"].unique()
    if len(alphabets) > 1:
        print(f"Label {label} contains examples from multiple alphabets: {alphabets}")
        multiple_alphabet_count += 1
    else:
        single_alphabet_count += 1
print(f"Labels with examples from multiple alphabets: {multiple_alphabet_count}")
print(f"Labels with examples from a single alphabet: {single_alphabet_count}")

for label in data["Label"].unique():
    label_data = data[data["Label"] == label]
    alphabets = label_data["Alphabet"].unique()
    if len(alphabets) > 1:
        print(f"Label {label} contains examples from multiple alphabets: {alphabets}")
        for alphabet in alphabets:
            example = label_data[label_data["Alphabet"] == alphabet]["Text"].iloc[0]
            print(f"Example of {label} in {alphabet}: {example}")

# Print the list of alphabet with only one associated label. It means that knowing the alphabet is enough to predict the label.
alphabet_to_label = data.groupby("Alphabet")["Label"].nunique()
single_label_alphabets = alphabet_to_label[alphabet_to_label == 1].index.tolist()
print(f"Alphabets with only one associated label: {single_label_alphabets}")
# Display the correspondance for these alphabets
for alphabet in single_label_alphabets:
    label = data[data["Alphabet"] == alphabet]["Label"].iloc[0]
    print(f"Alphabet {alphabet} corresponds to label {label}")

# Print the accuracy when using only this alphabet to predict the label.
# Calculate the number of data points with single_label_alphabets
single_label_data_count = data[data["Alphabet"].isin(single_label_alphabets)].shape[0]
total_data_count = data.shape[0]

# Print the ratio of data points with single_label_alphabets over the total number of data points
print(f"Number of data points with single_label_alphabets: {single_label_data_count}")
print(f"Total number of data points: {total_data_count}")
print(f"Ratio: {single_label_data_count / total_data_count:.2f}")

# Print one example of the canadian alphabet
canadian_example = data[data["Alphabet"] == "CANADIAN"]["Text"].iloc[0]
print(f"Example of Canadian alphabet: {canadian_example}")

# Plot the number of classes per alphabet

# Count the number of unique labels per alphabet
alphabet_label_counts = (
    data.groupby("Alphabet")["Label"].nunique().sort_values(ascending=False)
)

# Plot the counts
fig, ax = plt.subplots(figsize=(10, 6))
alphabet_label_counts.plot(kind="bar", ax=ax)
plt.title("Number of Classes per Alphabet")
plt.xlabel("Alphabet")
plt.ylabel("Number of Classes")
plt.xticks(rotation=45)

# Annotate the bars with the exact counts
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.show()
