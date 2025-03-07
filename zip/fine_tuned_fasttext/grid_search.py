import itertools
import os

import fasttext
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv("train_submission.csv")

# Prepare the data
texts = df["Text"].tolist()
labels = df["Label"].tolist()

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Write the training and validation data to temporary files
with open("fine_tuned_fasttext/temp_train.txt", "w", encoding="utf-8") as train_file:
    for text, label in zip(train_texts, train_labels):
        train_file.write(f"__label__{label} {text}\n")

with open("fine_tuned_fasttext/temp_val.txt", "w", encoding="utf-8") as val_file:
    for text, label in zip(val_texts, val_labels):
        val_file.write(f"__label__{label} {text}\n")

# Define the parameter search space
param_grid = {
    "lr": [0.35],
    "epoch": [25],
    "wordNgrams": [1],
    "dim": [100],
    "loss": ["softmax"],
    "bucket": [200000],
    "ws": [5],
    "t": [1e-4],
    "lrUpdateRate": [100],
    "neg": [5],
    "minCount": [2],
    "minCountLabel": [1],
    "minn": [0],
    "maxn": [0],
}

# Generate all combinations of parameters
param_combinations = list(itertools.product(*param_grid.values()))

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []
best_val_accuracy = 0
best_params = {}

for i, params in enumerate(param_combinations):
    print(f"Iteration {i + 1}/{len(param_combinations)}")

    # Map the parameter combination to a dictionary
    params_dict = dict(zip(param_grid.keys(), params))
    print(f"Parameters: {params_dict}")

    # Train the FastText model with the selected parameters
    model = fasttext.train_supervised(
        input="fine_tuned_fasttext/temp_train.txt",
        lr=params_dict["lr"],
        epoch=params_dict["epoch"],
        wordNgrams=params_dict["wordNgrams"],
        bucket=params_dict["bucket"],
        dim=params_dict["dim"],
        loss=params_dict["loss"],
        ws=params_dict["ws"],
        t=params_dict["t"],
        lrUpdateRate=params_dict["lrUpdateRate"],
        neg=params_dict["neg"],
        minCount=params_dict["minCount"],
        minCountLabel=params_dict["minCountLabel"],
        minn=params_dict["minn"],
        maxn=params_dict["maxn"],
    )

    # Evaluate the model on the validation set
    val_predictions = [
        model.predict(text)[0][0].replace("__label__", "") for text in val_texts
    ]
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Calculate training and validation loss
    train_loss = model.test("fine_tuned_fasttext/temp_train.txt")[2]
    val_loss = model.test("fine_tuned_fasttext/temp_val.txt")[2]
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Update the best parameters if the current model is better
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_params = params_dict

print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
print(f"Best Parameters: {best_params}")

# Delete the temporary files
os.remove("fine_tuned_fasttext/temp_train.txt")
os.remove("fine_tuned_fasttext/temp_val.txt")

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(param_combinations) + 1), train_losses, label="Training Loss")
plt.plot(range(1, len(param_combinations) + 1), val_losses, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training and Validation Losses per Iteration")
plt.legend()
plt.show()
