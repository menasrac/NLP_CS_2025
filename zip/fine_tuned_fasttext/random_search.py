import os
import random

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
    "lr": [0.1, 0.2, 0.3, 0.4, 0.5],
    "epoch": [25, 50, 75, 100],
    "wordNgrams": [1, 2, 3],
    "dim": [50, 100, 150],
}

# Number of random search iterations
n_iter = 20

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []
best_val_accuracy = 0
best_params = {}

for i in range(n_iter):
    print(f"Iteration {i + 1}/{n_iter}")

    # Randomly select parameters
    params = {key: random.choice(values) for key, values in param_grid.items()}
    print(f"Parameters: {params}")

    # Train the FastText model with the selected parameters
    model = fasttext.train_supervised(
        input="fine_tuned_fasttext/temp_train.txt",
        lr=params["lr"],
        epoch=params["epoch"],
        wordNgrams=params["wordNgrams"],
        bucket=200000,
        dim=params["dim"],
        loss="ova",
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
        best_params = params

print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
print(f"Best Parameters: {best_params}")

# Delete the temporary files
os.remove("fine_tuned_fasttext/temp_train.txt")
os.remove("fine_tuned_fasttext/temp_val.txt")

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_iter + 1), train_losses, label="Training Loss")
plt.plot(range(1, n_iter + 1), val_losses, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training and Validation Losses per Iteration")
plt.legend()
plt.show()
