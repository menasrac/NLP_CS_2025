import os

import fasttext
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Load the CSV file
df = pd.read_csv("train_submission.csv")

# Prepare the data
texts = df["Text"].tolist()
labels = df["Label"].tolist()

# Number of folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

for fold, (train_index, val_index) in enumerate(kf.split(texts)):
    print(f"Fold {fold + 1}/{k}")

    # Split the data into training and validation sets
    train_texts = [texts[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    val_texts = [texts[i] for i in val_index]
    val_labels = [labels[i] for i in val_index]

    # Write the training and validation data to temporary files
    with open(
        "fine_tuned_fasttext/temp_train.txt", "w", encoding="utf-8"
    ) as train_file:
        for text, label in zip(train_texts, train_labels):
            train_file.write(f"__label__{label} {text}\n")

    with open("fine_tuned_fasttext/temp_val.txt", "w", encoding="utf-8") as val_file:
        for text, label in zip(val_texts, val_labels):
            val_file.write(f"__label__{label} {text}\n")

    # Train the FastText model
    model = fasttext.train_supervised(
        input="fine_tuned_fasttext/temp_train.txt",
        lr=0.35,
        epoch=25,
        wordNgrams=1,
        dim=100,
        loss="softmax",
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

# Display average training and validation loss
avg_train_loss = sum(train_losses) / k
avg_val_loss = sum(val_losses) / k
print(f"Average Training Loss: {avg_train_loss:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")


# Delete the temporary files
os.remove("fine_tuned_fasttext/temp_train.txt")
os.remove("fine_tuned_fasttext/temp_val.txt")

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, k + 1), train_losses, label="Training Loss")
plt.plot(range(1, k + 1), val_losses, label="Validation Loss")
plt.xlabel("Fold")
plt.ylabel("Loss")
plt.title("Training and Validation Losses per Fold")
plt.legend()
plt.show()
