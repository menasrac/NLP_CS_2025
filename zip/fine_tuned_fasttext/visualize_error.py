import fasttext
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the model
model = fasttext.load_model("fine_tuned_fasttext/fasttext_model_quantized.bin")

# Load the dataset
with open("fine_tuned_fasttext/test_data.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Prepare the data
texts = []
labels = []
for line in lines:
    label, text = line.split(" ", 1)
    labels.append(label.replace("__label__", ""))
    texts.append(text.strip())

# Predict the labels
predictions = [model.predict(text)[0][0] for text in texts]

# Convert predictions to the same format as labels
predictions = [pred.replace("__label__", "") for pred in predictions]

# Calculate accuracy
accuracy = accuracy_score(labels, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Create a DataFrame for analysis
df = pd.DataFrame({"Text": texts, "TrueLabel": labels, "PredictedLabel": predictions})

# Calculate accuracy by class
class_accuracy = (
    df.groupby("TrueLabel")
    .apply(lambda x: (x["TrueLabel"] == x["PredictedLabel"]).mean())
    .sort_values(ascending=False)
)

# Plot the accuracy by class
plt.figure(figsize=(10, 6))
class_accuracy.plot(kind="bar")
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.title("Accuracy by Class")
plt.xticks(rotation=45)
plt.show()

# Plot the accuracy by class against the number of training examples
with open("fine_tuned_fasttext/training_data.txt", "r", encoding="utf-8") as file:
    train_lines = file.readlines()

train_labels = {}
for line in train_lines:
    label, _ = line.split(" ", 1)
    label = label.replace("__label__", "")
    train_labels[label] = train_labels.get(label, 0) + 1

df["TrainExamples"] = df["TrueLabel"].apply(
    lambda x: train_labels[x] if x in train_labels else 0
)

# Plot the accuracy by class against the number of training examples
plt.figure(figsize=(10, 6))
class_accuracy = df.groupby("TrueLabel").apply(
    lambda x: (x["TrueLabel"] == x["PredictedLabel"]).mean()
)
df = df.merge(
    class_accuracy.rename("ClassAccuracy"), left_on="TrueLabel", right_index=True
)

plt.scatter(df["TrainExamples"], df["ClassAccuracy"])
plt.xlabel("Number of Training Examples")
plt.ylabel("Accuracy")
plt.title("Accuracy by Class vs. Number of Training Examples")
plt.show()
