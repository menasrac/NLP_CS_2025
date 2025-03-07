import fasttext
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
