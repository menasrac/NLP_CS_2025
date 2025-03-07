import fasttext

# Path to the training data
training_data_path = "fine_tuned_fasttext/training_data.txt"

# Train the FastText model
model = fasttext.train_supervised(
    input=training_data_path,
    lr=0.35,
    epoch=25,
    wordNgrams=1,
    dim=100,
    loss="softmax",
)

# Save the model
model.save_model("fine_tuned_fasttext/fasttext_model.bin")

print("Model training completed and saved as 'fasttext_model.bin'")
