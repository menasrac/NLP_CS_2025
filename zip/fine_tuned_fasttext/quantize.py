import fasttext

# Load the pre-trained FastText model
model = fasttext.load_model("fine_tuned_fasttext/fasttext_model.bin")

# Quantize the model
model.quantize(input="fasttext_model.bin", retrain=True)

# Save the quantized model
model.save_model("fine_tuned_fasttext/fasttext_model_quantized.bin")

print(
    "Model quantization complete and saved as 'fine_tuned_fasttext/fasttext_model_quantized.bin'"
)
