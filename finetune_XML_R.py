import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Charger et préparer les données
df = pd.read_csv("train_submission.csv")
texts = df["Text"].tolist()
labels = df["Label"].tolist()

# Encoder les labels (convertir les classes en entiers)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)
print(f"Nombre de classes avant filtrage : {num_classes}")

# Compter les occurrences de chaque classe
counts = Counter(labels_encoded)
print("Distribution initiale :", counts)

# Filtrer les classes qui n'ont qu'un seul exemple
filtered_indices = [i for i, label in enumerate(labels_encoded) if counts[label] > 1]
texts_filtered = [texts[i] for i in filtered_indices]
labels_filtered = [labels_encoded[i] for i in filtered_indices]

# Réencoder les labels après filtrage
le_filtered = LabelEncoder()
labels_filtered = le_filtered.fit_transform(labels_filtered)
num_classes_filtered = len(le_filtered.classes_)

# Vérifier la nouvelle distribution
new_counts = Counter(labels_filtered)
print("Distribution après filtrage et ré-encodage :", new_counts)
print(f"Nombre de classes après filtrage : {num_classes_filtered}")

# Split train/validation sur les données filtrées
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts_filtered, labels_filtered, test_size=0.2, random_state=42, stratify=labels_filtered
)

# Vérification des labels après split
print("Labels uniques en train :", set(train_labels))
print("Labels uniques en val :", set(val_labels))

# Créer un Dataset HuggingFace
# Charger le tokenizer pour XLM-R
#tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
checkpoint_dir = "xlm_roberta_finetuned/checkpoint-14292"
tokenizer_dir = "xlm_roberta_finetuned"

# Charger le tokenizer et le modèle fine-tuné depuis le checkpoint
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

# Créer des dictionnaires pour les datasets
train_dict = {"text": train_texts, "labels": train_labels}
val_dict = {"text": val_texts, "labels": val_labels}

train_dataset = Dataset.from_dict(train_dict)
val_dataset = Dataset.from_dict(val_dict)

# Tokenisation
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Charger le modèle pré-entraîné XLM-R pour classification
#model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_classes_filtered)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)


# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./xlm_roberta_finetuned",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  
    gradient_accumulation_steps=2, 
)

# Définir la fonction de calcul de métriques
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Créer le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Lancer l'entraînement
print("Début de l'entrainement")
trainer.train()

# Sauvegarder le modèle finetuné
model.save_pretrained("xlm_roberta_finetuned")
tokenizer.save_pretrained("xlm_roberta_finetuned")
