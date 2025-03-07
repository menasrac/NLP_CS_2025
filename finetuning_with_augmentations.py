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
df = pd.read_csv("train_submission_augmented.csv")
df["Text"] = df["Text"].astype(str)
df["Label"] = df["Label"].astype(str)

texts = df["Text"].tolist()
labels = df["Label"].tolist()

# Encoder les labels (convertir les classes en entiers)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)
print(f"Nombre de classes : {num_classes}")

# Compter les occurrences de chaque classe
counts = Counter(labels_encoded)
print("Distribution après augmentation :", counts)

# Utiliser les labels encodés pour le split (au lieu de labels, qui sont des chaînes)
train_texts, val_texts, train_labels, val_labels = \
    train_test_split(texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Vérification des labels après split
print("Labels uniques en train :", set(train_labels))
print("Labels uniques en val   :", set(val_labels))

# Créer un Dataset HuggingFace
checkpoint_dir = "xlm_roberta_finetuned/checkpoint-16737"
tokenizer_dir = "xlm_roberta_finetuned"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

train_dict = {"text": train_texts, "labels": train_labels}
val_dict = {"text": val_texts, "labels": val_labels}

train_dataset = Dataset.from_dict(train_dict)
val_dataset = Dataset.from_dict(val_dict)

train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
val_dataset = val_dataset.map(preprocess_function, batched=True, num_proc=4)

# Charger le modèle pré-entraîné XLM-R pour classification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=num_classes).to(device)
#model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_classes)

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./xlm_roberta_finetuned",
    num_train_epochs=6,
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
from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Début de l'entraînement")
trainer.train()

# Sauvegarder le modèle fine-tuné
model.save_pretrained("xlm_roberta_finetuned")
tokenizer.save_pretrained("xlm_roberta_finetuned")

