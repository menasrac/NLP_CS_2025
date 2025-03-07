import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tqdm import tqdm
import numpy as np
import math

# Chargement des données et du modèle XLM-R
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv("test_without_labels.csv")

# Spécifier le chemin vers le checkpoint spécifique et le dossier du tokenizer
checkpoint_dir = "xlm_roberta_finetuned/checkpoint-9530"
tokenizer_dir = "xlm_roberta_finetuned"

# Charger le tokenizer et le modèle fine-tuné depuis le checkpoint
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
model.eval()

# Chargement du mapping des labels à partir du train
df_train = pd.read_csv("train_submission.csv")
y_train_raw = np.array(df_train["Label"].tolist())

# On garde toutes les classes (filtre ici avec >= 1 occurrence)
counts_train = Counter(y_train_raw)
valid_classes = {cls for cls, count in counts_train.items() if count >= 2}
mask = np.isin(y_train_raw, list(valid_classes))
y_filtered = y_train_raw[mask]

# Réencoder les labels pour qu'ils soient contigus
le = LabelEncoder()
labels_encoded = le.fit_transform(y_filtered)
labels_to_check = np.unique(labels_encoded)  # classes candidates (entiers)

# Calculer la distribution complète des probabilités pour le test ---
all_probabilities = []
for text in tqdm(test_df["Text"], desc="Calcul des probabilités"):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(dim=-1).cpu().numpy()[0]
    all_probabilities.append(probs)
all_probabilities = np.array(all_probabilities)  # shape: (num_test, num_classes)

num_test = len(test_df)
num_classes = len(labels_to_check)

# Définir la répartition cible basée sur le train ---
# Calculer la proportion de chaque classe dans le train
total_train = sum(counts_train.values())
train_counts_encoded = Counter(labels_encoded)
target_counts = {}
target_lower = {}
target_upper = {}

for c in labels_to_check:
    # Proportion de la classe c dans le train
    prop = train_counts_encoded[c] / total_train
    # Quota cible sur le test
    target = prop * num_test
    target_counts[c] = int(round(target))
    # Tolérance de ±5%
    target_lower[c] = math.floor(target * 0.98)
    target_upper[c] = math.ceil(target * 1.02)

# Ajuster la somme si besoin (optionnel) pour que la somme des quotas égale num_test
total_target = sum(target_counts.values())
if total_target != num_test:
    diff = num_test - total_target
    # On ajuste les classes avec le plus grand résidu
    residuals = {c: ((train_counts_encoded[c] / total_train) * num_test) - target_counts[c] for c in labels_to_check}
    # Si diff positif, on ajoute aux classes avec le plus grand résidu ; sinon on soustrait aux classes avec le plus petit résidu
    sorted_classes = sorted(labels_to_check, key=lambda c: residuals[c], reverse=(diff>0))
    for i in range(abs(diff)):
        c = sorted_classes[i]
        target_counts[c] += (1 if diff > 0 else -1)
        target_lower[c] = math.floor(target_counts[c] * 0.98)
        target_upper[c] = math.ceil(target_counts[c] * 1.02)

print("Distribution cible (basée sur le train) :", target_counts)
print("Bornes minimales :", target_lower)
print("Bornes maximales :", target_upper)

# Affectation transductive
final_assignment = {}  # index -> classe assignée (entier)
counts_assigned = {c: 0 for c in labels_to_check}

# Première passe : affectation greedy avec contrainte du quota supérieur de chaque classe
order = np.argsort(-np.max(all_probabilities, axis=1))
for i in order:
    sorted_classes = np.argsort(-all_probabilities[i])
    assigned = False
    for c in sorted_classes:
        if counts_assigned[c] < target_upper[c]:
            final_assignment[i] = c
            counts_assigned[c] += 1
            assigned = True
            break
    if not assigned:
        # Cas très rare : on affecte à la classe avec la meilleure probabilité
        c = sorted_classes[0]
        final_assignment[i] = c
        counts_assigned[c] += 1

print("Répartition initiale :", counts_assigned)

# Deuxième passe : réaffectation pour satisfaire le quota minimal de chaque classe
for c in labels_to_check:
    while counts_assigned[c] < target_lower[c]:
        candidate = None
        candidate_delta = -np.inf
        # Parcourir les exemples pour trouver un candidat pouvant être réassigné
        for i in range(num_test):
            current_class = final_assignment[i]
            if current_class != c and counts_assigned[current_class] > target_lower[current_class]:
                # Calcul du "coût" de réaffectation
                delta = all_probabilities[i][c] - all_probabilities[i][current_class]
                if delta > candidate_delta:
                    candidate = i
                    candidate_delta = delta
        if candidate is None:
            break  # plus de candidat disponible pour réaffecter
        old_class = final_assignment[candidate]
        final_assignment[candidate] = c
        counts_assigned[old_class] -= 1
        counts_assigned[c] += 1

print("Répartition finale :", counts_assigned)

# Conversion finale des labels (entiers) en labels originaux (string)
test_df["Predicted_Label"] = test_df.index.map(lambda i: le.inverse_transform([final_assignment[i]])[0])

# Conserver les colonnes souhaitées (on supprime "Text" et "Usage" si présents)
for col in ["Text", "Usage"]:
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

test_df.to_csv("test_submission_t9_uniform.csv", index=False)
print("Affectation transductive terminée. Fichier généré : 'test_submission_t9_uniform.csv'.")
