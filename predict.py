import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tqdm import tqdm
import numpy as np


# Chargement des données et du modèle XLM-R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv("test_without_labels.csv")

# Spécifier le chemin vers le checkpoint spécifique et le dossier du tokenizer
checkpoint_dir = "xlm_roberta_finetuned/checkpoint-14292"
tokenizer_dir = "xlm_roberta_finetuned"

# Charger le tokenizer et le modèle fine-tuné depuis le checkpoint
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
model.eval()


# Chargement du mapping des labels à partir du train (pour revenir aux chaînes)

df_train = pd.read_csv("train_submission.csv")
y_train_raw = df_train["Label"].tolist()
# Convertir en array
y_train_raw = np.array(y_train_raw)

# Filtrer les classes rares dans l'ensemble d'entraînement
counts = Counter(y_train_raw)
valid_classes = {cls for cls, count in counts.items() if count >= 2}
mask = np.isin(y_train_raw, list(valid_classes))
y_filtered = y_train_raw[mask]

# Réencoder les labels pour qu'ils soient contigus (0 à num_class-1)
le = LabelEncoder()
labels_encoded = le.fit_transform(y_filtered)
# Le mapping inverse permettra de revenir aux chaînes
# print("Mapping (chaine -> encodé) :", dict(zip(le.inverse_transform(np.unique(labels_encoded)), np.unique(labels_encoded))))

# Ici, on utilise le même LabelEncoder pour les prédictions transductives.
# Les labels candidats seront donc les entiers encodés.
labels_to_check = np.unique(labels_encoded)
labels_to_check_str = le.inverse_transform(labels_to_check)
# print("Labels candidats (encodés) :", labels_to_check)
# print("Labels candidats (str)    :", labels_to_check_str)

# Fonction de prédiction avec XLM-R (calcul uniquement du label avec la probabilité max)

def get_xlmr_probabilities(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        # Calculer la distribution softmax
        outputs = model(**inputs).logits.softmax(dim=-1).cpu().numpy()[0]
    
    # print(f"outputs: {outputs}, shape: {outputs.shape}, type: {type(outputs)}")
    max_prob_idx = int(np.argmax(outputs))  # indice du max
    max_prob = float(outputs[max_prob_idx])
    max_label = max_prob_idx  
    return max_label, max_prob


# Traitement par batch pour réduire la consommation de mémoire

def process_batch(batch_texts):
    batch_pred_labels = []
    batch_pred_probs = []
    for text in batch_texts:
        label, prob = get_xlmr_probabilities(text)
        batch_pred_labels.append(label)
        batch_pred_probs.append(prob)
        # Libère la mémoire GPU si nécessaire
        torch.cuda.empty_cache()
    return batch_pred_labels, batch_pred_probs

# Appliquer par batch sur le test set
batch_size = 16
predicted_labels = []
predicted_probs = []
for i in tqdm(range(0, len(test_df), batch_size), desc="Prédictions"):
    batch = test_df["Text"].iloc[i:i+batch_size]
    batch_labels, batch_probs = process_batch(batch)
    predicted_labels.extend(batch_labels)
    predicted_probs.extend(batch_probs)

# Stocker ces résultats dans le DataFrame
test_df["predicted_label"] = predicted_labels  # labels sous forme d'entiers
test_df["max_probability"] = predicted_probs


# Affectation transductive avec contrainte souple (marge 5%)
# Ici, on se base sur les labels déjà prédits et leur probabilité max.
num_candidates = len(labels_to_check)  # nombre de classes (entiers)
num_test = len(test_df)
base_quota = num_test // num_candidates
tolerance = int(0.05 * base_quota)  # 5% de marge
min_quota = max(base_quota - tolerance, 0)
max_quota = min(base_quota + tolerance, num_test)

# Définir le quota par classe : pour chaque label (entier), [min_quota, max_quota]
target_quota = {lang: [min_quota, max_quota] for lang in labels_to_check}

# Préparer la liste des candidats avec (index, label prédit, probabilité)
assignment_candidates = []
for idx, row in test_df.iterrows():
    # On récupère le label prédit et sa probabilité
    pred_label = row["predicted_label"]
    pred_prob = row["max_probability"]
    assignment_candidates.append((idx, pred_label, pred_prob))

# Trier par probabilité décroissante
assignment_candidates.sort(key=lambda x: x[2], reverse=True)

# Allocation souple respectant le quota
assigned = {}  # index -> label affecté (entier)
language_counts = {lang: 0 for lang in labels_to_check}

# Remplir le min_quota pour chaque classe
for idx, label, prob in assignment_candidates:
    if idx not in assigned and language_counts[label] < target_quota[label][0]:
        assigned[idx] = label
        language_counts[label] += 1

# Affecter les exemples restants sans dépasser max_quota
for idx, label, prob in assignment_candidates:
    if idx not in assigned and language_counts[label] < target_quota[label][1]:
        assigned[idx] = label
        language_counts[label] += 1

# Dernier recours : pour les indices non affectés, on utilise le label déjà prédit
for idx in range(num_test):
    if idx not in assigned:
        assigned[idx] = test_df.at[idx, "predicted_label"]


# Conversion finale des labels (entiers) en str
# On utilise le LabelEncoder inverse pour revenir aux chaînes telles qu'elles étaient lors de l'entraînement.
test_df["Predicted_Label"] = test_df.index.map(lambda i: le.inverse_transform([assigned.get(i)])[0] if i in assigned else None)

# Supprimer les colonnes intermédiaires
test_df = test_df.drop(columns=["predicted_label", "max_probability", "Usage","Text"])

# Sauvegarder le résultat final dans un fichier CSV
test_df.to_csv("test_submission_t9.csv", index=False)

print("Affectation transductive terminée avec marge de 5%. Fichier généré : 'test_submission_t9.csv'.")
