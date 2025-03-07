import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Charger le CSV d'entraînement
df = pd.read_csv("train_submission.csv")

# Extraire les colonnes Text et Label
texts = df["Text"].tolist()
labels = df["Label"].tolist()

# Encoder les labels en entiers
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Calculer la fréquence de chaque label
counts = Counter(labels_encoded)

# Identifier les labels rares : ceux qui apparaissent exactement une fois
rare_labels_enc = [label for label, count in counts.items() if count <= 499]
rare_labels_str = le.inverse_transform(rare_labels_enc)
print("Labels rares identifiés :", rare_labels_str)

# Extraire les lignes dont le label (en chaîne) appartient aux rares
rare_df = df[df["Label"].isin(rare_labels_str)]

# Créer un dictionnaire associant chaque rare label à la liste des phrases correspondantes
rare_dict = {}
for rare_label in rare_labels_str:
    sentences = rare_df[rare_df["Label"] == rare_label]["Text"].tolist()
    rare_dict[rare_label] = sentences

# Afficher quelques informations
print("\nNombre de phrases par label rare :")
for label, sentences in rare_dict.items():
    print(f"Label {label}: {len(sentences)} phrases")
    if sentences:
        print(f"Exemple: {sentences[0]}\n")
