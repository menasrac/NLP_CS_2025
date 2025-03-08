# Challenge Kaggle NLP-CS-2025- Code du Groupe  

Ce dépôt regroupe le code utilisé par notre groupe dans le cadre du challenge Kaggle.  

## Structure du dépôt  

- **`add_id.py`** : Met en forme le fichier de prédictions pour soumission sur Kaggle.  
- **`extract_rare_languages.py`** : Renvoie la liste des langues peu représentées dans le trainset.  
- **`fine_tune_XML_R.py`** : Script de fine-tuning du modèle retenu (voir rapport).  
- **`fine_tuning_with_augmentations.py`** : Script d'entraînement sur une version augmentée des données pour équilibrer les classes.  
- **`predict.py`** & **`pred_tranductive.py`** : Deux stratégies de prédiction utilisées pour l'inférence.  
- **`train_submission_augmented.csv`** : Fichier de données enrichi.  

### Dossier `zip/`  
Ce dossier contient :  
- Des éléments d'analyse du dataset.  
- Des visualisations.  
- Nos premiers tests (peu concluants) avec FastText et LangDetect.  
