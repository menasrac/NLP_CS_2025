##!/usr/bin/env python
import argparse
import pandas as pd

def add_id_and_remove_others(input_csv, output_csv, start=1):
    # Charger le fichier CSV
    df = pd.read_csv(input_csv)
    
    # Supprimer la colonne "Usage" si elle existe
    if "Usage" in df.columns:
        df = df.drop(columns=["Usage"])
        print("Colonne 'Usage' supprimée.")
    if "Text" in df.columns:
        df = df.drop(columns=["Text"])
        print("Colonne 'Text' supprimée.")
    else:
        print("La colonne 'Usage' n'existe pas dans le fichier.")

    # Ajouter une colonne "ID" commençant à la valeur 'start'
    df.insert(0, "ID", range(start, start + len(df)))
    
    # Sauvegarder le DataFrame modifié dans le fichier CSV de sortie
    df.to_csv(output_csv, index=False)
    print(f"Colonne ID ajoutée (commençant à {start}) et sauvegardée dans '{output_csv}'.")

def main():
    parser = argparse.ArgumentParser(description="Ajoute une colonne ID et supprime la colonne 'Usage' dans un fichier CSV.")
    parser.add_argument("input_csv", help="Chemin du fichier CSV d'entrée.")
    parser.add_argument("output_csv", help="Chemin du fichier CSV de sortie.")
    parser.add_argument("--start", type=int, default=1, help="Valeur de départ pour la colonne ID (défaut : 1).")
    args = parser.parse_args()
    
    add_id_and_remove_others(args.input_csv, args.output_csv, args.start)

if __name__ == "__main__":
    main()
