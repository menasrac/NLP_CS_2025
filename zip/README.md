# Project Files Explanation

This folder contains the initial tests we conducted using `langdetect` and `fasttext` (including fine-tuned versions) as well as visualizations to understand their limitations and the dataset. We also attempted a preliminary classification based on alphabets.

- `langdetect_vs_fasttext.py`: Compares the performance of `langdetect` and `fasttext` on the dataset and visualizes the agreement between the two models.
- `lang_detect.py`: Uses `langdetect` to identify languages in the dataset and maps ISO 639-1 codes to ISO 639-3.
- `visualize_error.py`: Visualizes the errors and accuracy of the fine-tuned `fasttext` model.
- `train_fasttext.py`: Script to train a `fasttext` model on the provided training data.
- `fast_text.py`: Uses a pre-trained `fasttext` model for language detection and maps ISO 639-1 codes to ISO 639-3.
- `alphabet.py`: Classifies texts based on the most frequent alphabet detected in each example.

# langdetect
Utilise la lib `langdetect` qui utilise elle-même un filtre de Bayes à partir de profils de langues générés à partir d'un Wikipedia abstract xml.
55 classes
Accuracy annoncée sur les 55 classes : 99%
Donne 12% sur notre dataset
Langdetect renvoie esp à 0,9999 sur les texte classifié en arg. Fasttext gère arg

# fasttext
! Il faut la version 1 de numpy pour faire tourner la lib
Utilise la lib `fasttext`
Beaucoup plus rapide et beaucoup plus de classes (176)

Accuracy with top 1 languages: 12.23%
Accuracy with top 2 languages: 12.38%
Accuracy with top 3 languages: 12.38%
Accuracy with top 4 languages: 12.39%
Accuracy with top 5 languages: 12.39%
Accuracy with top 6 languages: 12.39%
Accuracy with top 7 languages: 12.39%
Accuracy with top 8 languages: 12.39%
Accuracy with top 9 languages: 12.39%

On remarque que le gain d'accuracy est très faible en augmentant le nombre de langues prédites. Il est donc pertinent de prendre uniquement la première prédiction.

# langdetect vs fasttext
Lanngdetect est plus précis mais moins rapide que fasttext. L'accuracy est très variable passant de 0 à 1 en fonction des langues. Les deux modèles sont assez performant (>0,7) sur une dizaine de langage et assez mauvais (<0,4) sur la majorité des autres langages.

# première classification par alphabet
On détecte l'alphabet le plus fréquent dans chaque exemple grâce au code unicode des caractères.

On obtient 33 labels qui contiennent des examples dans des alphabets différents. Dans la plupart des cas, on obtient un couple d'un alphabet rare et d'alphabet latin car le dataset contient entre autre des traductions. On remarque que dans la plupart des cas, c'est l'alphabet rare qui aurait du être choisi.

> "Le tadjik était autrefois écrit en alphabet perso-arabe. Les Soviétiques introduisirent l'alphabet latin à la fin des années 19202, puis introduisirent l'alphabet cyrillique en 19403,4. En 1989, une politique de retour à l'alphabet perso-arabe fut décidée par le gouvernement du Tadjikistan. Mais de facto, l'usage reste d'écrire le tadjik en cyrillique." - Wikipedia

On obtient 15 alphabets qui permettent à eux seuls de déterminer le label :
```python
alphabet_to_label = {
    'CANADIAN': 'iku',
    'GUJARATI': 'guj',
    'GURMUKHI': 'pan',
    'HANGUL': 'kor',
    'HIRAGANA': 'jpn',
    'KANNADA': 'kan',
    'KATAKANA': 'jpn',
    'KHMER': 'khm',
    'LAO': 'lao',
    'MALAYALAM': 'mal',
    'OL': 'sat',
    'SINHALA': 'sin',
    'TAMIL': 'tam',
    'THAANA': 'div',
    'THAI': 'tha'
}
```

Cette méthode permet probablement de classifier les langues les plus rares. Il peut donc être interessant de le combiner avec nos autres modèles.