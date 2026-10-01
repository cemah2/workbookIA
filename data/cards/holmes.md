# Holmes : *The Adventures of Sherlock Holmes*

| | |
|---|---|
| **Fichier** | `data/text/holmes_adventures_pg1661.txt` (0,6 Mo, UTF-8, fins de ligne LF) |
| **Chargement** | `wb.datasets.load_holmes(strip_header=True)` → `str` |
| **Source** | Project Gutenberg, eBook n° 1661 |
| **Taille (sans en-tête Gutenberg)** | ≈ 562 000 caractères, ≈ 104 500 mots |
| **Langue** | anglais |
| **Licence** | domaine public aux États-Unis (publié en 1892) ; le fichier garde l'en-tête et la licence Project Gutenberg, retirés au chargement |
| **Téléchargé le** | 2026-09-29 |

## Provenance
Arthur Conan Doyle, recueil de 12 nouvelles publiées dans *The Strand Magazine* (1891-1892).

## Utilisation
Fil rouge « texte anglais » : découverte et générateur de bigrammes (ch. 1), fréquences de lettres et entropie (ch. 6 : 431 462 lettres a à z et 88 caractères différents ; entropie des lettres, codes Morse et de Huffman, compression avec zlib, modèles unigramme et bigramme appris sur la première moitié des lettres et jugés sur la seconde), Naive Bayes pour distinguer Holmes et Verne (ch. 13), génération de texte caractère par caractère (ch. 22, 24), tokenisation, embeddings et mini-GPT (B2, B3, B4).
Le nombre de mots dépend de la définition : ≈ 104 500 en coupant aux espaces, 105 849 avec la fonction `words` du ch. 6, qui coupe aussi à la ponctuation et aux apostrophes (7 819 mots différents).
Par défaut, `load_holmes()` retire l'en-tête et le pied de page Gutenberg (licence, crédits) ; `strip_header=False` renvoie le fichier complet.

## Biais et limites
- Anglais victorien : vocabulaire et tournures datés ; un modèle entraîné ici écrit « comme en 1890 ».
- Un seul auteur, un seul genre : corpus minuscule comparé à ceux des LLM (des milliers de milliards de tokens).
- Représentations sociales de l'époque (classes, genres, colonies).
