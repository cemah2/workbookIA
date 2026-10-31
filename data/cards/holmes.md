# Holmes : *The Adventures of Sherlock Holmes*

| | |
|---|---|
| **Fichier** | `data/text/holmes_adventures_pg1661.txt` (0,6 Mo, UTF-8, fins de ligne LF) |
| **Chargement** | `wb.datasets.load_holmes(strip_header=True)` → `str` |
| **Source** | Project Gutenberg, [eBook n° 1661](https://www.gutenberg.org/ebooks/1661) |
| **Taille (sans en-tête Gutenberg)** | ≈ 562 000 caractères, ≈ 104 500 mots |
| **Langue** | anglais |
| **Licence** | domaine public aux États-Unis (publié en 1892) ; le fichier garde l'en-tête et la licence Project Gutenberg, retirés au chargement |
| **Empreinte** | SHA-256 du fichier : `51bf912e17919195e50cfa25458a6660ef35e00a5f44598b221196285c2220a5` |
| **Téléchargé le** | 2026-09-29 |

## Provenance
Arthur Conan Doyle, recueil de 12 nouvelles publiées dans *The Strand Magazine* (1891-1892).

## Utilisation
Fil rouge « texte anglais » : découverte et générateur de bigrammes (ch. 1), fréquences de lettres et entropie (ch. 6 : 431 462 lettres a à z et 88 caractères différents ; entropie des lettres, codes Morse et de Huffman, compression avec zlib, modèles unigramme et bigramme appris sur la première moitié des lettres et jugés sur la seconde), vocabulaire du raisonnement (ch. 11 : 29 mots de la famille « deduc » contre 57 « observ », le faux positif *infernal*, la maxime « when you have excluded the impossible » dans « The Beryl Coronet » et « It is a capital mistake to theorise before one has data » dans « A Scandal in Bohemia » ; découpage en douze nouvelles), Naïve Bayes multinomial (ch. 13 : anglais ou français à partir des lettres de morceaux de 10 à 200 caractères ; phrases de dialogue, entre guillemets « “ » et « ” », contre narration : 82 % de dialogue, sac de mots, TF-IDF et régression logistique), vocabulaire des caractères et génération de texte caractère par caractère (ch. 22 : 88 caractères, l'espace en premier ; un modèle de paires de lettres à quatre températures ; un LSTM de deux couches entraîné sur les 90 premiers pour cent, validé sur les 10 derniers, perplexité de validation de 4,7 au mieux contre 22,4 pour un modèle sans contexte ; ch. 24), tokenisation, embeddings et mini-GPT (B2, B3, B4). Mini-projet MP1 (détecteur de langue) : nouvelles 1 à 8 pour l'entraînement, 9 et 10 pour la validation, 11 et 12 pour le test.
Le nombre de mots dépend de la définition : ≈ 104 500 en coupant aux espaces, 105 849 avec la fonction `words` du ch. 6, qui coupe aussi à la ponctuation et aux apostrophes (7 819 mots différents).
Par défaut, `load_holmes()` retire l'en-tête et le pied de page Gutenberg (licence, crédits) ; `strip_header=False` renvoie le fichier complet.

## Biais et limites
- Anglais victorien : vocabulaire et tournures datés ; un modèle entraîné ici écrit « comme en 1890 ».
- Un seul auteur, un seul genre : corpus minuscule comparé à ceux des LLM (des milliers de milliards de tokens).
- Représentations sociales de l'époque (classes, genres, colonies).
