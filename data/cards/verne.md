# Verne : *Le Tour du monde en quatre-vingts jours*

| | |
|---|---|
| **Fichier** | `data/text/verne_tour_du_monde_pg800.txt` (0,45 Mo, UTF-8, fins de ligne LF) |
| **Chargement** | `wb.datasets.load_verne(strip_header=True)` → `str` |
| **Source** | Project Gutenberg, [eBook n° 800](https://www.gutenberg.org/ebooks/800) (transcription « ebooksgratuits ») |
| **Taille (sans en-tête Gutenberg)** | ≈ 421 000 caractères, ≈ 67 400 mots |
| **Langue** | français |
| **Licence** | domaine public (roman de 1872-1873) ; en-tête et licence Project Gutenberg conservés dans le fichier, retirés au chargement |
| **Empreinte** | SHA-256 du fichier : `7ec8a7d64e89bce4d86ed83a3b588ce6cd518a987588848e1ccbda7473ba1831` |
| **Téléchargé le** | 2026-09-29 |

## Provenance
Jules Verne, roman paru en feuilleton en 1872 puis en volume en 1873.

## Utilisation
Fil rouge « texte français » et point de comparaison avec Holmes : découverte et faux Verne (ch. 1), fréquences de lettres et entropie de deux langues (ch. 6 : 319 510 lettres a à z, plus de 3 % de lettres accentuées, dont « é » à 1,8 %, et neuf lettres accentuées absentes de Holmes ; le français envoyé avec le code de l'anglais), vocabulaire du raisonnement comparé à Holmes (ch. 11 : aucun « déduction » ni « déduire », une famille « infér » réduite à « inférieurs » et « inférieures », 24 mots de la famille « observ » ; un découpage en mots limité aux lettres a à z perd les mots accentués), anglais ou français par Naïve Bayes multinomial sur les lettres (ch. 13 : de 77 % pour 10 caractères à 99,9 % pour 200 ; le « w », rare, apparaît tout de même 248 fois), vocabulaire de mots et génération de texte (ch. 28 : 88 250 tokens avec les lettres accentuées, 4,3 % de `<unk>` pour 5 000 mots, perplexité de validation ≈ 113 après 12 epochs d'un LSTM à deux couches ; ch. 22 : les 2 000 mots les plus fréquents, `<unk>` compris, laissent 13,5 % des mots inconnus ; le même LSTM que Holmes tire davantage du contexte sur Verne, 2,85 bits par caractère contre 3,07 après deux epochs sur 120 000 caractères), tokenisation (B2 : l'extrait figé `[100000:150000]`, 50 000 caractères et 51 497 octets, mesuré par six tokenizers réels ; `wb.text.train_val("verne")`, les 90 premiers pour cent coupés au début d'une ligne, 379 133 caractères, et les 42 203 derniers pour la validation des BPE de ta librairie ; ton BPE caractère de 2 000 tokens fait de « Passepartout » un seul token ; word2vec en 📦 B2.25 sur les 1 590 mots vus au moins 5 fois, soit 60 979 mots une fois les rares retirés, les voisins de « fogg » et « passepartout » en 🔮 B2.26 et leurs cartes en 📈 B2.27 ; le défi 🏆 B2.31, la meilleure compression sans perte de la validation avec 2 000 tokens, contrôlée sur le dernier neuvième de l'entraînement, tenu à l'écart ; en B3, 🔮 B3.22 : une phrase de la validation et toute la validation, réécrites avec les 88 caractères de Holmes (l'apostrophe et les guillemets de Holmes, les accents que Holmes n'emploie pas retirés, 42 203 caractères), données au mini-GPT de Holmes ; en B4, `wb.rag.documents("verne")`, les chapitres I à VII (93 chunks de 120 mots), et 10 questions annotées en français (`data/qa/verne_qa.json`, voir [qa.md](qa.md)) : e5 multilingue contre MiniLM et BM25 en 📦 B4.20). Mini-projet MP1 (détecteur de langue) : chapitres 1 à 25 pour l'entraînement, 26 à 31 pour la validation, 32 à 37 pour le test.

Le nombre de mots dépend de la définition : ≈ 67 400 en coupant aux espaces, 72 367 avec la fonction `words` du ch. 6, qui coupe aussi aux apostrophes (« l'homme » compte deux mots), pour 8 793 mots différents ; `wb.text.words("verne")` (B2, `[^\W\d_]+` sur le texte en minuscules) donne les mêmes 72 367 mots.

## Biais et limites
- Particularités de la transcription : certaines majuscules ne sont pas accentuées (« TROUVE SON IDEAL ») et les tirets sont codés « -- ». À garder en tête pour les statistiques de caractères.
- Français du XIXᵉ siècle, un seul auteur ; stéréotypes de l'époque sur les peuples rencontrés (sujet de discussion ⚖️ sur les biais des corpus).
