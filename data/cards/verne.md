# Verne : *Le Tour du monde en quatre-vingts jours*

| | |
|---|---|
| **Fichier** | `data/text/verne_tour_du_monde_pg800.txt` (0,45 Mo, UTF-8, fins de ligne LF) |
| **Chargement** | `wb.datasets.load_verne(strip_header=True)` → `str` |
| **Source** | Project Gutenberg, eBook n° 800 (transcription « ebooksgratuits ») |
| **Taille (sans en-tête Gutenberg)** | ≈ 421 000 caractères, ≈ 67 400 mots |
| **Langue** | français |
| **Licence** | domaine public (roman de 1872-1873) ; en-tête et licence Project Gutenberg conservés dans le fichier, retirés au chargement |
| **Téléchargé le** | 2026-09-29 |

## Provenance
Jules Verne, roman paru en feuilleton en 1872 puis en volume en 1873.

## Utilisation
Fil rouge « texte français » et point de comparaison avec Holmes : découverte et faux Verne (ch. 1), fréquences de lettres et entropie de deux langues (ch. 6 : 319 510 lettres a à z, plus de 3 % de lettres accentuées, dont « é » à 1,8 %, et neuf lettres accentuées absentes de Holmes ; le français envoyé avec le code de l'anglais), vocabulaire du raisonnement comparé à Holmes (ch. 11 : aucun « déduction » ni « déduire », une famille « infér » réduite à « inférieurs » et « inférieures », 24 mots de la famille « observ » ; un découpage en mots limité aux lettres a à z perd les mots accentués), classification de la langue ou de l'auteur (ch. 13), génération de texte (ch. 22), tokenisation (B2 : combien de tokens pour le même sens en anglais et en français ?). Mini-projet MP1 (détecteur de langue) : chapitres 1 à 25 pour l'entraînement, 26 à 31 pour la validation, 32 à 37 pour le test.

Le nombre de mots dépend de la définition : ≈ 67 400 en coupant aux espaces, 72 367 avec la fonction `words` du ch. 6, qui coupe aussi aux apostrophes (« l'homme » compte deux mots), pour 8 793 mots différents.

## Biais et limites
- Particularités de la transcription : certaines majuscules ne sont pas accentuées (« TROUVE SON IDEAL ») et les tirets sont codés « -- ». À garder en tête pour les statistiques de caractères.
- Français du XIXᵉ siècle, un seul auteur ; stéréotypes de l'époque sur les peuples rencontrés (sujet de discussion ⚖️ sur les biais des corpus).
