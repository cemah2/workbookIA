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
Fil rouge « texte français » et point de comparaison avec Holmes : fréquences de lettres et entropie de deux langues (ch. 6), classification de la langue ou de l'auteur (ch. 13), génération de texte (ch. 22), tokenisation (B2 : combien de tokens pour le même sens en anglais et en français ?).

## Biais et limites
- Particularités de la transcription : certaines majuscules ne sont pas accentuées (« TROUVE SON IDEAL ») et les tirets sont codés « -- ». À garder en tête pour les statistiques de caractères.
- Français du XIXᵉ siècle, un seul auteur ; stéréotypes de l'époque sur les peuples rencontrés (sujet de discussion ⚖️ sur les biais des corpus).
