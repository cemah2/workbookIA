# Data card · TODO : le nom du dataset

<!-- Le modèle de data card du ch. 29 (🛠️ 29.15) : reprends ta CARD_15 et complète-la (PF.2). Remplace chaque paragraphe qui commence par TODO. Les fiches de data/cards/ du workbook montrent le niveau de détail attendu. Une data card dit d'où viennent les données, ce qu'elles contiennent, ce qu'on a le droit d'en faire et ce qu'elles ne représentent pas : un recruteur la lit pour savoir si tu as regardé tes données avant de les entraîner. -->

| | |
|---|---|
| **Source** | TODO : l'adresse de la page officielle (pas un miroir) |
| **Version figée** | TODO : la révision (commit du Hugging Face Hub, version Kaggle, version OpenML, date de l'archive) |
| **Licence** | TODO : le nom exact (CC BY 4.0, MIT, CC0…) et le lien vers son texte |
| **Taille** | TODO : nombre d'exemples, de features ou d'images, poids des fichiers |
| **Fichiers et SHA-256** | TODO : un fichier par ligne, avec sa somme SHA-256 (64 caractères) |
| **Citation** | TODO : l'article ou la page à citer |

## Provenance

TODO : qui a créé ces données, quand, pourquoi, comment elles ont été collectées (capteurs, photos, formulaires, extraction d'un site), et qui les a annotées (experts, foule, règle automatique). Si tu ne le sais pas, dis-le : c'est une limite (29.5, question 6).

## Contenu

TODO : ce qu'est un exemple (une ligne, une image, un texte, une fenêtre de série, un épisode) ; les variables et leurs unités ; la cible et sa répartition ; les valeurs manquantes et leur codage ; les identifiants et les groupes (plusieurs exemples d'une même personne, d'un même objet, d'une même date) ; la période couverte. Les chiffres viennent de ton exploration (notebooks/01_exploration.ipynb), pas de la page de la source.

## Licence

TODO : la licence, ce qu'elle permet (un dépôt public, un usage commercial, une modification) et ce qu'elle impose (citer, partager à l'identique) ; les données personnelles éventuelles et leur anonymisation ; ce que tu ne peux pas vérifier toi-même (le consentement des personnes, les droits des images collectées sur le web).

## Usage prévu

TODO : la tâche de ton projet (docs/cadrage.md), et les usages que tu déconseilles avec ces données (une décision sur des personnes, un autre pays, une autre période…).

## Limites et biais

TODO : qui ou quoi est sous-représenté ; les défauts trouvés à l'audit (doublons, fuites, labels douteux, décalage entre périodes ou sources) et ce que tu en as fait ; ce que les données ne permettent pas de conclure.

## Téléchargement

TODO : la commande (`python scripts/download_data.py`), où vont les fichiers (`data/raw/`, jamais versionné), la vérification par SHA-256, la conduite à tenir si la source disparaît ou change (le script échoue sur une somme fausse : il ne continue jamais avec d'autres données).
