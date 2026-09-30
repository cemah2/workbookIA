# Taches solaires mensuelles (série temporelle réelle)

| | |
|---|---|
| **Fichier** | `data/sunspots_monthly.csv` (0,11 Mo) |
| **Chargement** | `wb.datasets.load_sunspots(definitive_only=False)` → DataFrame |
| **Source** | WDC-SILSO, Observatoire royal de Belgique, Bruxelles : nombre total de taches solaires, moyenne mensuelle, version 2.0 (`SN_m_tot_V2.0.csv`) |
| **Période** | janvier 1749 → août 2026 : 3 332 mois |
| **Licence** | **CC BY-NC 4.0** : usage non commercial, attribution obligatoire |
| **Citation** | *Source: WDC-SILSO, Royal Observatory of Belgium, Brussels*, DOI : 10.24414/qnza-ac80 |
| **Téléchargé le** | 2026-09-29 |

## Pourquoi ce choix (décision de la session 1)
Série réelle, légère, longue (plus de 3 300 points), avec un cycle d'environ 11 ans dont l'amplitude varie : assez régulière pour qu'un RNN apprenne quelque chose, assez irrégulière pour que la prévision reste un vrai défi. C'est aussi un classique des tutoriels de prévision. Elle est complétée par un sinus bruité synthétique (`wb.synth.noisy_sine`) pour les expériences contrôlées.

## Variables
| Colonne | Description |
|---|---|
| `date` | premier jour du mois (ajoutée au chargement) |
| `year`, `month` | année et mois |
| `decimal_year` | milieu du mois en année décimale |
| `sunspots` | nombre moyen de taches solaires du mois (≥ 0) |
| `std` | écart-type des observations journalières (vide avant 1818) |
| `n_obs` | nombre d'observations utilisées (vide avant 1818) |
| `definitive` | `False` pour les derniers mois, encore provisoires |

## Biais et limites
- Les premières décennies (XVIIIᵉ siècle) reposent sur peu d'observations : bruit beaucoup plus fort, pas d'incertitude publiée.
- Les derniers mois sont **provisoires** et peuvent être révisés : `definitive_only=True` pour les exclure.
- Licence non commerciale : parfait pour apprendre et pour un portfolio, pas pour un produit vendu.

## Chapitres
1 (tracer, lisser, repérer le cycle), 22 (RNN, prévision), 24 (PyTorch en pratique), B7 (du notebook à la production).
