# Questions annotées sur Holmes et Verne (RAG, bonus B4)

| | |
|---|---|
| **Fichiers** | `data/qa/holmes_qa.json` (13 Ko, UTF-8) et `data/qa/verne_qa.json` (4 Ko, UTF-8) |
| **Chargement** | `wb.rag.load_qa("holmes")` (30 questions), `wb.rag.load_unanswerable()` (5), `wb.rag.load_qa("verne")` (10) → listes de dictionnaires ; `wb.rag.documents(nom)` et `wb.rag.reference_chunks(nom)` donnent le corpus et son découpage de référence |
| **Source** | questions, réponses et extraits écrits pour le workbook le 2 novembre 2026, à partir du texte de `wb.datasets.load_holmes()` et de `wb.datasets.load_verne()` (Project Gutenberg, eBooks n° 1661 et 800 ; voir [holmes.md](holmes.md) et [verne.md](verne.md)) ; aucun jeu de questions existant n'a été repris |
| **Licence** | questions, réponses et annotations écrites pour le workbook (aucun emprunt à un jeu existant), réutilisables pour apprendre ; les deux textes sont dans le domaine public |
| **Empreintes** | SHA-256 de `data/qa/holmes_qa.json` : `83f44f1020e9d9f4337fd0b7169e0b0af01164d8a3e2606ed7d99f8399a9e402` ; SHA-256 de `data/qa/verne_qa.json` : `d7e2fc8d3b8050ef980b64801dd4140927b54379a7bf180241a4461a8a4eaddb` |

## Contenu
- **Holmes, 30 questions factuelles** en anglais sur les douze nouvelles de *The Adventures of Sherlock Holmes*, deux ou trois par nouvelle. Chaque question a un identifiant (`H01` à `H30`), sa **réponse** courte, des **variantes** acceptées (`aliases`), son **extrait source** (`evidence` : au plus 21 mots du texte, blancs réduits à une espace, qui contiennent la réponse), la position de cet extrait dans le texte chargé (`start`, `end`, en caractères), le numéro et le titre de sa nouvelle (`story`, `story_title`, l'ordre de `wb.text.holmes_stories()`), et le drapeau `paraphrased`.
- **10 questions paraphrasées** (`H21` à `H30`) : aucun de leurs mots-clés (mots pleins, à la racine près) n'apparaît dans les 140 mots qui précèdent ou suivent leur extrait ; les 20 autres partagent au moins un mot-clé avec ce texte. Les deux propriétés sont vérifiées par un test.
- **5 questions sans réponse** dans les nouvelles (`U01` à `U05`), hors des 30 : trois portent sur ce qui n'existe pas dans le recueil (`kind: "absent"`), deux reposent sur un présupposé faux (`"false_premise"`) ; la bonne réponse est « I don't know ».
- **Verne, 10 questions en français** (`V01` à `V10`) sur les chapitres I à VII du *Tour du monde en quatre-vingts jours*, avec les mêmes champs (`chapter` à la place de `story`) ; le corpus est le texte qui va du titre du chapitre I à celui du chapitre VIII (`corpus_start`, `corpus_stop`).
- Un chunk est **pertinent** pour une question quand il contient l'extrait entier (`wb.rag.relevant_ids`) : la vérité ne dépend pas du découpage. Avec des chunks de 120 mots et un recouvrement de 20 (le découpage de référence : 1 045 chunks pour Holmes, 93 pour Verne), chaque extrait est entier dans au moins un chunk.
- `wb.rag.FAST_IDS` : les 10 questions du mode rapide (7 directes, 3 paraphrasées).

## Utilisation
Bonus B4 : la recherche (🔮 B4.19 : les 10 paraphrasées ; 📦 B4.20 : BM25, recherche dense et hybride, recall@k et MRR sur les 30 questions et sur Verne), le RAG avec citations (📦 B4.21 : 10 questions en mode rapide, 30 en mode complet), et la seconde moitié du chapitre (évaluation, injection, défi).

## Biais et limites
- **Petit** : 30 questions donnent un intervalle de confiance à 95 % d'environ ± 18 points autour d'une exactitude de 50 % (fiche §204.5) ; deux systèmes qui diffèrent de 5 points ne sont pas départagés.
- Un seul annotateur : les réponses sont des faits du texte, sans interprétation, mais le choix des variantes acceptées reste un jugement (« Duke of Balmoral » ou « Balmoral »).
- **Contamination** : les nouvelles de Holmes sont célèbres et figurent dans les données d'entraînement des LLM ; un modèle peut répondre juste sans le passage. Les questions sur des détails mineurs (un tarif, un nombre de marches) le rendent moins probable, sans l'exclure.
