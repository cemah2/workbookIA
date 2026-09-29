# MÉTHODE — Construire (et utiliser) le workbook Deep Learning avec Claude

Ce guide est pour toi. Il explique **comment piloter Claude, session après session**, pour obtenir le workbook le plus complet et le plus fiable possible. Les prompts à copier-coller sont dans `02_PROMPTS.md`, la spécification dans `01_BIBLE_WORKBOOK.md`.

---

## 1. Principes

1. **Une spécification unique, la bible.** Toutes les sessions suivent le même document : c'est ce qui garantit que le chapitre 27 ait la même structure, les mêmes conventions et les mêmes fils rouges que le chapitre 2.
2. **Un chapitre par conversation.** Claude travaille mieux avec un contexte « frais ». On ne génère jamais tout le workbook dans une seule conversation.
3. **Un dépôt GitHub comme mémoire.** Chaque session repart du dépôt (chapitres déjà faits, code mylearn, `PROGRESS.md`) et y pousse son travail. Rien ne se perd entre deux conversations, et Colab ouvre directement les notebooks depuis GitHub.
4. **Rien n'est livré sans avoir été exécuté.** Claude fait tourner les notebooks de solutions et les tests, et fait revérifier les calculs par un second agent indépendant.
5. **Génération « juste à temps » avec deux chapitres d'avance.** Tu étudies le chapitre N pendant que N+1 et N+2 existent déjà. Tes retours sur N (trop dur, trop long, pas clair) améliorent N+3 et les suivants.

## 2. Avant de commencer (une seule fois)

1. **Compte GitHub** : crée un dépôt **vide**, par exemple `dl-workbook`. Je te conseille de le rendre **public** : il servira de portfolio de reconversion, et Colab l'ouvre sans configuration. Les solutions y seront visibles, donc la discipline repose sur toi. Les PDF du livre n'y seront jamais.
2. **Connecte GitHub à Claude** si ce n'est pas fait. Claude te le demandera lors de la session 1 s'il ne peut pas accéder au dépôt.
3. **Prépare les pièces jointes** :
   - `01_BIBLE_WORKBOOK.md` et `00_METHODE.md` (ces fichiers) ;
   - le PDF du **Volume 1** (chapitres 1 à 19) et celui du **Volume 2** (chapitres 20 à 29).
4. Utilise **le modèle Claude le plus puissant disponible** pour les sessions de génération. Pour le mode tuteur (P8), n'importe quel modèle suffit.

**Plan B si GitHub ne fonctionne pas** : Claude livre à la fin de chaque session un `.zip` du dépôt complet. Tu joins le dernier zip au début de la session suivante. C'est plus lourd, mais ça marche.

## 3. Déroulé des sessions

| # | Session | Prompt | Pièces jointes | Ton rôle après la session |
|---|---|---|---|---|
| 1 | Mise en place du dépôt, des outils et des datasets | **P0** | Bible, Méthode, Vol. 1 et 2 | Ouvrir `00_setup/demo.ipynb` dans Colab et vérifier qu'il tourne ; répondre aux questions de Claude |
| 2 | Syllabus détaillé : tous les exercices planifiés, signatures mylearn, parcours | **P1** | Vol. 1 et 2 | **Relire `docs/SYLLABUS.md`** (c'est le moment de tout changer) ; demander des modifications avec P6 si besoin |
| 3–4 | Chapitres 0A et 0B (prérequis) | **P2** | aucune | Commencer à étudier 0A |
| 5 → ~40 | Chapitres 1 à 29, un par session. Les chapitres denses (18, 21, 22, 23-24, 26) peuvent prendre deux sessions | **P2** (+ **P11** si la session est interrompue) | le volume du chapitre | Étudier, puis envoyer ton retour avec **P7** |
| après chaque partie | Checkpoint de partie : examen blanc, synthèse, mini-projet | **P3** | le volume concerné | Faire l'examen **en conditions réelles** avant de lire le corrigé |
| après les parties II, IV, VI | Audit qualité indépendant | **P5** | aucune | Valider les corrections majeures proposées |
| ~41–48 | Chapitres bonus B1 à B8 | **P4** | aucune | idem |
| dernière | Finalisation : index des notions, deck Anki global, README final, projet final | **P10** | aucune | 🎓 |

Compte **environ 50 sessions de génération** au total.

## 4. Pendant chaque session de génération

- Colle le prompt en remplaçant les `{{VARIABLES}}`.
- Claude affiche une **liste de tâches**, ce qui te permet de suivre l'avancement sans rester devant l'écran.
- S'il pose une question et que tu ne réponds pas, il prend le choix le plus raisonnable et le note dans `PROGRESS.md`.
- **Si la session s'interrompt** (limite d'usage, déconnexion), ouvre une nouvelle conversation avec **P11** : Claude reprend exactement à l'étape notée dans `PROGRESS.md`.
- À la fin, Claude envoie un **rapport** : nombre d'exercices par type, temps d'étude estimé, points à valider sur Colab, écarts par rapport au syllabus.

### Ta vérification rapide après chaque chapitre (5 minutes)
1. Ouvre `03_notebook.ipynb` dans Colab et fais « Exécuter tout » : ça doit aller jusqu'au bout, avec des « ⏳ pas encore fait ».
2. Parcours `02_exercices.md` : les énoncés sont-ils compréhensibles pour toi ?
3. Si le rapport mentionne des cellules « à valider sur Colab », exécute `05_solutions.ipynb` en `FAST_MODE = False` avec le GPU activé et signale le résultat avec P6.
4. Si quelque chose ne va pas, utilise **P6** (correction ciblée).

## 5. Comment étudier avec le workbook

Routine conseillée pour chaque chapitre :

1. **`python tools/start_chapter.py N`** : copie le notebook et les squelettes mylearn dans `mon_travail/`, ton espace personnel où Claude n'écrit jamais.
2. **Lire la fiche** (`01_fiche.md`) : objectifs et vue d'ensemble, 10 minutes.
3. **Lire le chapitre du livre** en suivant le guide de lecture de la fiche.
4. **Quiz 🧠 et rappels 🔁**, sans le livre.
5. **Exercices papier ✏️ et ∂**, en écrivant tes réponses dans `06_mes_reponses.md`.
6. **Notebook** : les 🔮 d'abord à l'instinct, puis le reste. Règle des 15 minutes : bloqué 15 minutes, tu ouvres l'indice 1 ; encore 15 minutes, l'indice 2, et ainsi de suite. Tu peux aussi ouvrir une conversation **tuteur (P8)**, qui te guide sans donner la réponse.
7. **Corriger** : solutions et `pytest`. Tu peux faire corriger ton travail par Claude avec **P9**.
8. **Réponses 💼 à voix haute**, comme en entretien.
9. **Flashcards** : importe `flashcards.csv` dans Anki, et fais tes révisions chaque jour, c'est le plus rentable.
10. **Journal et auto-évaluation** dans `suivi/`.
11. **Retour à Claude (P7)**, pour calibrer les chapitres suivants.

**Règle d'or** : ne lis jamais une solution avant d'avoir *vraiment* essayé. L'effort de récupération, c'est ce qui fait apprendre.

## 6. Mises à jour de la bible

Si tu veux changer une règle globale (plus d'exercices de maths, moins de quiz, un nouveau type d'exercice…), fais-le avec P6 en écrivant « modifie la bible ». Claude met à jour `docs/BIBLE.md`, note la décision au §22 et indique quels chapitres déjà générés sont concernés.

## 7. Problèmes fréquents

| Symptôme | Que faire |
|---|---|
| Claude recopie de longs passages du livre | P6 : « viole la règle §3 de la bible, réécris ces passages de manière originale » |
| Un exercice utilise une notion pas encore vue | P6, en citant la règle §15.2 |
| Un notebook est trop lent sur ton ordinateur | P6 : réduire le FAST_MODE ou passer la cellule en 🚀 |
| Le style dérive d'un chapitre à l'autre | P5 (audit), puis P6 |
| Tu trouves un chapitre trop dur ou trop facile | P7 : le calibrage s'applique aux chapitres suivants |
| Le dépôt n'est plus accessible | Plan B (zip) |
