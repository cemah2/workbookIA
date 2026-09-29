# Glossaire français ↔ anglais

Règle du workbook (BIBLE §5) : on garde le terme anglais quand c'est l'usage professionnel. À sa première apparition dans un chapitre, un terme est écrit « terme retenu (*autre langue*) », puis seul le terme retenu est utilisé. Chaque nouveau terme d'un chapitre est ajouté ici.

**Colonne « Retenu »** : le terme utilisé dans le workbook. **Ch.** : chapitre où il est introduit (complété au fil de la génération).

## Termes gardés en anglais

| Retenu | En français | Définition courte | Ch. |
|---|---|---|---|
| dataset | jeu de données | ensemble d'exemples utilisés pour entraîner ou évaluer un modèle | |
| feature | caractéristique, variable explicative | une information mesurée sur chaque exemple (ex. la longueur du bec d'un manchot) | |
| label | étiquette | la réponse attendue pour un exemple (ex. l'espèce du manchot) | |
| batch | lot | groupe d'exemples traités ensemble en une étape de calcul | |
| mini-batch | mini-lot | petit lot (quelques dizaines d'exemples) utilisé à chaque mise à jour des poids | |
| epoch | époque | un passage complet sur tout le dataset d'entraînement | |
| learning rate | taux d'apprentissage | taille des pas faits à chaque mise à jour des poids | |
| loss | perte, fonction de coût | nombre qui mesure à quel point les prédictions sont mauvaises ; on cherche à le minimiser | |
| pipeline | chaîne de traitement | suite d'étapes (préparation, modèle…) enchaînées automatiquement | |
| framework | cadriciel | bibliothèque qui fournit la structure d'un programme (ex. PyTorch) | |
| fine-tuning | réglage fin, affinage | réentraîner un peu un modèle déjà entraîné sur une nouvelle tâche | |
| embedding | plongement, représentation vectorielle | vecteur de nombres qui représente un objet (mot, image…) | |
| dropout | abandon | désactiver au hasard des neurones pendant l'entraînement pour limiter l'overfitting | |
| pooling | agrégation, sous-échantillonnage | résumer une zone d'une image par un seul nombre (maximum, moyenne) | |
| padding | remplissage, marge | ajouter des valeurs (souvent des zéros) autour d'une donnée pour garder sa taille | |
| stride | pas | décalage entre deux positions successives d'un filtre de convolution | |
| kernel | noyau, filtre | petite grille de poids que l'on fait glisser sur une image (CNN) | |
| token | jeton, unité lexicale | morceau de texte (mot, sous-mot, caractère) traité par un modèle de langage | |
| prompt | invite, instruction | texte donné en entrée à un modèle de langage | |
| overfitting | surapprentissage, surajustement | le modèle apprend par cœur l'entraînement et généralise mal | |
| underfitting | sous-apprentissage | le modèle est trop simple pour capter la structure des données | |
| benchmark | banc d'essai, référence | jeu de test standard pour comparer des méthodes | |
| notebook | carnet | document qui mêle texte, code exécutable et résultats (Jupyter, Colab) | |
| accuracy | exactitude, taux de bonnes réponses | proportion de prédictions correctes | |
| precision | précision (ambigu) | parmi les exemples prédits positifs, proportion vraiment positive | |
| recall | rappel, sensibilité | parmi les exemples vraiment positifs, proportion retrouvée | |
| F1-score | score F1 | moyenne harmonique de la precision et du recall | |

> ⚠️ **accuracy, precision et recall restent en anglais** : en français, « précision » peut désigner l'une ou l'autre notion.

## Termes dits en français

| Retenu | En anglais | Définition courte | Ch. |
|---|---|---|---|
| réseau de neurones | neural network | modèle fait de couches de neurones artificiels reliés entre eux | |
| couche | layer | ensemble de neurones qui reçoivent les mêmes entrées | |
| poids | weights | coefficients appris qui multiplient les entrées d'un neurone | |
| biais | bias | nombre appris ajouté à la somme pondérée d'un neurone (à ne pas confondre avec un biais statistique) | |
| neurone | neuron, unit | calcule une somme pondérée de ses entrées puis applique une fonction d'activation | |
| fonction d'activation | activation function | fonction non linéaire appliquée à la sortie d'un neurone (ReLU, sigmoïde…) | |
| descente de gradient | gradient descent | méthode qui ajuste les poids par petits pas dans la direction qui fait baisser la loss | |
| rétropropagation | backpropagation | algorithme qui calcule efficacement le gradient de la loss par rapport à tous les poids | |
| entraînement | training | phase où le modèle ajuste ses poids à partir des données | |
| validation croisée | cross-validation | évaluer un modèle en le réentraînant sur plusieurs découpages des données | |
| apprentissage supervisé | supervised learning | apprendre à partir d'exemples étiquetés | |
| apprentissage non supervisé | unsupervised learning | trouver une structure dans des données sans étiquettes | |
| apprentissage par renforcement | reinforcement learning | apprendre par essais et erreurs grâce à des récompenses | |
| matrice de confusion | confusion matrix | tableau qui croise les classes réelles et les classes prédites | |

## Termes ajoutés au fil des chapitres

| Retenu | Autre langue | Définition courte | Ch. |
|---|---|---|---|
| | | | |
