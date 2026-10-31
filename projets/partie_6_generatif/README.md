# Mini-projet MP6 — Banc d'essai génératif : VAE contre GAN sur Fashion-MNIST

> Le mini-projet du checkpoint VI, à faire après l'examen blanc (environ 10 h). Il réunit les chapitres 25 et 27 (et le 29 pour la data card) dans un vrai banc d'essai : un VAE convolutif et un DCGAN entraînés sur les mêmes images, puis jugés par un protocole d'évaluation écrit **avant** les expériences. Un projet de portfolio : des résultats reproductibles (graines, configuration), une galerie, un README et une fiche modèle qui discutent les limites et les usages responsables.

| | |
|---|---|
| **Problème** | générer des images de vêtements 28 × 28 en niveaux de gris, et dire honnêtement lequel de deux générateurs le fait le mieux |
| **Données** | Fashion-MNIST (Xiao, Rasul et Vollgraf, 2017 ; licence MIT ; data card : `data/cards/fashion_mnist.md`) : les 60 000 images d'entraînement, coupées une fois pour toutes en 50 000 pour l'entraînement et 10 000 pour la validation (`data.py`, fourni) ; le test n'est pas utilisé |
| **Méthode imposée** | PyTorch dans le package `genbench/` ; un VAE convolutif et un DCGAN selon les règles de Radford (un GAN conditionnel en option) ; le protocole d'évaluation fixé avant les expériences ; trois graines par générateur ; ta fonction `interpolate_latents` (« Ta librairie », plus bas) |
| **Prérequis** | chapitres 25 (VAE) et 27 (GAN, distance de Fréchet) ; le checkpoint complet (ch. 23) ; la data card (ch. 29) |
| **Évaluation** | sur 20, avec la grille ci-dessous |

## Le contexte

Juger un générateur est plus difficile que juger un classifieur : il n'y a pas de bonne réponse à comparer. Une loss qui baisse ne dit rien (celle d'un GAN oscille par construction), une belle grille d'images peut cacher un générateur qui ne sait faire que des pantalons, ou qui recopie ses images d'entraînement. Un recruteur qui ouvre ton dépôt cherche donc moins de jolies images qu'une **méthode** : des mesures choisies à l'avance, plusieurs graines, des contrôles de l'effondrement de mode et de la mémorisation, et un texte qui dit ce que les chiffres ne disent pas. Ce projet t'y entraîne sur deux générateurs que tu connais (ch. 25 et 27), assez petits pour tourner sur un CPU en mode rapide.

## L'architecture du code

Le package `genbench/` a trois modules ; leurs docstrings sont le contrat (le kit de départ les donne).

| Module | Contenu | Ce qu'il appelle dans ta librairie |
|---|---|---|
| `models.py` | le classifieur de référence (`Classifier`, ses `features`), le VAE convolutif (`ConvVAE` : `encode`, `reparameterize`, `decode`, `forward`), le DCGAN (`generator_body`, `discriminator_body`, `Generator` et `Discriminator`, conditionnels en option, `init_dcgan_weights`) ; fournie : `n_parameters` | |
| `train.py` | `set_seed`, `save_config`, `train_classifier`, `vae_loss` et `train_vae`, `gan_step` (loss non saturante), `save_checkpoint` et `load_checkpoint`, `train_gan` (journal `log.csv`, checkpoint complet, reprise exacte) ; fournies : `images_to_tensor`, `library_versions` | |
| `evaluate.py` | `classifier_features`, `predict_proba`, `accuracy`, `decode_codes`, `sample_images`, `kl_per_latent`, `class_report` (effondrement de mode), `nearest_neighbors` (mémorisation), `interpolate` (la galerie) ; fournies : `frechet_distance`, `image_grid` | `vae.interpolate_latents` ; `gen_metrics.frechet_distance_from_features` si tu l'as écrite |

Le fichier `data.py` (fourni) charge Fashion-MNIST et fait le découpage. Les deux fonctions qui se servent de `mylearn` l'importent à l'intérieur de leur corps : le package se recharge sans ta librairie, et `frechet_distance` calcule alors la même formule avec SciPy.

## Le protocole d'évaluation

Il s'écrit **avant** les expériences, dans ton README (étape MP6.1), et ne change plus ensuite ; s'il doit changer, le README le dit et explique pourquoi.

1. **L'instrument** : un classifieur de référence (un petit CNN) entraîné sur les images d'entraînement ; au moins 90 % d'accuracy de validation en mode complet. Ses **features** (les 128 valeurs de l'avant-dernière couche) servent à toutes les mesures, comme celles d'Inception pour le FID (ch. 27). Avant de s'en servir, on vérifie qu'il voit la différence entre des images réelles et des images floutées ou bruitées.
2. **La distance de Fréchet** entre les features de `n_eval` images générées et celles de `n_eval` images de **validation**, que ni le classifieur ni les générateurs n'ont vues. Le même nombre d'images pour chaque modèle (la distance en dépend), à côté du **plancher** : la distance entre deux échantillons d'images réelles, ce que vaudrait un générateur parfait.
3. **Trois graines** d'entraînement par générateur, écrites dans `config.json` ; on rapporte la moyenne et l'écart entre graines, jamais la meilleure seule.
4. **L'effondrement de mode** : la répartition des classes que le classifieur prédit pour les images générées (son entropie, la plus grande part), comparée à celle des images réelles. Une classe absente ou envahissante est un effondrement partiel.
5. **La mémorisation** : pour chaque image générée, la distance (entre pixels) à sa plus proche image d'entraînement, comparée à la même distance pour de vraies images de validation. Une **quasi-copie** est plus proche de l'entraînement que le quantile `copy_quantile` (1 % par exemple) des images de validation ; une part de quasi-copies bien au-dessus de ce quantile signale une mémorisation. La distance de Fréchet ne la voit pas : un générateur qui recopierait l'entraînement aurait une distance proche du plancher.
6. **Le réglage de β** du VAE se fait par une règle écrite avant la comparaison (par exemple : la plus petite distance de Fréchet, avec une graine de tirage réservée à ce choix), sur la graine 0 ; les mesures finales tirent d'autres images.
7. **L'inspection visuelle** : des grilles pour un bruit fixé, les plus proches voisins des échantillons les plus suspects, des interpolations. Elle complète les nombres, elle ne les remplace pas.

## Les contraintes

- **Deux modes** : le mode rapide (`FAST_MODE = True`) tient en une dizaine de minutes sur un CPU à deux cœurs, sur des sous-ensembles (10 000 images pour les générateurs, 2 000 images par mesure) ; le mode complet (les 50 000 images, 10 000 images par mesure) se lance sur un GPU de Colab (🚀). Les fichiers du mode rapide sont suffixés `_rapide` et n'écrasent jamais ceux du mode complet.
- **Un package réutilisable** : le notebook n'a pas de boucle d'entraînement à lui ; il appelle `genbench`.
- **Reproductible** : les graines et la configuration dans `config.json`, les versions dans `config.json` et `results.json` ; la reprise d'un GAN interrompu donne les mêmes poids, au bit près.
- **Lisible** : des docstrings, des annotations de type, des tests dont le message dit ce qui était attendu.

## Les livrables

1. Le notebook `mp6_generatif.ipynb` : propre, « Tout exécuter » sans erreur (moins de 10 minutes en mode rapide sur CPU), avec tes observations.
2. Le package `genbench/` (`models.py`, `train.py`, `evaluate.py`) et `test_genbench.py`, au moins **trois** tests, qui attrapent au moins deux des cinq mutants de l'étape MP6.7.
3. `config.json` (écrit avant les expériences) et `results.json` (les métriques par modèle et par graine, le choix de β, la mémorisation, les historiques d'entraînement).
4. Les figures (`figures/`) : les courbes d'entraînement du VAE et du GAN, les grilles d'échantillons, les interpolations slerp, les classes générées, les plus proches voisins, la distance de Fréchet par graine.
5. Le **README de portfolio** (`README.md` : objectif, protocole, méthode, résultats chiffrés, galerie légendée, limites, reproduction en une commande) et la **fiche modèle** (`MODEL_CARD.md` : usage prévu, données, métriques, limites, risques et usages responsables) : leurs modèles sont dans le kit.

## Les étapes

| Étape | Ce que tu fais | Ce que tu produis | ⏱️ |
|---|---|---|---|
| MP6.1 | Cadrage : protocole d'évaluation écrit, data card relue, configuration et graines | « L'objectif » et « Le protocole d'évaluation » du README, `set_seed`, `save_config`, `config.json`, `figures/donnees.png` | 60 min |
| MP6.2 | Classifieur de référence Fashion-MNIST (features pour la distance de Fréchet, accuracy ≥ 90 %) | `Classifier`, `train_classifier`, `classifier_features`, `predict_proba`, `accuracy`, les repères de la distance | 60 min |
| MP6.3 | VAE convolutif : entraînement, réglage de β, courbes reconstruction/KL | `ConvVAE`, `vae_loss`, `train_vae`, `sample_images`, `kl_per_latent`, la règle de β, deux figures | 120 min |
| MP6.4 | DCGAN (option cGAN) : entraînement journalisé, checkpoints, reprise | `generator_body`, `discriminator_body`, `Generator`, `Discriminator`, `gan_step`, `train_gan` et le checkpoint, `log.csv`, deux figures | 150 min |
| MP6.5 | Évaluation : distance de Fréchet sur 3 graines, classes générées, plus proches voisins | `class_report`, `nearest_neighbors`, `results.json`, trois figures | 90 min |
| MP6.6 | Galerie : grilles, interpolations sphériques, échantillons conditionnels | `interpolate`, deux ou trois figures légendées | 45 min |
| MP6.7 | README, model card, tests pytest et nettoyage du dépôt | `test_genbench.py`, `README.md`, `MODEL_CARD.md` | 75 min |

## La grille d'évaluation (sur 20)

| Critère | Points | Tous les points | La moitié des points |
|---|---|---|---|
| **Reproductibilité** | 3 | graines et configuration écrites avant les expériences (`config.json`), versions notées, notebook exécutable de bout en bout, reprise exacte vérifiée | des graines fixées mais une configuration incomplète, ou un notebook qui ne tourne qu'avec des cellules relancées à la main |
| **VAE correct et analysé** | 4 | une loss à la bonne échelle (reconstruction sommée sur les pixels, comme la KL), courbes de reconstruction et de KL lues, latents actifs comptés, β choisi par une règle écrite avant la comparaison | un VAE qui génère, mais un β choisi sans règle ou des courbes sans commentaire |
| **GAN stable, journalisé, repris** | 4 | règles de Radford, loss non saturante, faux détachés, `log.csv` et courbes commentées, un entraînement interrompu puis repris depuis `last.pt` (deux réseaux, deux optimiseurs, états aléatoires) | un GAN qui génère, mais sans checkpoint complet ou sans journal |
| **Évaluation rigoureuse** | 4 | protocole fixé à l'avance et suivi, trois graines par générateur avec leur écart, plancher et repères de la distance, effondrement de mode et mémorisation vérifiés et interprétés | une seule graine, ou une distance de Fréchet sans repère ni contrôle de mémorisation |
| **Galerie** | 2 | des grilles pour un bruit fixé, des interpolations slerp (et la ligne droite à côté), des échantillons conditionnels si le cGAN est fait, chaque figure légendée | des figures sans légende |
| **README et model card** | 3 | résultats chiffrés (moyenne et écart sur trois graines), limites honnêtes (petits modèles, features d'un petit CNN, distance non comparable aux FID publiés), usages responsables et risques des images générées | un README sans chiffres, ou une fiche modèle sans limites ni risques |

## Les extensions (pour aller plus loin)

- **Une diffusion latente miniature** : un petit modèle de diffusion dans l'espace latent du VAE, comparé aux deux autres avec le même protocole (préparation du bonus B5).
- **Deux extracteurs de features** : la distance de Fréchet sur les features de ton classifieur et sur celles d'un autre réseau (un autre CNN, ou un réseau pré-entraîné sur des images agrandies) ; le classement des générateurs change-t-il ? Discute la dépendance de la mesure.
- **Une démo interactive** (Gradio) publiée sur un Space Hugging Face, avec la fiche modèle.
- **Un volet apprentissage par renforcement** : un agent de morpion entraîné en self-play (ch. 26), évalué contre minimax, présenté dans le même dépôt.

## Ta librairie : les fonctions que le projet appelle

Le projet appelle une fonction de ta librairie `mylearn` que les quatre parcours écrivent, et une seconde si tu l'as écrite. Avant l'étape qui s'en sert, vérifie qu'elle passe ses tests : le repli sur la référence ne remplace qu'un module **absent** de `mon_travail/mylearn/`. Dans un module que tu as copié, une fonction que tu n'as pas encore écrite lève `NotImplementedError`, et l'étape s'arrête.

| Étape | Fonction de `mylearn` | Exercice qui l'écrit |
|---|---|---|
| MP6.6 | `vae.interpolate_latents` (avec `method="slerp"`) | 25.20 |
| MP6.2 à MP6.5 | `gen_metrics.frechet_distance_from_features`, si tu l'as écrite (parcours maths et code) ; sinon, `genbench.frechet_distance` calcule la même formule avec SciPy | 27.22 |

Pour vérifier une fonction, lance depuis la racine du dépôt les tests de son chapitre, filtrés sur son nom :

```bash
python -m pytest tests/test_ch25_vae.py -q -k "interpolate"
```

## Démarrer

```bash
python tools/start_chapter.py CP6
```

copie le kit de départ dans `mon_travail/projets/partie_6_generatif/` : `mp6_generatif.ipynb` (les sept étapes), le package `genbench/` (les classes, les signatures et les docstrings ; les fonctions à écrire lèvent `NotImplementedError`), `test_genbench.py` (un test d'exemple, à compléter), `data.py` (fourni : Fashion-MNIST et le découpage), `conftest.py` (fourni : pytest trouve ta librairie), et les modèles de `README.md` et de `MODEL_CARD.md`. Travaille **dans ce dossier-là** ; les tests se lancent depuis la racine du dépôt :

```bash
python -m pytest mon_travail/projets/partie_6_generatif -q
```

Fashion-MNIST se charge avec `data.load_split()` (environ 30 Mo la première fois, puis le cache du workbook). Pour le mode complet sur Colab : *Exécution → Modifier le type d'exécution → GPU*, puis `FAST_MODE = False` dans la cellule de setup ; enregistre ensuite ta copie et les fichiers produits (`config.json`, `results.json`, `figures/`) dans ton dépôt. Les checkpoints du GAN (`runs/`) sont lourds et se régénèrent : ils ne se commitent pas.

Tu peux faire corriger ton projet par Claude avec le prompt P9 : il lit ton dossier, sans jamais y écrire, et le note avec cette grille.

## La solution de référence

Le dossier [`solution/`](solution/) contient un projet complet : le package, ses tests, le notebook exécuté (en mode rapide), la configuration et les résultats des deux modes, avec un README de portfolio et une fiche modèle. **Ne l'ouvre qu'après avoir fini le tien** : c'est une façon de faire parmi d'autres, et tes chiffres n'ont pas à être les mêmes.
