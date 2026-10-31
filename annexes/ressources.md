# Ressources

Sélection de ressources de qualité, **vérifiées** avant d'être citées (BIBLE §20). Chaque fiche de chapitre propose en plus 3 à 5 ressources ciblées ; les meilleures sont reprises ici.

## Le livre
- Andrew Glassner, *Deep Learning: From Basics to Practice*, 2 volumes, 2018.
- Code (notebooks) et figures du livre, licence MIT : [ressources de l'auteur](https://github.com/blueberrymusic/DeepLearningBook-Resources). Les dépôts officiels des notebooks ne sont plus en ligne (volume 2 vérifié le 2026-09-29, volume 1 le 2026-10-18) ; des copies, avec l'historique et la licence de l'auteur, existent : [bssrdf/DeepLearningBookCode-Volume1](https://github.com/bssrdf/DeepLearningBookCode-Volume1) et [bssrdf/DeepLearningBookCode-Volume2](https://github.com/bssrdf/DeepLearningBookCode-Volume2).
- L'auteur a depuis publié une version révisée en un volume, *Deep Learning: A Visual Approach* (2021).

## Documentation officielle (à garder sous la main)
| Outil | Lien |
|---|---|
| Python (tutoriel officiel, en français) | <https://docs.python.org/fr/3/tutorial/> |
| NumPy | <https://numpy.org/doc/2.1/> |
| pandas | <https://pandas.pydata.org/pandas-docs/version/2.2/> |
| Matplotlib | <https://matplotlib.org/3.10.0/> |
| scikit-learn (guide utilisateur) | <https://scikit-learn.org/1.6/user_guide.html> |
| PyTorch | <https://docs.pytorch.org/docs/2.11/> |
| Hugging Face | <https://huggingface.co/docs> |
| Gymnasium | <https://gymnasium.farama.org/v1.4.0/> |
| Google Colab | <https://colab.research.google.com/> |

Les liens suivent les versions figées du workbook (BIBLE §21) ; la documentation la plus récente est à l'adresse `…/stable/` de chaque projet.

## Outils d'apprentissage
- **Git** : le livre *Pro Git*, en français et gratuit : <https://git-scm.com/book/fr/v2>
- **Anki** (flashcards à répétition espacée) : <https://apps.ankiweb.net/>

## Par partie
*(complété à chaque chapitre : cours, articles fondateurs, vidéos)*

| Partie | Ressources |
|---|---|
| 0 · Prérequis | |
| I · Fondations | |
| II · Concepts | Ch. 8 : S. Raschka, [« Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning »](https://arxiv.org/abs/1811.12808), 2018 (hold-out, bootstrap, validation croisée imbriquée, tests de comparaison) ; S. Kapoor et A. Narayanan, [« Leakage and the reproducibility crisis in machine-learning-based science »](https://doi.org/10.1016/j.patter.2023.100804), *Patterns*, 2023 ; scikit-learn, [« Common pitfalls and recommended practices »](https://scikit-learn.org/1.6/common_pitfalls.html). |
| III · ML classique | |
| IV · Réseaux | |
| V · Architectures | Ch. 21 : [notes du cours CS231n sur les CNN](https://cs231n.github.io/convolutional-networks/) ; V. Dumoulin et F. Visin, [« A guide to convolution arithmetic for deep learning »](https://arxiv.org/abs/1603.07285) ; K. He et coll., [« Deep Residual Learning for Image Recognition »](https://arxiv.org/abs/1512.03385), 2015. Ch. 22 : C. Olah, [« Understanding LSTM Networks »](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), 2015. Ch. 23-24 : les tutoriels [« Learn the Basics »](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) de PyTorch ; A. Karpathy, [« A Recipe for Training Neural Networks »](https://karpathy.github.io/2019/04/25/recipe/), 2019 ; le [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) de Google. |
| VI · Génératif et RL | Ch. 25 : D. Kingma et M. Welling, [« Auto-Encoding Variational Bayes »](https://arxiv.org/abs/1312.6114), 2013 ; C. Doersch, [« Tutorial on Variational Autoencoders »](https://arxiv.org/abs/1606.05908), 2016. Ch. 26 : R. Sutton et A. Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), 2ᵉ éd., 2018 (en ligne) ; [Spinning Up in Deep RL](https://spinningup.openai.com/) d'OpenAI. Ch. 27 : I. Goodfellow et coll., [« Generative Adversarial Nets »](https://arxiv.org/abs/1406.2661), 2014, et le [tutoriel NIPS 2016](https://arxiv.org/abs/1701.00160). Ch. 28 : le tutoriel [« Neural Transfer Using PyTorch »](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html). Ch. 29 : T. Gebru et coll., [« Datasheets for Datasets »](https://arxiv.org/abs/1803.09010), 2018. |
| VII · Bonus | |
