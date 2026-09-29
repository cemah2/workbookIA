# Fiches des datasets (*data cards*)

Une fiche par dataset fil rouge (BIBLE §9) : provenance, licence, taille, variables, biais et limites, chapitres qui l'utilisent. Au ch. 29, tu écriras la fiche du dataset de ton projet final sur le même modèle.

| Dataset | Chargement | Stockage | Fiche |
|---|---|---|---|
| Palmer Penguins (+ version brute) | `wb.datasets.load_penguins()` | versionné (`data/penguins.csv`, `data/penguins_raw.csv`) | [penguins.md](penguins.md) |
| California Housing | `wb.datasets.load_california()` | scikit-learn, copie CSV versionnée | [california_housing.md](california_housing.md) |
| MNIST | `wb.datasets.load_mnist()` | versionné (`data/mnist.npz`, 11,5 Mo) | [mnist.md](mnist.md) |
| Fashion-MNIST | `wb.datasets.load_fashion_mnist()` | téléchargé (≈ 30 Mo) | [fashion_mnist.md](fashion_mnist.md) |
| CIFAR-10 | `wb.datasets.load_cifar10()` | téléchargé (≈ 170 Mo) | [cifar10.md](cifar10.md) |
| Holmes (texte anglais) | `wb.datasets.load_holmes()` | versionné (`data/text/`) | [holmes.md](holmes.md) |
| Verne (texte français) | `wb.datasets.load_verne()` | versionné (`data/text/`) | [verne.md](verne.md) |
| Taches solaires mensuelles | `wb.datasets.load_sunspots()` | versionné (`data/sunspots_monthly.csv`) | [sunspots.md](sunspots.md) |
| Données synthétiques | `wb.synth.*` | générées | [synthetic.md](synthetic.md) |
| Environnements RL | `wb.datasets.make_env()` | code | [rl_envs.md](rl_envs.md) |

Les fichiers versionnés sont reconstruits depuis leurs sources officielles par `python tools/build_datasets.py` (téléchargés le 2026-09-29). Les téléchargements vont dans `data/downloads/` en local et dans `/content/wb_cache` sur Colab ; ils ne sont jamais versionnés.
