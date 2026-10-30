# Le dataset piégé des fruits (`wb.datasets.make_trap_dataset`)

**Provenance.** Généré par le workbook (ch. 29), avec une graine : `make_trap_dataset(seed=2916, n=1200)`. Aucune donnée réelle : les mesures sont tirées de lois normales autour de valeurs plausibles pour trois fruits.

**Contenu.** Un `DataFrame` pandas de 1 230 lignes et 11 colonnes : `id_lot`, `poids_g`, `diametre_cm`, `sucre` (°Brix), `acidite`, `couleur_r`, `couleur_v`, `pays`, `code_qualite`, `fruit` (le label : pomme, poire ou coing) et `split` (« train » ou « test »).

**Usage prévu.** S'exercer à l'audit d'un dataset (🏆 29.16) : le tableau contient **huit défauts plantés**, du genre qu'on trouve dans les données réelles (doublons, fuites, valeurs manquantes codées par un nombre, unités incohérentes, labels mal écrits, classe très rare…). La liste exacte n'est donnée que par `trap_dataset_answers(seed, n)`, que le notebook n'appelle qu'au moment de la vérification : ne la lis pas avant d'avoir cherché.

**Licence.** Code du workbook ; données générées, sans droits de tiers.

**Limites.** Les lois sont simplistes (mesures indépendantes, aucune variété ni saison) : un modèle qui marche ici ne dit rien des vrais fruits. Avec une autre graine ou une autre taille, les nombres de défauts changent, pas leur nature.
