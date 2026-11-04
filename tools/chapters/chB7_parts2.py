"""End of part B (the service: B7.17 to B7.21) and part C (the operations: B7.22 to B7.25) of the chapter B7 notebook,
second half: statements and cells (used by build_chB7.py).

B7.17 to B7.21 are appended to PART_B.exercises (chB7_parts.py); part C is appended to PARTS. The answers of 🔮 B7.19,
of 📈 B7.22 and the speed constant of 🏆 B7.25 come from tools/chapters/history_chB7.json (entries "l19", "d22",
"c25").
"""

from __future__ import annotations

from itertools import combinations

from chapter_kit import Ex, Part, guarded

import chB7_code2 as C2
import chB7_code3 as C3
from chB7_parts import HISTORY, PART_B, PARTS, package_howto


def fill(text: str, values: dict) -> str:
    """Replace each «NAME» of a cell by its value; every placeholder must be filled."""
    for key, value in values.items():
        text = text.replace(f"«{key}»", str(value))
    if "«" in text:
        raise ValueError(f"unfilled placeholder in {text[text.index('«'):text.index('«') + 30]!r}")
    return text


def fr_num(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def letters_record(ex_id: str, answer: str, hints: dict, prediction: bool = False) -> str:
    """wb.record of a letter answer, one message per other letter (letters with the same message share it)."""
    grouped = {}
    for letter, message in hints.items():
        if letter != answer:
            grouped.setdefault(message, []).append(letter)
    value = lambda ls: f'"{ls[0]}"' if len(ls) == 1 else "{" + ", ".join(f'"{x}"' for x in ls) + "}"
    entries = ", ".join(f'"{message}": {value(ls)}' for message, ls in grouped.items())
    return f'wb.record("{ex_id}", "{answer}", mistakes={{{entries}}}' + (", prediction=True)" if prediction else ")")


def several_record(ex_id: str, answer: str, options: str, where: str) -> str:
    """wb.record of a set of letters: every other non-empty set, grouped in three messages that name no letter."""
    groups = {"missing": [], "extra": [], "both": []}
    for size in range(1, len(options) + 1):
        for combo in combinations(options, size):
            chosen = "".join(combo)
            if chosen == answer:
                continue
            missing, extra = set(answer) - set(chosen), set(chosen) - set(answer)
            groups["both" if missing and extra else "missing" if missing else "extra"].append(chosen)
    messages = {"missing": f"il manque au moins une fenêtre : relis {where}",
                "extra": f"au moins une des fenêtres choisies n'a pas sa place : relis {where}",
                "both": f"il manque au moins une fenêtre, et une autre n'a pas sa place : relis {where}"}
    entries = ", ".join(f'"{messages[k]}": {set(v)!r}' for k, v in groups.items() if v)
    return f'wb.record("{ex_id}", "{answer}", mistakes={{{entries}}})'


# ---------------------------------------------------------------------------------------------------------------------
# History: the predictions of B7.19, the answers of B7.22, the speed of B7.25
# ---------------------------------------------------------------------------------------------------------------------
L19 = HISTORY.get("l19", {"answers": {"a": "B", "b": "B"}, "direct_p50_default": 0.8})
D22 = HISTORY.get("d22", {"answers": {"a": "B", "b": "BD", "c": "C", "d": "C", "e": 0.16}, "psi_D_exact": 0.156})
C25 = HISTORY.get("c25", {"calib_speed": 0.26, "worst_total": 14.0})
LIMIT_25, DOWNLOAD_25 = 600, 180          # the 10 min of the contract; about 3 min to download PyTorch CPU elsewhere

# ---------------------------------------------------------------------------------------------------------------------
PART_B.intro += (" Puis tu le livres : un artefact qui se suffit (B7.17), une API qui le sert (B7.18), dont tu prédis puis "
                 "mesures la latence (B7.19) et que tu accélères sur CPU (B7.20), et un export qui se recharge sans ton "
                 "code (B7.21). Le modèle servi est le même VGG, `MODEL_B` (`SOURCE_B` dans la cellule donnée de B7.17).")

PART_B.exercises += [
    Ex("B7.17", "📦", 2, 30, "L'artefact du modèle : safetensors, configuration, normalisation, classes, version, empreinte des données",
       "livrer un modèle dans un dossier qui suffit à prédire, sans le notebook ni ses variables.",
       "Ex B7.14 · Ex B7.16 · ch. 23 (fiche modèle, `state_dict`) · fiche §207.4 (empreinte), §207.7 (« L'artefact du "
       "modèle »)", tracks="R, C",
       body=package_howto() + r"""

**Le contrat.** Un nouveau fichier, `src/cifarprod/artifact.py`, de trois fonctions (leurs docstrings, dans la cellule à compléter, sont le contrat) :
- `save_artifact(folder, model, *, arch, widths, mean, std, classes, data, metrics=None)` écrit `model.safetensors` (le `state_dict`, avec les métadonnées safetensors `{"package_version": "0.1.0"}`) et `meta.json`, aux clés `package_version`, `created` (la date ISO 8601 en UTC), `arch`, `widths`, `input` (`{"shape": [32, 32, 3], "dtype": "uint8", "channels": "RGB"}`), `mean`, `std`, `classes` (dans l'ordre des sorties), `data` (le dictionnaire reçu, tel quel), `metrics` (`{}` sans métriques), `torch`, `python` et `weights_sha256` (le SHA-256 du fichier de poids, par `file_sha256` de B7.14) ; elle rend le dossier ;
- `load_artifact(folder)` rend `(model, meta)` : le modèle reconstruit par `build_model(meta["arch"], meta["widths"], len(meta["classes"]))`, ses poids chargés, en mode `eval()` ; elle refuse (`ValueError`) un fichier de poids dont le SHA-256 n'est pas celui de `meta.json` ;
- `predict_images(model, meta, images)` rend `(labels, probabilities)` pour des images `uint8` RGB `(N, 32, 32, 3)` ou une seule `(32, 32, 3)` : les noms `meta["classes"][k]` et un tableau `float32` `(N, n_classes)` de probabilités softmax, calculés sans graphe avec `preprocess` de B7.9 et la normalisation de l'artefact.

Puis, dans la cellule de **B7.10** (`CLI_PY`), fais écrire l'artefact à `cifarprod train` : `save_artifact(out, ...)` remplace l'écriture des seuls poids (`arch` et `widths` de la configuration, la normalisation apprise, `CLASSES`, et pour `data` l'entrée `data` du journal de B7.14) ; la liste `artifacts` du journal garde les poids en premier, puis `meta.json`. Réexécute la cellule de vérification de B7.10 (elle écrit aussi `artifact.py`), puis celle-ci.

La cellule donnée charge le modèle servi par la partie B, `MODEL_B` (le VGG du MP5, `INFO_B` : sa configuration, sa normalisation, ses classes, sa val_acc ; `DATA_B` : l'empreinte de ses 45 000 images d'entraînement, calculée une fois par le chapitre), et 200 images de test (`X_B`, `Y_B`). La cellule de vérification emballe `MODEL_B` avec ton `save_artifact`, relit les deux fichiers (tenseurs au bit près, en-tête, champs de `meta.json`), lance un **processus neuf** qui recharge l'artefact avec ton `load_artifact` et prédit les 200 images (mêmes noms et mêmes probabilités que le notebook, à $10^{-5}$ près), essaie une image seule, un modèle aux sorties rangées dans un autre ordre (les noms doivent suivre) et un fichier de poids remplacé (refusé) ; elle lance enfin `cifarprod train --set n_train=200` et lit l'artefact du run : architecture, classes, normalisation des 200 images et empreinte de leurs données. L'artefact du VGG reste dans `ARTIFACT_B`, que servent B7.18 à B7.21.

Dans tes notes : sur les cinq écarts de B7.16, lesquels un tel artefact aurait-il évités, et lesquels restent possibles malgré lui ? Pourquoi garder l'empreinte des données **et** celle des poids, et à quelles questions répondent-elles six mois plus tard ? Que manque-t-il encore pour qu'un autre que toi rejoue l'entraînement (207.4) ? Où placerais-tu la fiche modèle du ch. 23 (ou de 🛠️ B6.28) par rapport à ce dossier ?""",
       given=C2.GIVEN_17, todo=C2.TODO_17, check=C2.CHECK_17, solution=C2.SOLUTION_17 + "\n\n\n" + C2.CHECK_17),

    Ex("B7.18", "📦", 3, 45, "Une API de prédiction avec FastAPI (repli : http.server) : /health, /predict, validation des entrées",
       "servir un artefact par une API HTTP qui valide ses entrées, répond comme l'évaluation hors ligne et s'arrête proprement.",
       "Ex B7.17 · ⚖️ B7.7 (le champ `input`) · fiche §207.7 (« En ligne ou par batches, et une API HTTP »)", tracks="R, C",
       body=package_howto() + r"""

**Le contrat** de `src/cifarprod/serve.py` (sa docstring et celles de chaque objet sont dans la cellule à compléter) :
- `Image` et `Batch`, deux modèles pydantic : `{"pixels": 32 × 32 × 3 entiers de 0 à 255}` (celui de la fiche) et `{"images": une liste de 1 à 64 telles images}` ; tout le reste lève une `ValueError` dans leur validateur ;
- `Service(artifact_dir)` charge l'artefact (`load_artifact`) : `model`, `meta`, `version` (les 12 premiers caractères de `meta["weights_sha256"]`) ; `health()` rend `{"status": "ok", "package_version", "model": {"arch", "widths", "classes", "weights_sha256", "data_sha256", "created"}, "torch"}` ; `predict(images)` rend une liste de `{"label", "probability"}` (la probabilité du label, arrondie à 4 décimales), par `predict_images` ;
- `create_app(artifact_dir)`, l'application FastAPI, son `Service` dans `app.state.service` : `GET /health` ; `POST /predict` (`Image`) → `{"label", "probability", "model_version"}` ; `POST /predict_batch` (`Batch`) → `{"predictions": [...], "model_version"}` ; une requête mal formée → **422** `{"detail": [{"loc", "msg", "type"}, ...]}` **sans** l'entrée envoyée (FastAPI la renvoie par défaut dans le champ `input` : ⚖️ B7.7) ; une exception du modèle → **500** `{"detail": "internal error"}`, et une ligne `log.exception` ;
- `Handler` et `make_server(artifact_dir, host="127.0.0.1", port=0)`, le **repli** sans FastAPI : un `ThreadingHTTPServer` (pas encore démarré), les mêmes routes, la même validation (les mêmes modèles pydantic), les mêmes réponses et codes de statut, 404 pour une autre route ; son `Service` dans `server.service` ;
- `running(artifact_dir, backend="auto")`, un gestionnaire de contexte (`with running(...) as url:`) qui sert l'artefact dans un **thread**, sur un port libre, rend l'URL de base (`"http://127.0.0.1:<port>"`) et **arrête toujours** le serveur à la sortie du bloc, même après une erreur ; `backend` : `"fastapi"` (uvicorn), `"stdlib"` (`make_server`) ou `"auto"` (FastAPI s'il s'importe) ;
- `main(argv)`, pour `python -m cifarprod.serve --artifact DIR --port 8000` (par défaut, `--artifact` vaut la variable d'environnement `MODEL_DIR`, sinon `artifact`) : sert jusqu'à l'arrêt ; code 1 et une ligne ERROR si l'artefact ne se charge pas.

La cellule de vérification sert `ARTIFACT_B` (B7.17) : avec FastAPI par `TestClient` (sans serveur), puis avec ton `make_server` dans un thread qu'elle démarre et **arrête** elle-même, elle envoie les mêmes requêtes aux deux : `/health`, huit images (les noms du notebook : un test de non-régression), huit requêtes mal formées (422, sans l'entrée dans la réponse), un batch de 32, un modèle qui plante (500 propre), une route inconnue. Elle essaie ensuite ton `running` avec les deux backends, une erreur levée dans le bloc (le port doit être libéré), puis ta commande `python -m cifarprod.serve` dans un **autre processus** (arrêté ensuite), et un dossier sans artefact. L'avertissement de Starlette 1.10 sur `httpx2` est tu (`TestClient` marche avec le `httpx` du workbook).

Dans tes notes : pourquoi la validation des entrées appartient-elle à l'API et pas au modèle ? Pourquoi une réponse 500 ne contient-elle ni la trace de l'erreur ni l'entrée ? Que gagne-t-on à ce que FastAPI et `http.server` partagent `Service` et les modèles pydantic, et que perdrait-on à les écrire deux fois ? Qu'arrive-t-il à un notebook dont le serveur n'est pas arrêté, quand on réexécute la cellule ?""",
       todo=C2.TODO_18, check=C2.CHECK_18, solution=C2.SOLUTION_18 + "\n\n\n" + C2.CHECK_18),

    Ex("B7.19", "🔮", 1, 10, "Prédire la latence p95 de l'API, requête par requête puis par batches de 32",
       "prédire ce que coûte le chemin HTTP d'une prédiction, et ce que change un batch, avant de le mesurer.",
       "Ex B7.18 · ✏️ B7.3 · fiche §207.7 (« Latence, percentiles et débit »), §207.8", tracks="C", hypothesis=True,
       body=r"""L'expérience sert le VGG du MP5 par l'API **de référence** du chapitre (FastAPI et uvicorn, dans un autre processus, arrêté à la fin), pour que tes prédictions se comparent à des mesures qui ne dépendent pas de ton code ; ton API est mesurée ensuite, sans vérification. Un client `requests` (une `Session`, qui garde la connexion) envoie 200 requêtes `POST /predict` d'une image, après 20 appels à blanc, puis 40 requêtes `POST /predict_batch` de 32 images ; elle mesure aussi l'appel direct du modèle (`model(x)`, sans HTTP, sous `torch.inference_mode()`). Le modèle (appel direct) et le serveur tournent sur **un** thread, comme un service lancé en plusieurs processus sur une machine à peu de cœurs (207.8 : threads × processus ≈ cœurs). Les durées sont celles que voit le client, en millisecondes. **Avant d'exécuter**, écris tes deux prédictions (une lettre chacune) :
- a) `http_19` : le p95 de `POST /predict` (une image), divisé par le p95 de l'appel direct du modèle sur une image, vaut : **A** moins de 2 (le HTTP ne coûte presque rien) ; **B** de 2 à 20 ; **C** plus de 20 (le modèle ne pèse presque rien dans la requête) ;
- b) `batch_19` : le p95 de `POST /predict_batch` (32 images), divisé par le p95 de `POST /predict` (une image), vaut : **A** moins de 3 (32 images coûtent à peine plus qu'une) ; **B** de 3 à 32 ; **C** plus de 32 (pas mieux que 32 requêtes l'une après l'autre).

Ce sont des **rapports** de durées mesurées sur la même machine : ils dépendent bien moins du processeur que les durées elles-mêmes, que la cellule rapporte aussi à la vitesse de la machine de génération du chapitre. La cellule d'expérience n'affiche rien tant que tes deux prédictions ne sont pas écrites ; elle les vérifie à la fin.""",
       todo=C2.TODO_19,
       solution=f'http_19, batch_19 = "{L19["answers"]["a"]}", "{L19["answers"]["b"]}"',
       record="\n".join([
           letters_record("B7.19a", L19["answers"]["a"],
                          {"A": "relis le tableau : la ligne POST /predict, 1 image, contre l'appel direct",
                           "C": "relis le tableau : la ligne POST /predict, 1 image, contre l'appel direct"},
                          prediction=True),
           letters_record("B7.19b", L19["answers"]["b"],
                          {"A": "relis le tableau : le p95 de /predict_batch contre celui de /predict",
                           "C": "relis le tableau : le p95 de /predict_batch contre celui de /predict"},
                          prediction=True)]),
       after=[("code", guarded(fill(C2.EXPERIMENT_19, {"DIRECT_P50": fr_num(L19["direct_p50_default"], 2),
                                                       "ANSWER_A": L19["answers"]["a"], "ANSWER_B": L19["answers"]["b"]}),
                               ["http_19", "batch_19"],
                               "⏳ Ex B7.19 : écris d'abord tes deux prédictions dans la cellule précédente.")),
              ("code", guarded(C2.EXPERIMENT_19_OWN, ["http_19", "batch_19"],
                               "⏳ Ex B7.19 : la mesure de ton API vient après tes prédictions.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Où passe le temps d'une "
                     "requête d'une image, en dehors du modèle (pense à ce que fait le JSON de 3 072 entiers, à l'aller "
                     "et au retour) ? Pourquoi le débit gagné par le batch est-il bien inférieur à 32 ? Avec un objectif "
                     "« 95 % des requêtes en moins de 100 ms » (un SLO), lequel des deux usages le tiendrait ici, et que "
                     "ferais-tu du regroupement dynamique (*dynamic batching*, 207.7) ? Compare avec les nombres "
                     "synthétiques de ✏️ B7.3.")]),

    Ex("B7.20", "🔬", 2, 30, "Accélérer l'inférence sur CPU : batches, threads, inference_mode, bfloat16, quantification",
       "mesurer, sur la machine qui sert, ce que chaque levier change à la latence et aux décisions du modèle.",
       "Ex B7.19 · ✏️ B7.1 (la part de la couche dense) · ch. 24 (bfloat16) · fiche §207.8 (« Accélérer sur CPU »)",
       tracks="C",
       body=r"""**Le contrat** des quatre fonctions de la cellule suivante (leurs docstrings le donnent) :
- `latency_20(fn, x, n=50, warmup=5)` rend `(p50, p95)` en millisecondes, deux floats Python, de `n` appels `fn(x)` mesurés par `time.perf_counter`, après `warmup` appels à blanc non mesurés ;
- `bfloat16_20(model)` rend une fonction `x -> logits` : une **copie** du modèle convertie en `torch.bfloat16`, appliquée à `x` converti, qui rend des logits `float32` ; le modèle donné ne change pas ;
- `quantized_20(model)` rend une **copie** du modèle dont les `nn.Linear` sont quantifiées en int8 par `torch.ao.quantization.quantize_dynamic` (`dtype=torch.qint8`) ; dans PyTorch 2.11, elle affiche un avertissement de dépréciation (207.8), que la cellule tait ;
- `agreement_20(a, b)` rend `(part, écart)` pour deux tenseurs de logits `(N, C)` : la part des lignes dont les argmax sont les mêmes, et le plus grand écart absolu, deux floats Python.

La cellule de vérification essaie chaque fonction sur un petit cas (une fonction lente pendant ses appels à blanc, puis rapide ; des logits dont trois argmax sur quatre s'accordent ; le type et la structure des modèles rendus). L'expérience mesure ensuite le VGG du MP5 sur une grille : batch de 1 et de 32 ; `inference_mode`, `no_grad` ou l'autograd actif ; le nombre de threads par défaut ou un seul ; float32, bfloat16 ou int8 dynamique. Pour chaque ligne, le p50, le p95 et les images par seconde ; puis, sur 1 000 images de test, l'accuracy de chaque précision et l'accord de ses décisions avec le float32. **Aucune mesure n'est enregistrée** : elles dépendent de la machine et varient d'une exécution à l'autre ; réexécute la cellule deux fois pour voir ce qui tient.

Dans tes notes : quels leviers accélèrent vraiment ici, lesquels ne changent rien, lesquels **ralentissent**, et pourquoi (pense au processeur, à ce qu'il sait calculer en 16 bits, et à la part des paramètres que la quantification dynamique touche, ✏️ B7.1 j) ? Les décisions changent-elles, et sur combien d'images sur 1 000 ? Lequel de ces leviers mettrais-tu en production pour un service qui reçoit une image à la fois, et lequel pour une prédiction par batches de nuit ? Que retiens-tu de l'écart entre deux exécutions de la même ligne ?""",
       todo=C2.TODO_20, check=C2.CHECK_20, solution=C2.SOLUTION_20 + "\n\n\n" + C2.CHECK_20,
       after=[("code", C2.EXPERIMENT_20)]),

    Ex("B7.21", "📦", 2, 25, "Exporter avec torch.export et vérifier l'équivalence numérique (ONNX en option)",
       "exporter le modèle servi en un graphe qui se recharge sans ton code, et vérifier qu'il calcule la même chose.",
       "Ex B7.17 · 🧠 Q10 · fiche §207.8 (« torch.export et ONNX »)", tracks="C",
       body=package_howto() + r"""

**Le contrat** de `src/cifarprod/export.py` (les docstrings dans la cellule à compléter) :
- `export_model(model, path, max_batch=1024)` met le modèle en mode `eval()`, l'exporte par `torch.export.export` avec une taille de batch **libre** de 1 à `max_batch` (à partir d'un batch d'exemple de 2 images `(2, 3, 32, 32)`), enregistre l'`ExportedProgram` dans `path` (un `.pt2`, `torch.export.save`) et rend ce chemin ;
- `export_artifact(folder, max_batch=1024)` exporte le modèle de l'artefact `folder` (`load_artifact`) dans `folder / "model.pt2"` et rend ce chemin ;
- `max_gap(model, exported, x)` rend le plus grand écart absolu entre les logits du modèle et ceux d'`exported` sur le batch `x`, calculés sans graphe, un float Python.

La cellule de vérification exporte `ARTIFACT_B` avec ton `export_artifact`, puis lance un **processus sans ton paquet** (`cifarprod` n'y est pas importable) qui recharge `model.pt2` par `torch.export.load` et calcule des batches de 1, 7 et 64 images de test : l'écart maximal avec `MODEL_B` doit rester **au plus** $10^{-6}$ (un verdict, puis un `wb.check` de ce verdict : rien à taper). Ce processus essaie aussi un batch de 1 025 images et une image de 28 × 28, que les gardes doivent refuser. Puis elle exporte avec ton `export_model` un modèle **laissé en mode entraînement** par son appelant (l'export doit être celui du mode `eval()`), et essaie `max_gap` sur un modèle dont on a décalé les biais de sortie de 0,01. L'export ONNX n'est essayé que si `onnxscript` et `onnxruntime` sont installés (extra `prod`, ni dans le workbook ni sur Colab).

Dans tes notes : que vaut l'écart, et pourquoi n'est-ce pas surprenant (que fait l'export des opérations du modèle) ? Que donnerait un export fait en mode entraînement sur ce VGG, et pourquoi la batchnorm en est-elle la cause ? Pourquoi le batch d'exemple a-t-il 2 images et pas 1 ? Combien pèse `model.pt2` à côté de `model.safetensors`, et que contient-il de plus ? Quand choisirais-tu ONNX plutôt que `torch.export` ?""",
       todo=C2.TODO_21, check=C2.CHECK_21, solution=C2.SOLUTION_21 + "\n\n\n" + C2.CHECK_21, record=C2.RECORD_21,
       after=[("code", C2.VERIFY_21)]),
]

# ---------------------------------------------------------------------------------------------------------------------
PART_C = Part("C", "L'exploitation : la dérive, l'intégration continue, le Dockerfile et le défi",
              "Le service tourne ; il faut maintenant le **surveiller** (B7.22 : des images plus sombres ou bruitées "
              "arrivent, lesquelles faut-il signaler ?), **protéger** chaque modification par une intégration continue "
              "(B7.23), **emballer** le service dans une image Docker (B7.24), et rendre le tout reproductible par "
              "n'importe qui, de `git clone` à une prédiction servie (🏆 B7.25). Rien n'est envoyé sur GitHub ni "
              "construit par Docker : le workflow et le Dockerfile sont vérifiés par lecture, et le défi se joue sur "
              "un clone local de ton dépôt.",
              exercises=[
    Ex("B7.22", "📈", 2, 25, "Surveiller la dérive : PSI et test KS sur des images assombries ou bruitées, quand alerter ?",
       "mesurer une dérive sur les entrées et sur les sorties, et décider de ce qui mérite une alerte.",
       "✏️ B7.4 · Ex B7.17 · ch. 8 (p-valeur) · fiche §207.10 (PSI, test KS, « Agir »)", tracks="C",
       body=r"""Une **référence** (500 images de test, et les prédictions du VGG du MP5 sur elles) et quatre **fenêtres** de production de 500 **autres** images de test, comme un service pourrait les recevoir : **A** telles quelles ; **B** plus sombres (pixels × 0,6 : une caméra au crépuscule) ; **C** bruitées (un bruit gaussien d'écart-type 25 sur l'échelle 0-255 : un capteur moins cher) ; **D** un peu plus sombres (pixels × 0,9). On surveille une variable des **entrées**, la luminosité moyenne de chaque image, et une des **sorties**, la classe prédite.

**Le contrat** des trois fonctions de la cellule suivante (leurs docstrings le donnent) :
- `brightness_22(images)` : la luminosité moyenne de chaque image `uint8` `(N, 32, 32, 3)`, la moyenne de ses 3 072 valeurs divisée par 255, un tableau `float64` `(N,)` ;
- `psi_22(ref, prod, bins=10, eps=1e-4)` : le PSI de 207.10 sur `bins` intervalles coupés aux quantiles $k/\text{bins}$ de la **référence** (`np.quantile`, $k = 1, \dots, \text{bins} - 1$ : des déciles, et non les quartiles de ✏️ B7.4), le premier et le dernier ouverts ; une valeur égale à une coupure va dans l'intervalle **au-dessus** ; $\varepsilon$ ajouté à **toutes** les parts ; un float Python ;
- `psi_classes_22(ref_labels, prod_labels, n_classes=10, eps=1e-4)` : le même PSI sur les parts des dix classes prédites.

La cellule de vérification compare tes fonctions à un calcul écrit autrement, sur quatre cas (la référence contre elle-même, des valeurs tirées ailleurs, des parts nulles, des valeurs égales aux coupures). Puis le tableau : pour chaque fenêtre, le PSI de la luminosité, le test KS sur la luminosité (`scipy.stats.ks_2samp` : $D$ et la p-valeur), le PSI des classes prédites, la confiance moyenne (la probabilité de la classe prédite) et l'accuracy (avec les vrais labels, connus ici, rarement en production) ; et deux figures, les histogrammes de la luminosité et les parts des classes prédites. **Lis-les**, puis réponds dans la cellule qui suit (des lettres de fenêtres, dans n'importe quel ordre, entre guillemets) :
- a) `inputs_22` : les fenêtres dont le PSI de la luminosité dépasse 0,25 (« changement important ») ;
- b) `ks_22` : les fenêtres où le test KS sur la luminosité rejette à 5 % ($p < 0{,}05$) ;
- c) `outputs_22` : les fenêtres dont le PSI des classes prédites dépasse 0,25 ;
- d) `harm_22` : les fenêtres où l'accuracy perd plus de 10 points par rapport à la référence ;
- e) `psi_d_22` : le PSI de la luminosité de la fenêtre D, à 2 décimales.

Dans tes notes : pour chaque fenêtre, que ferais-tu de l'alerte (rien, surveiller de plus près, vérifier la chaîne d'acquisition, revenir en arrière ou réentraîner, 207.10) ? Quelle fenêtre une surveillance des seules entrées laisserait-elle passer, et pourquoi ? Pourquoi le test KS rejette-t-il D alors que l'accuracy ne bouge pas, et que deviendrait sa p-valeur avec 50 000 images par fenêtre ? Que changerait une référence prise sur les images d'entraînement plutôt que sur le test ?""",
       given=C3.GIVEN_22, todo=C3.TODO_22, check=C3.CHECK_22, solution=C3.SOLUTION_22 + "\n\n\n" + C3.CHECK_22,
       after=[("code", C3.EXPERIMENT_22),
              ("todo", C3.TODO_ANSWERS_22),
              ("check", C3.CHECK_ANSWERS_22),
              ("solution", f'inputs_22, ks_22, outputs_22, harm_22 = "{D22["answers"]["a"]}", "{D22["answers"]["b"]}", '
                           f'"{D22["answers"]["c"]}", "{D22["answers"]["d"]}"\npsi_d_22 = round(TABLE_22["D"]["psi"], 2)'),
              ("record", "\n".join([
                  several_record("B7.22a", D22["answers"]["a"], "ABCD", "la colonne PSI lum. et le seuil 0,25"),
                  several_record("B7.22b", D22["answers"]["b"], "ABCD", "la colonne KS p et le seuil 0,05"),
                  several_record("B7.22c", D22["answers"]["c"], "ABCD", "la colonne PSI classes et le seuil 0,25"),
                  several_record("B7.22d", D22["answers"]["d"], "ABCD", "la colonne accuracy, comparée à celle de la référence"),
                  'wb.record("B7.22e", psi_d_22, decimals=2, mistakes={"relis la question : la fenêtre D, colonne PSI lum.": '
                  '{round(TABLE_22[k]["psi"], 2) for k in "ABC"}, "relis la question : le PSI de la luminosité, pas celui '
                  'des classes": round(TABLE_22["D"]["psi_classes"], 2), "relis la question : le PSI, pas la statistique D '
                  'du test KS": round(TABLE_22["D"]["D"], 2)})']))]),

    Ex("B7.23", "🛠️", 2, 25, "Intégration continue : un workflow GitHub Actions qui installe, teste et lance un entraînement fumée",
       "écrire le workflow qui rejoue l'installation, les tests et un entraînement fumée à chaque push.",
       "Ex B7.15 · fiche §207.6 (« L'intégration continue »), §207.9 (secrets)", tracks="R, C",
       body=r"""**Le contrat** de `.github/workflows/ci.yml` (la fiche en donne le début) : déclenché à chaque `push` et `pull_request` ; un job sur `ubuntu-latest` ; les étapes, dans cet ordre : `actions/checkout@v7` ; `actions/setup-python@v7` avec `python-version: "3.13"` (**entre guillemets**) et `cache: pip` ; l'installation de torch depuis l'**index CPU** de PyTorch, **puis** celle de ton paquet en mode éditable avec son extra `dev` (`pip install -e ".[dev]"`, ou `".[dev,serve]"` pour les tests de l'API) ; `pytest` (les tests `slow` compris) ; un **entraînement fumée** : `scripts/make_synthetic.py` écrit un fichier de données synthétiques (les données du projet ne sont pas dans git), puis `cifarprod train --config configs/smoke.yaml --set data=<ce fichier>`. Aucune étape n'avale son échec (`|| true`, `continue-on-error`), aucun secret n'est écrit en clair (un secret passerait par `${{ secrets.NOM }}`).

La cellule suivante donne `scripts/make_synthetic.py` (256 images d'entraînement et 64 de validation tirées par `synthetic` de B7.9, au schéma du fichier du projet). La cellule de vérification écrit les deux fichiers dans ton projet et relit ton workflow **sans le lancer** (ni GitHub ni Docker ne sont là) : PyYAML d'abord (attention, il lit la clé `on:` comme le booléen `True` : le vérificateur le sait, pas forcément ton prochain script), puis chaque point du contrat, l'ordre des étapes, la syntaxe de chaque bloc `run:` par `bash -n` (qui lit sans exécuter), et la présence dans le projet des fichiers que le workflow cite.

Dans tes notes : que se passe-t-il, sur GitHub, si `python-version: 3.10` est écrit sans guillemets ? Pourquoi installer torch **avant** le paquet ? Pourquoi un entraînement fumée **en plus** des tests, et pourquoi sur des données synthétiques ? Combien de temps prendrait ce workflow (pense au cache de pip et au téléchargement de torch CPU), et que verrait un recruteur sur la page de ton dépôt ?""",
       given=C3.GIVEN_23, todo=C3.TODO_23, check=C3.CHECK_23, solution=C3.SOLUTION_23 + "\n\n\n" + C3.CHECK_23),

    Ex("B7.24", "📦", 2, 20, "Lire et compléter un Dockerfile : image de base, dépendances CPU, utilisateur non root, variables",
       "lire un Dockerfile, en repérer les erreurs, et le compléter selon les bonnes pratiques d'une image de service.",
       "Ex B7.18 · fiche §207.9 (« Image, conteneur et Dockerfile », « Variables d'environnement et secrets »)",
       tracks="C",
       body=r"""Un collègue a commencé le Dockerfile de ton API (`DOCKERFILE`, dans la cellule suivante) : plusieurs lignes sont fausses, d'autres manquent (les `TODO`). **Lis-le, corrige-le et complète-le** pour qu'il construise l'image du service sur CPU. **Le contrat** (le tableau de 207.9) :
- une image de base **slim** de Python 3.13 à une **version précise** (pas un tag qui bouge) ;
- des réglages en `ENV`, **jamais un secret** : `PYTHONUNBUFFERED=1` et `MODEL_DIR`, le dossier de l'artefact dans l'image ;
- les dépendances **avant** le code : `COPY requirements-serve.txt`, `RUN` torch `2.11.0` depuis l'index CPU, puis `pip install -r requirements-serve.txt` ; ensuite seulement `pyproject.toml` et `src/`, le paquet lui-même (`pip install --no-deps .`), et le dossier `artifact/` (`MODEL_DIR` le désigne) ;
- un utilisateur **sans privilèges**, créé (`useradd`) puis choisi (`USER`) avant la commande ;
- `EXPOSE 8000` et une commande en **forme exec** (une liste JSON) qui lance ton API (`python -m cifarprod.serve`, B7.18) sur `0.0.0.0` et le port exposé.

Écris aussi `DOCKERIGNORE` (ce qui ne doit jamais entrer dans l'image : au moins `.git`, `data/`, `runs/`, les caches `__pycache__` et un éventuel `.env`) et `REQUIREMENTS_SERVE`, les dépendances de l'API (`numpy`, `pyyaml`, `safetensors`, `fastapi`, `uvicorn`, `pydantic`), chacune **figée** avec `==` à la version de ton environnement (`importlib.metadata.version(...)`), sans torch. La cellule de vérification écrit les trois fichiers dans ton projet, relit le Dockerfile **sans le construire** (Docker n'est ni sur Colab ni dans le workbook), point par point, et lance `python -m cifarprod.serve --help` pour vérifier que la commande de `CMD` existe dans ton paquet.

Dans tes notes : pour chaque ligne fausse du collègue, que se serait-il passé (au build, au lancement, ou six mois plus tard) ? Pourquoi `COPY . .` avant `pip install` rend-il chaque build lent ? Pourquoi `--host 0.0.0.0` dans un conteneur, et pourquoi pas sur ton PC ? Où mettrais-tu la clé d'API qu'il avait écrite, et que doit-il faire maintenant qu'elle a été écrite dans un fichier ?""",
       todo=C3.TODO_24, check=C3.CHECK_24, solution=C3.SOLUTION_24 + "\n\n\n" + C3.CHECK_24),

    Ex("B7.25", "🏆", 4, 180, "Défi : un dépôt prêt pour la production, de git clone à une prédiction servie en moins de 10 min de CPU",
       "livrer un dépôt que n'importe qui clone, installe, teste et fait prédire en une commande.",
       "Ex B7.9 à B7.24 · fiche §207.1 à §207.9 · le ML Test Score (§207.6)", tracks="C",
       body=r"""**Le défi.** Ton dépôt `cifar-prod`, tel que B7.9 à B7.24 l'ont construit, doit aller **de `git clone` à une prédiction servie en moins de 10 minutes de CPU** sur une machine neuve, et se juger sur une **grille de 10 critères**, notée **par toi**, chaque critère à 0, ½ ou 1 point (la grille détaillée, avec ce qui vaut ½ et 1, est dans `05_solutions.md`) :

| # | Critère | Ce qu'on regarde |
|---|---|---|
| 1 | installation sur un clone neuf | `git clone`, puis `pip install -e ".[dev,serve]"` (après torch CPU), sans autre geste |
| 2 | `make` ou script unique | une commande, `python scripts/demo.py` (ou `make demo`), de l'entraînement à la prédiction servie |
| 3 | tests verts | `pytest -q` passe sur le clone, sans le fichier de données |
| 4 | intégration continue | `.github/workflows/ci.yml` (B7.23) |
| 5 | configuration | `configs/`, validée (B7.11) |
| 6 | artefact versionné | version, empreintes des données et des poids (B7.17) |
| 7 | API qui répond | `/health` et `/predict` (B7.18) |
| 8 | latence mesurée | un p50 et un p95, en ms, dans le README |
| 9 | README | ce que c'est, installer, tester, entraîner, servir, les résultats |
| 10 | fiche modèle | `MODEL_CARD.md` : modèle, usage prévu, données, performances, limites (ch. 23, B6) |

**À écrire** dans la cellule suivante : `README_MD`, `MODEL_CARD_MD` et `DEMO_PY` (`scripts/demo.py`, dont la docstring est le contrat : dans un clone neuf, sans `data/` ni `runs/`, des données synthétiques si le fichier manque, un entraînement fumée par ta ligne de commande, l'artefact, l'API sur un port libre et toujours arrêtée, `/health`, des requêtes `/predict` pour la latence, et pour dernière ligne de stdout un résumé JSON `{"status": 200, "label", "probability", "p50_ms", "p95_ms", "artifact"}`). Puis **commite** ton projet (`git init`, `git add -A`, `git commit`, dans un terminal, dans son dossier) : l'auto-contrôle clone ce que tu as commité, comme le ferait une autre machine.

**L'auto-contrôle** (la cellule de vérification) clone ton dépôt dans un dossier temporaire (ou le copie, s'il n'est pas un dépôt git), l'installe dans un environnement virtuel neuf, y lance `pytest`, puis ta démo, et chronomètre chaque étape. Il ne télécharge rien (il réutilise les bibliothèques du notebook) : il compte ≈ 3 min pour PyTorch CPU sur une machine neuve, et ramène la durée mesurée à la vitesse de la machine de référence du chapitre (`wbp.machine_speed()`, un petit calcul fixe). Il affiche ensuite, pour chaque critère, un indice (✅ ou ❌) tiré de ce qu'il a vu : **une aide pour ta grille, pas une note** ; aucune note n'est calculée, ni avant ni après.

Dans tes notes : ta note sur 10, critère par critère, et ce qui manque pour chaque demi-point perdu. Qu'est-ce que l'auto-contrôle ne peut pas voir (la qualité de ton README, la justesse de ta fiche modèle, ce qui n'est pas commité) ? Si tu publies le dépôt sur GitHub, que montre l'onglet Actions après ton premier push ? Quel test du ML Test Score (207.6) ton dépôt passe-t-il, et lequel ajouterais-tu en premier ?""",
       todo=C3.TODO_25, check=fill(C3.CHECK_25, {"CALIB_SPEED": round(C25["calib_speed"], 4), "DOWNLOAD_S": DOWNLOAD_25,
                                                 "LIMIT_S": LIMIT_25}),
       solution=C3.SOLUTION_25 + "\n\n\n" + C3.COMMIT_25 + "\n\n\n" + fill(C3.CHECK_25, {
           "CALIB_SPEED": round(C25["calib_speed"], 4), "DOWNLOAD_S": DOWNLOAD_25, "LIMIT_S": LIMIT_25})),
])

PARTS.append(PART_C)
