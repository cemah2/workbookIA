"""Code cells of the chapter B7 notebook, second half, part C (the operations: B7.22 to B7.25), shared by build_chB7.py
(through chB7_parts2.py) and history_chB7.py (used by Claude). Same conventions as chB7_code.py and chB7_code2.py.
"""

from __future__ import annotations

from chB7_code import file_cell, reference_file

# =====================================================================================================================
# 📈 B7.22 · drift
# =====================================================================================================================
GIVEN_22 = r'''# A reference (500 test images, the model's predictions on them) and four production windows of 500 OTHER test images,
# as a service could receive them: A as they are; B darker (pixels x 0.6, a camera at dusk); C noisy (Gaussian noise
# of standard deviation 25 on the 0-255 scale, a cheaper sensor); D a little darker (pixels x 0.9)
import matplotlib.pyplot as plt
from scipy import stats

X_22, Y_22 = wbp.test_images(100)                        # 1 000 test images, 100 per class
REF_22, REF_Y_22 = X_22[::2], Y_22[::2]
PROD_22, PROD_Y_22 = X_22[1::2], Y_22[1::2]
WINDOWS_22 = {"A": PROD_22,
              "B": np.round(PROD_22 * 0.6).astype(np.uint8),
              "C": np.clip(PROD_22 + np.random.default_rng(22).normal(0, 25, PROD_22.shape), 0, 255).round().astype(np.uint8),
              "D": np.round(PROD_22 * 0.9).astype(np.uint8)}
MODEL_22 = wbp.reference_model()[0]                      # the served model (the reference VGG of the MP5)
PROBA_22 = {name: offline_b(MODEL_22, images, REF_INFO["mean"], REF_INFO["std"])[0]
            for name, images in {"ref": REF_22, **WINDOWS_22}.items()}
print(f"ℹ️ référence : {len(REF_22)} images ; fenêtres A à D : {len(PROD_22)} images chacune (d'autres images du test).")'''

TODO_22 = r'''def brightness_22(images):
    """The mean brightness of each uint8 image (N, 32, 32, 3): the mean of its 3 072 values divided by 255, a float64
    NumPy array (N,)."""
    raise NotImplementedError("brightness_22")


def psi_22(ref, prod, bins=10, eps=1e-4):
    """The PSI of the values `prod` against the reference `ref` (1-D arrays): `bins` intervals cut at the quantiles
    k / bins of ref (np.quantile, k = 1 .. bins - 1), the first and the last open; a value equal to a cut goes into
    the interval above it; the shares q_k (ref) and p_k (prod), eps added to every share; the sum of
    (p_k - q_k) ln(p_k / q_k), a Python float."""
    raise NotImplementedError("psi_22")


def psi_classes_22(ref_labels, prod_labels, n_classes=10, eps=1e-4):
    """The PSI of the shares of the classes 0 .. n_classes - 1 of prod_labels against those of ref_labels (integer
    arrays), eps added to every share, a Python float."""
    raise NotImplementedError("psi_classes_22")'''

SOLUTION_22 = r'''def brightness_22(images):
    """The mean brightness of each uint8 image (N, 32, 32, 3): the mean of its 3 072 values divided by 255, a float64
    NumPy array (N,)."""
    images = np.asarray(images)
    return images.reshape(len(images), -1).mean(axis=1, dtype=np.float64) / 255


def psi_22(ref, prod, bins=10, eps=1e-4):
    """The PSI of the values `prod` against the reference `ref` (1-D arrays): `bins` intervals cut at the quantiles
    k / bins of ref (np.quantile, k = 1 .. bins - 1), the first and the last open; a value equal to a cut goes into
    the interval above it; the shares q_k (ref) and p_k (prod), eps added to every share; the sum of
    (p_k - q_k) ln(p_k / q_k), a Python float."""
    cuts = np.quantile(ref, np.arange(1, bins) / bins)
    q = np.bincount(np.searchsorted(cuts, ref, side="right"), minlength=bins) / len(ref) + eps
    p = np.bincount(np.searchsorted(cuts, prod, side="right"), minlength=bins) / len(prod) + eps
    return float(np.sum((p - q) * np.log(p / q)))


def psi_classes_22(ref_labels, prod_labels, n_classes=10, eps=1e-4):
    """The PSI of the shares of the classes 0 .. n_classes - 1 of prod_labels against those of ref_labels (integer
    arrays), eps added to every share, a Python float."""
    q = np.bincount(ref_labels, minlength=n_classes) / len(ref_labels) + eps
    p = np.bincount(prod_labels, minlength=n_classes) / len(prod_labels) + eps
    return float(np.sum((p - q) * np.log(p / q)))'''

CHECK_22 = r'''tools_ok_22 = False


def oracle_psi_22(ref, prod, bins=10, eps=1e-4):
    """The PSI written another way (np.digitize, right=False: a value equal to a cut goes above it)."""
    cuts = np.quantile(np.asarray(ref, float), [k / bins for k in range(1, bins)])
    share = lambda v: np.array([(np.digitize(v, cuts) == k).mean() for k in range(bins)]) + eps
    q, p = share(np.asarray(ref, float)), share(np.asarray(prod, float))
    return float(((p - q) * np.log(p / q)).sum())


with checked("B7.22"):
    light_22 = brightness_22(REF_22[:5])
    want_light_22 = REF_22[:5].astype(np.float64).mean(axis=(1, 2, 3)) / 255
    ok_b_22 = isinstance(light_22, np.ndarray) and light_22.shape == (5,) and np.allclose(light_22, want_light_22, rtol=0, atol=1e-12)
    verdict("B7.22", ok_b_22, f"brightness_22 : la luminosité moyenne de chaque image ({fr(float(want_light_22[0]), 4)} pour "
            "la première).", f"brightness_22 de 5 images : attendu un tableau (5,) de moyennes dans [0, 1], reçu "
            f"{np.asarray(light_22)!r:.120}.")
    cases_22 = {"la référence contre elle-même": (np.linspace(0, 1, 101), np.linspace(0, 1, 101)),
                "des valeurs tirées ailleurs": (np.random.default_rng(0).normal(0, 1, 400), np.random.default_rng(1).normal(0.5, 1, 300)),
                "tout dans un intervalle (des parts nulles)": (np.random.default_rng(2).uniform(0, 1, 200), np.full(50, 0.99)),
                "des valeurs égales aux coupures": (np.array([0, 0, 0, 1, 2, 2, 4, 4, 4, 5, 5, 6, 7, 7], float),
                                                    np.array([5, 5, 4, 4, 7, 2, 6, 5, 0, 3, 6, 4], float))}
    ok_p_22 = True
    for label_22, (r_22, p_22) in cases_22.items():
        with np.errstate(divide="ignore", invalid="ignore"):         # a log(0) of your code: the value says it
            got_22, want_22 = psi_22(r_22, p_22), oracle_psi_22(r_22, p_22)
        good_22 = type(got_22) is float and abs(got_22 - want_22) <= 1e-9 * max(1, abs(want_22))
        ok_p_22 &= good_22
        verdict("B7.22", good_22, f"psi_22, {label_22} : {fr(want_22, 4)}.",
                f"psi_22, {label_22} : attendu le float {want_22:.6f}, reçu {got_22!r} (coupures aux déciles de la "
                "référence, une valeur égale à une coupure va dans l'intervalle au-dessus, eps ajouté à chaque part).")
    with np.errstate(divide="ignore", invalid="ignore"):
        got_c_22 = psi_classes_22(np.arange(100) % 10, np.r_[np.zeros(40, int), np.arange(60) % 6])
    q_c_22 = np.full(10, 0.1) + 1e-4
    p_c_22 = np.bincount(np.r_[np.zeros(40, int), np.arange(60) % 6], minlength=10) / 100 + 1e-4
    want_c_22 = float(((p_c_22 - q_c_22) * np.log(p_c_22 / q_c_22)).sum())
    ok_c_22 = type(got_c_22) is float and abs(got_c_22 - want_c_22) < 1e-9
    verdict("B7.22", ok_c_22, f"psi_classes_22 : {fr(want_c_22, 4)} quand quatre classes disparaissent.",
            f"psi_classes_22 : attendu {want_c_22:.6f} (dix classes, quatre absentes de la production, eps ajouté à "
            f"chaque part), reçu {got_c_22!r}.")
    tools_ok_22 = bool(ok_b_22 and ok_p_22 and ok_c_22)'''

EXPERIMENT_22 = r'''if not tools_ok_22:
    print("⏳ Ex B7.22 : le tableau attend tes trois fonctions justes (cellule précédente).")
else:
    LIGHT_22 = {name: brightness_22(images) for name, images in {"ref": REF_22, **WINDOWS_22}.items()}
    PRED_22 = {name: proba.argmax(axis=1) for name, proba in PROBA_22.items()}
    acc_ref_22 = float((PRED_22["ref"] == REF_Y_22).mean())
    print(f"{'fenêtre':8s} {'PSI lum.':>9s} {'KS D':>6s} {'KS p':>9s} {'PSI classes':>12s} {'confiance':>10s} "
          f"{'accuracy':>9s}")
    print(f"{'réf.':8s} {'':>9s} {'':>6s} {'':>9s} {'':>12s} {PROBA_22['ref'].max(axis=1).mean():10.3f} {acc_ref_22:9.3f}")
    TABLE_22 = {}
    for name_22 in WINDOWS_22:
        ks_22 = stats.ks_2samp(LIGHT_22["ref"], LIGHT_22[name_22])
        TABLE_22[name_22] = {"psi": psi_22(LIGHT_22["ref"], LIGHT_22[name_22]), "D": float(ks_22.statistic),
                             "p": float(ks_22.pvalue), "psi_classes": psi_classes_22(PRED_22["ref"], PRED_22[name_22]),
                             "confidence": float(PROBA_22[name_22].max(axis=1).mean()),
                             "accuracy": float((PRED_22[name_22] == PROD_Y_22).mean())}
        row_22 = TABLE_22[name_22]
        print(f"{name_22:8s} {row_22['psi']:9.3f} {row_22['D']:6.3f} {row_22['p']:9.2g} {row_22['psi_classes']:12.3f} "
              f"{row_22['confidence']:10.3f} {row_22['accuracy']:9.3f}")
    print("PSI lum. : la luminosité moyenne de chaque image, 10 intervalles aux déciles de la référence ; KS : le test de "
          "Kolmogorov-Smirnov sur cette luminosité ; PSI classes : la part de chaque classe prédite ; confiance : la "
          "probabilité moyenne de la classe prédite ; accuracy : avec les vrais labels, connus ici, rarement en production.")
    fig_22, axes_22 = plt.subplots(1, 2, figsize=(12, 3.8))
    edges_22 = np.linspace(0, 1, 41)
    axes_22[0].hist(LIGHT_22["ref"], bins=edges_22, color="0.6", label="référence")
    for name_22, color_22 in zip(WINDOWS_22, ("tab:blue", "tab:orange", "tab:green", "tab:red")):
        axes_22[0].hist(LIGHT_22[name_22], bins=edges_22, histtype="step", lw=1.8, color=color_22, label=f"fenêtre {name_22}")
    axes_22[0].set(xlabel="luminosité moyenne d'une image", ylabel="images", title="Les entrées")
    axes_22[0].legend(fontsize=8)
    width_22 = 0.16
    for j_22, (name_22, color_22) in enumerate(zip(["ref", *WINDOWS_22], ("0.6", "tab:blue", "tab:orange", "tab:green", "tab:red"))):
        shares_22 = np.bincount(PRED_22[name_22], minlength=10) / len(PRED_22[name_22])
        axes_22[1].bar(np.arange(10) + (j_22 - 2) * width_22, shares_22, width_22, color=color_22,
                       label="référence" if name_22 == "ref" else f"fenêtre {name_22}")
    axes_22[1].set_xticks(np.arange(10), [FRENCH_16[c] if "FRENCH_16" in globals() else c for c in CLASSES], rotation=45,
                          fontsize=8)
    axes_22[1].set(ylabel="part des images", title="Les sorties : classes prédites")
    axes_22[1].legend(fontsize=8)
    fig_22.tight_layout()
    plt.show()'''

TODO_ANSWERS_22 = r'''# Ex B7.22: read the table and the figure above
inputs_22 = ...    # a) the windows whose brightness PSI is above 0.25 (« changement important »): the letters, e.g. "AB"
ks_22 = ...        # b) the windows where the KS test on the brightness rejects at 5 % (p < 0.05): the letters
outputs_22 = ...   # c) the windows whose PSI of the predicted classes is above 0.25: the letters
harm_22 = ...      # d) the windows where the accuracy loses more than 10 points against the reference: the letters
psi_d_22 = ...     # e) the brightness PSI of window D, 2 decimals'''

CHECK_ANSWERS_22 = r'''wb.check("B7.22a", inputs_22)
wb.check("B7.22b", ks_22)
wb.check("B7.22c", outputs_22)
wb.check("B7.22d", harm_22)
wb.check("B7.22e", psi_d_22)'''

# =====================================================================================================================
# 🛠️ B7.23 · continuous integration
# =====================================================================================================================
GIVEN_23 = file_cell("MAKE_SYNTHETIC_PY", reference_file("scripts/make_synthetic.py"),
                     "scripts/make_synthetic.py (given): the data file of the smoke training in the continuous "
                     "integration, which has no CIFAR-10")

TODO_23 = r'''# Ex B7.23: .github/workflows/ci.yml (start from the beginning given by the fiche, 207.6, and complete it)
CI_YML = ...'''

SOLUTION_23 = file_cell("CI_YML", reference_file(".github/workflows/ci.yml"), "Ex B7.23: .github/workflows/ci.yml")

CHECK_23 = r'''ok_23 = {}
with checked("B7.23"):
    wbp.write_files(PROJECT, {"scripts/make_synthetic.py": MAKE_SYNTHETIC_PY, ".github/workflows/ci.yml": CI_YML})
    if CI_YML is ...:
        print("⏳ Ex B7.23 : écris d'abord CI_YML.")
        raise StopCheck
    for item_23 in wbp.review_workflow(CI_YML, PROJECT):
        ok_23[item_23.key] = item_23.ok
        verdict("B7.23", item_23.ok, item_23.success, item_23.failure)
        if item_23.key == "yaml" and not item_23.ok:
            raise StopCheck
if ok_23 and all(ok_23.values()):
    print("✅ Ex B7.23 : ton workflow installe, teste et entraîne à chaque push (vérifié par lecture : rien n'est lancé ici). "
          "Sur GitHub, l'onglet Actions montre son premier run après ton prochain push.")'''

# =====================================================================================================================
# 📦 B7.24 · the Dockerfile
# =====================================================================================================================
DOCKERFILE_START = '''# The prediction API of cifarprod, on CPU. Build: docker build -t cifar-api .  Run: docker run -p 8000:8000 cifar-api
# (an artifact folder, written by cifarprod train --out artifact, must be next to this file)
# TODO 1: the base image: Python 3.13, slim, at a precise version
FROM python:latest

# TODO 2: the settings (never a secret): unbuffered Python logs, and MODEL_DIR, the folder of the model in the image
ENV HF_TOKEN=hf_xxxxxxxxxxxxxxxx

WORKDIR /app

# TODO 3: the dependencies FIRST (a layer kept in the cache while only the code changes): copy requirements-serve.txt,
# install torch 2.11.0 from the CPU index of PyTorch, then the requirements
COPY . .
RUN pip install torch -r requirements-serve.txt

# TODO 4: the code (pyproject.toml, src/), the package itself (pip install --no-deps .), then the artifact

# TODO 5: a user without privileges, who runs the server

EXPOSE 8000
# TODO 6: the command, in exec form, listening on every interface (0.0.0.0) and on the port of EXPOSE
CMD python -m cifarprod.serve
'''

TODO_24 = "\n\n".join([
    "# Ex B7.24: read this Dockerfile and complete it (every TODO); it is checked by reading, never built",
    file_cell("DOCKERFILE", DOCKERFILE_START, "Dockerfile"),
    "DOCKERIGNORE = ...         # .dockerignore: what never goes into the image\n"
    "REQUIREMENTS_SERVE = ...   # requirements-serve.txt: the dependencies of the API, pinned (torch is installed apart)",
])

SOLUTION_24 = "\n\n".join([
    "# Ex B7.24: the Dockerfile of the reference package, its .dockerignore and its requirements",
    file_cell("DOCKERFILE", reference_file("Dockerfile"), "Dockerfile"),
    file_cell("DOCKERIGNORE", reference_file(".dockerignore"), ".dockerignore"),
    file_cell("REQUIREMENTS_SERVE", reference_file("requirements-serve.txt"), "requirements-serve.txt"),
])

CHECK_24 = r'''ok_24 = {}
with checked("B7.24"):
    if DOCKERIGNORE is ... or REQUIREMENTS_SERVE is ...:
        print("⏳ Ex B7.24 : écris aussi DOCKERIGNORE et REQUIREMENTS_SERVE.")
        raise StopCheck
    wbp.write_files(PROJECT, {"Dockerfile": DOCKERFILE, ".dockerignore": DOCKERIGNORE,
                              "requirements-serve.txt": REQUIREMENTS_SERVE})
    for item_24 in wbp.review_dockerfile(DOCKERFILE, DOCKERIGNORE, REQUIREMENTS_SERVE, PROJECT):
        ok_24[item_24.key] = item_24.ok
        verdict("B7.24", item_24.ok, item_24.success, item_24.failure)
if ok_24 and all(ok_24.values()):
    print("✅ Ex B7.24 : ton Dockerfile suit les bonnes pratiques (vérifié par lecture : Docker n'est pas lancé). Pour le "
          "construire sur ta machine : cifarprod train --out artifact, puis docker build -t cifar-api . et "
          "docker run -p 8000:8000 cifar-api.")'''

# =====================================================================================================================
# 🏆 B7.25 · the challenge
# =====================================================================================================================
SKEL_DEMO = '''"""From a fresh clone to a served prediction, in one command: python scripts/demo.py (no argument).

It must work in a fresh clone (no data/, no runs/: the data are not in git): the data (data/cifar10_2000.npz if it is
there, else a synthetic file), a smoke training with configs/smoke.yaml through the command line, the artifact, the
API on a free port (always stopped at the end), GET /health, then POST /predict with one image, several times for the
latency. The LAST line of stdout is a JSON summary {"status": 200, "label", "probability", "p50_ms", "p95_ms",
"artifact": the folder of the artifact}; the exit code is 0 only if every step worked."""

import sys


def main():
    raise NotImplementedError("scripts/demo.py")


if __name__ == "__main__":
    sys.exit(main())
'''

TODO_25 = "\n\n".join([
    "# Ex B7.25: the last files of your repository (written by the next cell); then commit everything (git init, git add,\n"
    "# git commit: in a terminal, in the folder of the project), so that the self-check clones what you committed",
    "README_MD = ...        # README.md: what it is, how to install, test, train, serve; the results (with the latency)",
    "MODEL_CARD_MD = ...    # MODEL_CARD.md: the model card (ch. 23, B6): model, intended use, data, performance, limits",
    file_cell("DEMO_PY", SKEL_DEMO, "scripts/demo.py"),
])

SOLUTION_25 = "\n\n".join([
    "# Ex B7.25: the README, the model card and the demo of the reference package; then the project is committed with git",
    file_cell("README_MD", reference_file("README.md"), "README.md"),
    file_cell("MODEL_CARD_MD", reference_file("MODEL_CARD.md"), "MODEL_CARD.md"),
    file_cell("DEMO_PY", reference_file("scripts/demo.py"), "scripts/demo.py"),
    file_cell("MAKEFILE", reference_file("Makefile"), "Makefile"),
])

COMMIT_25 = r'''# The solutions notebook commits its temporary project (your copy: do it yourself, in a terminal)
wbp.write_files(PROJECT, {"README.md": README_MD, "MODEL_CARD.md": MODEL_CARD_MD, "scripts/demo.py": DEMO_PY,
                          "Makefile": MAKEFILE})
for args_25 in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.name=workbook", "-c", "user.email=workbook@example.org", "commit", "-q", "-m",
                 "cifar-prod: from the notebook to production"]):
    print(f"$ {' '.join(args_25[:2])} … → code {wbp.run(args_25, PROJECT, timeout=60).returncode}")'''

CHECK_25 = r'''# The self-check of the challenge: your repository cloned afresh, installed, tested, then your demo, timed. No grade is
# computed here: grade your repository yourself with the grid of 05_solutions.md (10 criteria)
FILES_25 = {"README.md": README_MD, "MODEL_CARD.md": MODEL_CARD_MD, "scripts/demo.py": DEMO_PY}
if all(text is ... for text in (README_MD, MODEL_CARD_MD)):
    print("⏳ Ex B7.25 : écris d'abord README_MD, MODEL_CARD_MD et DEMO_PY (et fais B7.9 à B7.24).")
else:
    with checked("B7.25"):
        wbp.write_files(PROJECT, FILES_25)
        steps_25, start_25 = {}, time.perf_counter()
        speed_25 = wbp.machine_speed()
        t_25 = time.perf_counter()
        clone_25, how_25, uncommitted_25 = wbp.clone_project(PROJECT, scratch("clone") / "cifar-prod")
        steps_25["clone"] = (True, time.perf_counter() - t_25, how_25)
        if how_25 != "git clone":
            print("ℹ️ ton projet n'est pas un dépôt git : la cellule le copie (sans data/, runs/ ni caches) au lieu de le "
                  "cloner. Fais git init, git add et git commit dans son dossier pour que le clone ne voie que ce que tu as "
                  "commité, comme une autre machine.")
        elif uncommitted_25:
            print(f"ℹ️ {len(uncommitted_25)} fichier(s) modifié(s) ou nouveau(x) pas encore commité(s), absent(s) du clone : "
                  f"{', '.join(uncommitted_25[:6])}.")
        install_25 = wbp.install_editable(clone_25)
        try:
            steps_25["install"] = (bool(install_25.ok), install_25.seconds, install_25.method)
            tests_25 = demo_25 = None
            if install_25.ok:
                tests_25 = install_25.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--strict-markers",
                                           "-c", str(clone_25 / "pyproject.toml"), "--rootdir", str(clone_25)], clone_25,
                                          timeout=600)
                summary_25 = (tests_25.stdout.strip().splitlines() or ["(rien)"])[-1]
                steps_25["tests"] = (tests_25.ok, tests_25.seconds, summary_25)
                demo_25 = install_25.run(["python", "scripts/demo.py"], clone_25, timeout=600)
                steps_25["demo"] = (demo_25.ok, demo_25.seconds, f"code {demo_25.returncode}")
        finally:
            install_25.cleanup()
        total_25 = time.perf_counter() - t_25
        equivalent_25 = total_25 * «CALIB_SPEED» / speed_25
        for name_25, (good_25, seconds_25, what_25) in steps_25.items():
            print(f"{'✅' if good_25 else '❌'} {name_25:8s} {seconds_25:6.1f} s  ({what_25})")
        verdict("B7.25", equivalent_25 + «DOWNLOAD_S» <= «LIMIT_S» and all(g for g, _, _ in steps_25.values()),
                f"de la copie à la prédiction servie : {fr(total_25, 0)} s ici, soit {fr(equivalent_25, 0)} s sur la "
                f"machine de référence du chapitre (cette machine va {fr(«CALIB_SPEED» / speed_25, 2)} fois sa vitesse) ; "
                f"avec ≈ {«DOWNLOAD_S» // 60} min pour télécharger PyTorch CPU sur une machine neuve, moins de "
                f"{«LIMIT_S» // 60} min.",
                f"de la copie à la prédiction servie : {fr(total_25, 0)} s ici ({fr(equivalent_25, 0)} s sur la machine de "
                f"référence) ; attendu chaque étape réussie, en moins de {«LIMIT_S» // 60} min avec ≈ {«DOWNLOAD_S» // 60} "
                "min de téléchargement de PyTorch CPU.")
        if demo_25 is not None and not demo_25.ok:
            print(f"   la démo : {demo_25.tail(4)[:400]}")
        if tests_25 is not None and not tests_25.ok:
            print("   les tests : " + " ⏎ ".join(l for l in tests_25.stdout.splitlines() if l.startswith(("FAILED", "ERROR")))[:400])
        summary_demo_25 = demo_25.last_json() if demo_25 is not None and demo_25.ok else None
        meta_25 = {}
        if isinstance(summary_demo_25, dict) and summary_demo_25.get("artifact"):
            try:
                meta_25 = json.loads((Path(summary_demo_25["artifact"]) / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta_25 = {}
        readme_25 = (clone_25 / "README.md").read_text(encoding="utf-8") if (clone_25 / "README.md").is_file() else ""
        card_25 = (clone_25 / "MODEL_CARD.md").read_text(encoding="utf-8") if (clone_25 / "MODEL_CARD.md").is_file() else ""
        ci_25 = (clone_25 / ".github/workflows/ci.yml")
        ci_ok_25 = ci_25.is_file() and all(r.ok for r in wbp.review_workflow(ci_25.read_text(encoding="utf-8"), clone_25))
        configs_25 = [p.name for p in (clone_25 / "configs").glob("*.yaml")] if (clone_25 / "configs").is_dir() else []
        evidence_25 = [
            ("installation sur un clone neuf", steps_25.get("install", (False,))[0] and how_25 == "git clone",
             f"{how_25}, puis pip install -e dans un environnement neuf"),
            ("make ou script unique", steps_25.get("demo", (False,))[0], "python scripts/demo.py, code 0"
             + (" ; un Makefile aussi" if (clone_25 / "Makefile").is_file() else "")),
            ("tests verts", steps_25.get("tests", (False,))[0], steps_25.get("tests", (0, 0, "pas lancés"))[2]),
            ("intégration continue", ci_ok_25, ".github/workflows/ci.yml, relu comme en B7.23"),
            ("configuration", {"fast.yaml", "smoke.yaml"} <= set(configs_25), f"configs/ : {', '.join(sorted(configs_25)) or 'vide'}"),
            ("artefact versionné", bool(meta_25.get("package_version") and meta_25.get("weights_sha256")
                                         and re.fullmatch(r"[0-9a-f]{64}", str((meta_25.get("data") or {}).get("sha256")))),
             "meta.json : version du paquet, SHA-256 des poids et empreinte des données"),
            ("API qui répond", isinstance(summary_demo_25, dict) and summary_demo_25.get("status") == 200
             and summary_demo_25.get("label") in CLASSES, f"la démo : {str(summary_demo_25)[:120]}"),
            ("latence mesurée", bool(re.search(r"p95[^\n]{0,80}\d+([.,]\d+)?\s*ms", readme_25)), "le README donne un p95 en ms"),
            ("README", len(readme_25.split()) >= 150 and "pip install" in readme_25 and "demo" in readme_25,
             f"README.md : {len(readme_25.split())} mots, installation et démo citées"),
            ("fiche modèle", len(card_25.split()) >= 150 and all(w in card_25.lower() for w in ("usage", "données", "limite")),
             f"MODEL_CARD.md : {len(card_25.split())} mots, usage prévu, données et limites")]
        print("\nIndices pour ta grille (une aide, pas une note : chaque critère se juge en lisant ton dépôt) :")
        for k_25, (name_25, seen_25, what_25) in enumerate(evidence_25, start=1):
            print(f"  {k_25:2d}. {'✅' if seen_25 else '❌'} {name_25} — {what_25}")
        print(f"(auto-contrôle en {fr(time.perf_counter() - start_25, 0)} s)")'''
