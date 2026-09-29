#!/usr/bin/env python
"""Check that your environment is ready for the workbook.

    python 00_setup/check_env.py          # from the repository root
    python 00_setup/check_env.py --quick  # skip the slower checks (torch computation, pytest)

Works with a bare Python: missing packages are reported, not fatal.
Exit code 0 = ready (warnings allowed), 1 = something important is missing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# import name -> distribution name in requirements.txt
PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "scipy": "scipy",
    "sklearn": "scikit-learn", "torch": "torch", "torchvision": "torchvision",
    "xgboost": "xgboost", "lightgbm": "lightgbm", "umap": "umap-learn",
    "gymnasium": "gymnasium", "transformers": "transformers", "datasets": "datasets",
    "shap": "shap", "pytest": "pytest", "nbformat": "nbformat", "nbclient": "nbclient",
}
CRITICAL = {"numpy", "pandas", "matplotlib", "sklearn", "pytest"}
NEEDED_FROM = {  # when a package becomes necessary
    "torch": "ch. 20", "torchvision": "ch. 20", "xgboost": "ch. 14", "lightgbm": "ch. 14",
    "umap": "ch. 12", "gymnasium": "ch. 26", "transformers": "bonus B2-B4",
    "datasets": "bonus B2-B4", "shap": "bonus B6", "nbformat": "outils", "nbclient": "outils",
    "scipy": "ch. 2",
}
DATA_FILES = [
    "data/penguins.csv", "data/penguins_raw.csv", "data/california_housing.csv",
    "data/mnist.npz", "data/sunspots_monthly.csv",
    "data/text/holmes_adventures_pg1661.txt", "data/text/verne_tour_du_monde_pg800.txt",
]

results = {"ok": 0, "warn": 0, "fail": 0}


def report(status: str, message: str) -> None:
    icon = {"ok": "✅", "warn": "⚠️ ", "fail": "❌"}[status]
    results[status] += 1
    print(f"  {icon} {message}")


def pinned_versions() -> dict[str, str]:
    pins = {}
    req = ROOT / "requirements.txt"
    if req.exists():
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                name, version = line.split("==", 1)
                pins[name.strip().lower()] = version.strip()
    return pins


def check_python() -> None:
    print("\n1. Python")
    version = platform.python_version()
    if sys.version_info >= (3, 12):
        report("ok", f"Python {version} ({sys.executable})")
    elif sys.version_info >= (3, 11):
        report("warn", f"Python {version} : ça fonctionne, mais le workbook vise Python 3.13 (comme Colab)")
    else:
        report("fail", f"Python {version} : trop ancien, installe Python 3.13 (voir 00_setup/INSTALL_LOCAL.md)")
    if sys.prefix == sys.base_prefix and "COLAB_RELEASE_TAG" not in os.environ:
        report("warn", "pas d'environnement virtuel actif : recommandé (voir INSTALL_LOCAL.md, étape 4)")


def check_packages() -> dict:
    print("\n2. Bibliothèques")
    pins = pinned_versions()
    found = {}
    for module, dist in PACKAGES.items():
        try:
            mod = importlib.import_module(module)
        except Exception:
            status = "fail" if module in CRITICAL else "warn"
            when = f" (nécessaire à partir du {NEEDED_FROM[module]})" if module in NEEDED_FROM else ""
            report(status, f"{dist} manquant{when}")
            continue
        version = getattr(mod, "__version__", "?")
        found[module] = version
        expected = pins.get(dist)
        if expected and version.split("+")[0].split(".")[:2] != expected.split(".")[:2]:
            report("warn", f"{dist} {version} (le workbook est testé avec {expected})")
        else:
            report("ok", f"{dist} {version}")
    return found


def check_torch(found: dict, quick: bool) -> None:
    print("\n3. PyTorch et device")
    if "torch" not in found:
        report("warn", "PyTorch absent : inutile avant le ch. 20 (INSTALL_LOCAL.md, étape 5)")
        return
    import torch

    if torch.cuda.is_available():
        report("ok", f"GPU disponible : {torch.cuda.get_device_name(0)}")
    else:
        report("ok", "pas de GPU : device = cpu (normal sur ton ordinateur ; GPU gratuit sur Colab)")
    if not quick:
        x = torch.randn(256, 256)
        y = (x @ x.T).sum().item()
        report("ok" if y == y else "fail", "calcul matriciel PyTorch")


def check_repo() -> None:
    print("\n4. Dépôt du workbook")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import wb

        report("ok", f"package wb {wb.__version__} importable")
    except Exception as exc:
        report("fail", f"import wb impossible : {exc}")
        return
    for rel in DATA_FILES:
        path = ROOT / rel
        if path.exists():
            report("ok", f"{rel} ({path.stat().st_size / 1e6:.2f} Mo)")
        else:
            report("fail", f"{rel} manquant : fais `git pull`")
    try:
        answers = json.loads((ROOT / "src" / "wb" / "answers.json").read_text(encoding="utf-8"))
        report("ok", f"answers.json lisible ({len(answers.get('answers', {}))} réponses)")
    except Exception as exc:
        report("fail", f"answers.json illisible : {exc}")
    learner = ROOT / "mon_travail" / "mylearn" / "__init__.py"
    if learner.exists():
        report("ok", "ta librairie mon_travail/mylearn existe")
    else:
        report("warn", "mon_travail/mylearn pas encore créé : python tools/start_chapter.py --init")
    downloads = ROOT / "data" / "downloads"
    try:
        downloads.mkdir(parents=True, exist_ok=True)
        probe = downloads / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        report("ok", "écriture possible dans data/downloads (cache des téléchargements)")
    except OSError as exc:
        report("fail", f"impossible d'écrire dans data/downloads : {exc}")


def check_tests(quick: bool) -> None:
    print("\n5. Tests")
    if quick:
        report("warn", "tests ignorés (--quick)")
        return
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_example_mylearn.py", "--impl=ref", "-q",
         "-p", "no:cacheprovider", "--color=no"],
        cwd=ROOT, capture_output=True, text=True,
    )
    last = (proc.stdout.strip().splitlines() or ["(pas de sortie)"])[-1]
    report("ok" if proc.returncode == 0 else "fail", f"pytest --impl=ref : {last}")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # emoji on Windows consoles and pipes
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the slower checks")
    args = parser.parse_args(argv)
    print(f"🔎 Vérification de l'environnement ({platform.system()} {platform.machine()})")
    check_python()
    found = check_packages()
    check_torch(found, args.quick)
    check_repo()
    check_tests(args.quick)
    print(f"\nBilan : {results['ok']} OK, {results['warn']} avertissement(s), {results['fail']} problème(s).")
    if results["fail"]:
        print("❌ Corrige les problèmes ci-dessus (voir 00_setup/INSTALL_LOCAL.md).")
        return 1
    print("✅ Environnement prêt. Ouvre 00_setup/demo.ipynb pour la démonstration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
