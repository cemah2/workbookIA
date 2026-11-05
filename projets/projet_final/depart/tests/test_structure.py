"""Structure tests: they pass from the first commit, and must keep passing (PF.10).

They check what a reviewer checks first: the package imports, every public function says what it does, the two
configurations load and differ only by their budget, the documents of the project exist, the data and the checkpoints
stay out of git.
"""

import importlib
import inspect
from pathlib import Path

import pytest
import yaml

from conftest import PACKAGES, ROOT

PACKAGE = sorted(PACKAGES)[0]
MODULES = ["config", "data", "preprocess", "models", "train", "evaluate", "predict", "serve", "cli"]
# the settings of the protocol: the same in the fast and the full mode (only the budget may change)
PROTOCOL_KEYS = ["task", "target", "group_column", "time_column", "test_size", "val_size", "metric", "split_file",
                 "seed"]
DOCUMENTS = ["README.md", "docs/cadrage.md", "docs/protocole.md", "docs/rapport.md", "docs/grille.md", "data/card.md", "model_card.md",
             "experiments/journal.md"]


def test_the_package_imports_and_has_a_version():
    package = importlib.import_module(PACKAGE)
    assert isinstance(package.__version__, str) and package.__version__, \
        f"expected {PACKAGE}.__version__ to be a non-empty string, got {getattr(package, '__version__', None)!r}"


@pytest.mark.parametrize("name", MODULES)
def test_every_module_imports(name):
    importlib.import_module(f"{PACKAGE}.{name}")


@pytest.mark.parametrize("name", MODULES)
def test_every_public_function_has_a_docstring(name):
    module = importlib.import_module(f"{PACKAGE}.{name}")
    missing = [n for n, obj in vars(module).items()
               if not n.startswith("_") and (inspect.isfunction(obj) or inspect.isclass(obj))
               and getattr(obj, "__module__", None) == module.__name__ and not inspect.getdoc(obj)]
    assert not missing, f"expected a docstring (the contract) for {', '.join(missing)} in {module.__name__}"


def _config(name):
    return importlib.import_module(f"{PACKAGE}.config").load_config(ROOT / "configs" / name)


def test_the_two_configurations_load():
    fast, full = _config("fast.yaml"), _config("full.yaml")
    assert fast.fast is True and full.fast is False, \
        f"expected fast: true in fast.yaml and fast: false in full.yaml, got {fast.fast} and {full.fast}"


def test_the_two_configurations_share_the_protocol():
    fast, full = _config("fast.yaml").to_dict(), _config("full.yaml").to_dict()
    different = [k for k in PROTOCOL_KEYS if fast[k] != full[k]]
    assert not different, f"expected the same protocol in both modes, different: {different} (only the budget changes)"


def test_a_configuration_refuses_an_unknown_key_and_a_wrong_type():
    config = importlib.import_module(f"{PACKAGE}.config")
    with pytest.raises(ValueError, match="unknown"):
        config.load_config(ROOT / "configs" / "fast.yaml", ["epoch=3"])        # a typo of epochs
    with pytest.raises(ValueError, match="epochs"):
        config.load_config(ROOT / "configs" / "fast.yaml", ["epochs=three"])


def test_every_key_of_the_yaml_files_is_a_setting():
    fields = set(importlib.import_module(f"{PACKAGE}.config").Config().to_dict())
    for name in ("fast.yaml", "full.yaml"):
        keys = set(yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8")))
        assert keys <= fields, f"expected only settings of Config in {name}, unknown: {sorted(keys - fields)}"


def test_the_command_line_lists_its_commands():
    text = importlib.import_module(f"{PACKAGE}.cli").build_parser().format_help()
    missing = [c for c in ("download", "split", "baselines", "train", "predict", "latency", "serve") if c not in text]
    assert not missing, f"expected the commands {missing} in the help of the command line"


@pytest.mark.parametrize("path", DOCUMENTS)
def test_the_documents_of_the_project_exist(path):
    assert (ROOT / path).is_file(), f"expected the file {path} (the kit gives a model of it)"


def test_the_protocol_holds_the_line_of_the_split_digest():
    text = (ROOT / "docs" / "protocole.md").read_text(encoding="utf-8")
    assert "SHA-256 du découpage :" in text, \
        "expected a line « SHA-256 du découpage : `...` » in docs/protocole.md (evaluate.open_sealed_test reads it)"


def test_data_and_checkpoints_stay_out_of_git():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    missing = [p for p in ("data/raw/", "checkpoints/", "artifacts/", "*.pt") if p not in ignored]
    assert not missing, f"expected {missing} in .gitignore (heavy files and data go elsewhere: README)"
