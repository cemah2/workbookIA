"""Versions the workbook is built and tested with.

Kept in sync with ``requirements.txt`` (a test checks it) and with
``docs/BIBLE.md`` §21. Values = versions pre-installed on Google Colab
(googlecolab/backend-info, runtime of 2026-09-28), verified on 2026-09-29;
checked again on 2026-10-01 (checkpoint I, runtime of 2026-09-29: same versions) and on
2026-10-03 (checkpoint II, runtime of 2026-10-02: same versions for the packages above;
transformers, huggingface_hub and peft moved, see requirements.txt).
"""

__version__ = "0.1.0"

PYTHON_EXPECTED = "3.13"

# import name -> expected version (the distribution name may differ)
EXPECTED_VERSIONS = {
    "numpy": "2.1.3",
    "pandas": "2.2.3",
    "matplotlib": "3.10.0",
    "sklearn": "1.6.1",
    "scipy": "1.16.3",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
}

# import name -> distribution name used in requirements.txt
DIST_NAMES = {
    "sklearn": "scikit-learn",
}

VERIFIED_ON = "2026-10-03"
