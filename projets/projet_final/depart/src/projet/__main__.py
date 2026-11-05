"""``python -m projet ...``: the same command line as ``projet ...`` (cli.py)."""

import sys

from .cli import main

sys.exit(main())
