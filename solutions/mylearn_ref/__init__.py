"""mylearn_ref: reference implementation of the mylearn library.

Same modules and signatures as ``templates/mylearn_stubs``, fully implemented.
Solutions notebooks load it as ``mylearn`` with ``wb.load_mylearn("ref")``
(equivalent to ``import mylearn_ref as mylearn``, but submodule imports such as
``from mylearn.metrics import f1`` also work).

Rule: imports between modules are *relative* (``from .stats import mean``).
"""

__version__ = "0.1.0"
