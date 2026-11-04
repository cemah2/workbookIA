# make install · make test · make demo (from a fresh clone to a served prediction)
PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev,serve]"

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) scripts/demo.py

.PHONY: install test demo
