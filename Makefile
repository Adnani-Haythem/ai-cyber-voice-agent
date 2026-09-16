PYTHON ?= .venv/bin/python
APP_MODULE = app.main

.PHONY: run install

run:
	$(PYTHON) -m $(APP_MODULE)

install:
	$(PYTHON) -m pip install -r requirements.txt
