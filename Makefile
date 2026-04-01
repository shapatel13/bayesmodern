PYTHON ?= python

.PHONY: install install-dev test lint format api console

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev,datasets]

test:
	pytest

lint:
	ruff check .
	mypy src

format:
	ruff format .

api:
	uvicorn apps.api.main:app --reload

console:
	streamlit run apps/research_console/app.py

