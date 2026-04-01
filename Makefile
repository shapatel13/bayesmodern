PYTHON ?= python
DATASET ?= medmcqa
BASELINE ?=
CANDIDATE ?=

.PHONY: install install-dev test lint format api console research-status rollout-dataset list-experiments compare-experiments

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

research-status:
	$(PYTHON) -m eval.experiment_cli status

rollout-dataset:
	$(PYTHON) -m eval.experiment_cli run-dataset-rollout $(DATASET)

list-experiments:
	$(PYTHON) -m eval.experiment_cli list-experiments

compare-experiments:
	$(PYTHON) -m eval.experiment_cli compare-experiments $(BASELINE) $(CANDIDATE)
