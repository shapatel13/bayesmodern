PYTHON ?= python
DATASET ?= medmcqa
BASELINE ?=
CANDIDATE ?=
PRESET ?= core_diagnostic_lab

.PHONY: install install-dev test lint format api console research-status rollout-dataset list-experiments list-presets rollout-preset compare-experiments warmup-demo

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

list-presets:
	$(PYTHON) -m eval.experiment_cli list-presets

rollout-preset:
	$(PYTHON) -m eval.experiment_cli run-preset-rollout $(PRESET)

compare-experiments:
	$(PYTHON) -m eval.experiment_cli compare-experiments $(BASELINE) $(CANDIDATE)

warmup-demo:
	powershell -ExecutionPolicy Bypass -File scripts\warmup_priori_x.ps1
