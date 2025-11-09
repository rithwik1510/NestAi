PYTHON ?= python
UVICORN ?= uvicorn
APP_MODULE ?= pka.app.main:app
VENV_DIR ?= .venv

export PYTHONPATH := $(PWD)

.PHONY: setup dev diagnostics validate eval format lint test run

setup:
	@echo "==> Creating virtual environment and installing dependencies"
	@if [ ! -d "$(VENV_DIR)" ]; then $(PYTHON) -m venv $(VENV_DIR); fi
	@. $(VENV_DIR)/Scripts/activate; pip install --upgrade pip
	@. $(VENV_DIR)/Scripts/activate; pip install -r requirements.txt || true

dev:
	@echo "==> Launching development server"
	@. $(VENV_DIR)/Scripts/activate; $(UVICORN) $(APP_MODULE) --reload

diagnostics:
	@echo "==> Running Ollama diagnostics"
	@. $(VENV_DIR)/Scripts/activate; $(PYTHON) -m pka.app.scripts.ollama_diagnostics

validate:
	@echo "==> Running NestAi validation flow"
	@. $(VENV_DIR)/Scripts/activate; $(PYTHON) -m pka.app.scripts.validate

eval:
	@echo "==> Running golden-set evaluations"
	@. $(VENV_DIR)/Scripts/activate; $(PYTHON) -m pka.app.services.evals.scorer --config pka/app/services/evals/datasets/personal_golden.yaml --report eval_report.md

lint:
	@. $(VENV_DIR)/Scripts/activate; ruff check pka

format:
	@. $(VENV_DIR)/Scripts/activate; ruff format pka

test:
	@. $(VENV_DIR)/Scripts/activate; pytest -q

run:
	@$(PYTHON) run.py
