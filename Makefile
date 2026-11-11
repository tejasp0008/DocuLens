# DocuLens - reproducible build and experiment targets.
#
# Quick start on a clean checkout:
#     make setup
#     make all
#
# Every target runs inside the project virtualenv at $(VENV).

VENV        ?= .venv
PY          ?= python3
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
DOCULENS    := $(VENV)/bin/doculens
CONFIG      ?= configs/default.yaml
JOBS        ?=
JOBS_FLAG   := $(if $(JOBS),--jobs $(JOBS),)
PORT        ?= 8000
CHROME      ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome

.DEFAULT_GOAL := help
.PHONY: help setup data generate degrade run ablate report paper demo test lint format all clean clean-results

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

$(PYTHON):
	$(PY) -m venv $(VENV)

setup: $(PYTHON) ## Create the virtualenv and install pinned dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo
	@echo "Checking for Tesseract OCR ..."
	@command -v tesseract >/dev/null 2>&1 \
		&& tesseract --version | head -1 \
		|| (echo "WARNING: tesseract not found on PATH."; \
		    echo "  macOS:  brew install tesseract"; \
		    echo "  Ubuntu: sudo apt-get install -y tesseract-ocr"; \
		    echo "  Windows: winget install -e --id UB-Mannheim.TesseractOCR")

generate: ## Generate the synthetic clean corpus
	$(DOCULENS) -c $(CONFIG) generate

degrade: ## Apply every degradation at every severity level
	$(DOCULENS) -c $(CONFIG) degrade

data: generate degrade ## Build the full dataset (clean + degraded)

run: ## Evaluate baseline and pipelines on the test split
	$(DOCULENS) -c $(CONFIG) run $(JOBS_FLAG)

ablate: ## Leave-one-stage-out ablation of the headline pipeline
	$(DOCULENS) -c $(CONFIG) ablate $(JOBS_FLAG)

report: ## Write tables, charts and figures from existing results
	$(DOCULENS) -c $(CONFIG) report

paper: ## Build the four-page IEEE write-up at paper/DocuLens_IEEE.pdf
	$(PYTHON) scripts/make_paper_figure.py -c $(CONFIG)
	@test -x "$(CHROME)" || (echo "Chrome not found at $(CHROME)."; \
	    echo "Set CHROME=/path/to/chrome, or open paper/paper.html and print to PDF."; exit 1)
	cd paper && "$(CHROME)" --headless --disable-gpu --no-pdf-header-footer \
	    --print-to-pdf=DocuLens_IEEE.pdf --virtual-time-budget=10000 paper.html
	@echo "wrote paper/DocuLens_IEEE.pdf"

all: ## Full reproduction: data, evaluation, ablation, failures and report
	$(DOCULENS) -c $(CONFIG) all $(JOBS_FLAG)

demo: ## Serve the live demo UI at http://127.0.0.1:$(PORT)
	@$(PYTHON) -c "import flask" 2>/dev/null || $(PIP) install flask
	$(PYTHON) demo/app.py -c $(CONFIG) --port $(PORT)

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Lint with ruff
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Auto-format with ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

clean-results: ## Delete generated results and outputs (keeps the dataset)
	rm -rf results/* outputs/*
	touch results/.gitkeep outputs/.gitkeep

clean: clean-results ## Delete the dataset as well
	rm -rf data/clean data/degraded data/*.csv
	touch data/.gitkeep
