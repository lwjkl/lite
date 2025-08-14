# Makefile

# Variables
PYTHON = uv run python
UV = uv run
APP_DIR = lite

.PHONY: api gradio test lint format

api:
	uv run python -m lite.api

gui:
	uv run python -m lite.gui

cli:
	uv run python -m lite.cli $(filter-out $@,$(MAKECMDGOALS))

lint:
	uvx ruff check --fix

format:
	uvx ruff format .

# Prevent make from complaining about extra args
%:
	@: