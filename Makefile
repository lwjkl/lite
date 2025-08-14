# Makefile

# Variables
PYTHON = uv run python
UV = uv run
APP_DIR = lite

.PHONY: api gradio test lint format

api:
	uv run python -m lite.api

gradio:
	uv run python -m lite.gradio

cli:
	uv run python -m lite.cli $(filter-out $@,$(MAKECMDGOALS))

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

# Prevent make from complaining about extra args
%:
	@: