# Makefile

# Variables
PYTHON = uv run python
UV = uv run
APP_DIR = lite

.PHONY: api gradio test lint format

api:
	uv run api.py

gui:
	uv run gui.py

lint:
	uvx ruff check --fix

format:
	uvx ruff format .

# Prevent make from complaining about extra args
%:
	@:

clean:
	rm -r results/ logs/