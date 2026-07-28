.PHONY: setup phase0 phase0-refresh stage0 test lint

setup:
	uv sync --extra dev

phase0:
	uv run crisisforge-phase0

phase0-refresh:
	uv run crisisforge-phase0 --refresh

stage0:
	uv run crisisforge-stage0

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts
