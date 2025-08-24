.PHONY: bench check format

bench:
	uv run scripts/bench.py --batches 10 --batch-size 128

check:
	uv run --group dev ruff check .
	uv run --group dev pyrefly check .

format:
	uv run --group dev ruff format .