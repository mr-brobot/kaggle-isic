AWS_REGION ?= $(shell aws configure list | grep region | awk '{print $$3}')

.PHONY: bootstrap bench check format trackio

bootstrap:
	uv run opentelemetry-bootstrap -a requirements | uv pip install --requirement -

bench: bootstrap
	@if [ -z "$(AWS_REGION)" ]; then \
		echo "Warning: AWS region not configured. Traces will not be sent to X-Ray."; \
		uv run opentelemetry-instrument python scripts/bench.py --batches 100 --batch-size 128; \
	else \
		OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://xray.$(AWS_REGION).amazonaws.com/v1/traces \
		uv run opentelemetry-instrument python scripts/bench.py --batches 100 --batch-size 128; \
	fi

check:
	uv run --group dev ruff check .
	uv run --group dev pyrefly check .

format:
	uv run --group dev ruff format .

trackio:
	uv run trackio show