#!/bin/bash
set -e

uv sync --frozen

uv run opentelemetry-bootstrap -a requirements | uv pip install --requirement -
