#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv sync
uv run python -c "import lob_project; print('lob_project environment is ready')"
uv run pytest -q

echo "Setup complete. Start Jupyter with: uv run jupyter lab"
