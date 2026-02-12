#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[dev,procgen]"

echo "Virtual environment ready."
echo "To activate later: source ${VENV_DIR}/bin/activate"
