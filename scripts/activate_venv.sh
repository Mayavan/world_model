#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run scripts/setup_venv.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
