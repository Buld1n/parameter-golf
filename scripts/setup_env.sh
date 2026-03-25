#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup_env.sh
#
# Optional overrides:
#   PYTHON=python3.11 VENV_DIR=.venv

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Environment ready."
echo "Activate with: source ${VENV_DIR}/bin/activate"
