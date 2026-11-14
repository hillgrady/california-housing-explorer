#!/bin/bash
# Run the California Housing Explorer locally.
#   bash run_app.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ from https://www.python.org/downloads/ and re-run."; exit 1
fi
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
echo "Installing packages (first run only takes a minute) ..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "Starting the app. It opens at http://localhost:8501  (press Ctrl+C to stop)"
streamlit run app.py
