#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Getting started (Linux / macOS)
# Reproduces the project environment and runs the pipeline.
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# 1) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) configure environment
cp .env.example .env

# 4) Run the full streaming fraud-detection pipeline
python main.py

# 5) Run the test suite
pytest tests/ -q

# 6) (Optional) regenerate a standalone dataset
python -m data.generate_data --rows 50000 --out data/transactions.parquet
