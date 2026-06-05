# ─────────────────────────────────────────────────────────────
# Getting started (Windows / PowerShell)
# Reproduces the project environment and runs the pipeline.
# ─────────────────────────────────────────────────────────────

# 1) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) configure environment
Copy-Item .env.example .env

# 4) Run the full streaming fraud-detection pipeline
python main.py

# 5) Run the test suite
pytest tests/ -q

# 6) (Optional) regenerate a standalone dataset
python -m data.generate_data --rows 50000 --out data/transactions.parquet
