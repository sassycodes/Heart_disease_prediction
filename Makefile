# Makefile — Heart Disease Risk Prediction, Phase 1
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   make setup     → create .venv and install dependencies
#   make db-up     → start the PostgreSQL container
#   make db-down   → stop and remove the container
#   make db-check  → verify the DB connection (SELECT 1)
#   make ingest    → load the raw CSV into patients_raw
#   make test      → run the full pytest suite (no Docker needed)
#
# WINDOWS NOTE:
#   This Makefile is designed for Windows with Git Bash or WSL.
#   If you're using plain PowerShell/CMD, run the commands manually instead.
# ─────────────────────────────────────────────────────────────────────────────

PYTHON     := python
VENV       := .venv
VENV_BIN   := $(VENV)/Scripts   # Windows path; change to $(VENV)/bin on Linux/macOS
PIP        := $(VENV_BIN)/pip
HEART_RISK := $(VENV_BIN)/heart-risk

.PHONY: setup db-up db-down db-check ingest test eda clean split train-cv

# ── 1. Create venv and install all dependencies ───────────────────────────────
setup:
	@echo "→ Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "→ Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "✓ setup complete. Activate with: .venv\Scripts\activate"

# ── 2. Start PostgreSQL via Docker Compose ────────────────────────────────────
db-up:
	@echo "→ Starting PostgreSQL container..."
	docker compose up -d --wait
	@echo "✓ db-up complete. Postgres is healthy."

# ── 3. Stop and remove PostgreSQL container ───────────────────────────────────
db-down:
	@echo "→ Stopping PostgreSQL container..."
	docker compose down
	@echo "✓ db-down complete."

# ── 4. Verify the database connection ────────────────────────────────────────
db-check:
	@echo "→ Checking database connection..."
	$(HEART_RISK) db-check && echo "✓ db-check passed." || (echo "✗ db-check FAILED." && exit 1)

# ── 5. Run the full ingest pipeline ──────────────────────────────────────────
ingest:
	@echo "→ Running ingest..."
	$(HEART_RISK) ingest && echo "✓ ingest complete." || (echo "✗ ingest FAILED." && exit 1)

# ── 6. Run pytest (no Docker required) ───────────────────────────────────────
test:
	@echo "→ Running tests..."
	$(VENV_BIN)/pytest tests/ -v --tb=short && echo "✓ All tests passed." || (echo "✗ Tests FAILED." && exit 1)

# ── 7. Run Exploratory Data Analysis ─────────────────────────────────────────
eda:
	@echo "→ Running Exploratory Data Analysis..."
	$(HEART_RISK) eda && echo "✓ EDA complete." || (echo "✗ EDA FAILED." && exit 1)

# ── 8. Run Data Cleaning ─────────────────────────────────────────────────────
clean:
	@echo "→ Running Phase 3 Data Cleaning..."
	$(HEART_RISK) clean && echo "✓ Cleaning complete." || (echo "✗ Cleaning FAILED." && exit 1)

# ── 9. Run Data Split ────────────────────────────────────────────────────────
split:
	@echo "→ Running Phase 4 Data Split..."
	$(HEART_RISK) split && echo "✓ Split complete." || (echo "✗ Split FAILED." && exit 1)

# ── 10. Run CV Model Comparison ──────────────────────────────────────────────
train-cv:
	@echo "→ Running Phase 4 CV Model Comparison..."
	$(HEART_RISK) train-cv && echo "✓ CV Training complete." || (echo "✗ CV Training FAILED." && exit 1)
