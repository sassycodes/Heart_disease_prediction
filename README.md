# Heart Disease Risk Prediction

A production-quality ML pipeline for predicting heart disease risk from the UCI Heart Disease dataset (Kaggle mirror: `redwankarimsony/heart-disease-data`).

**Phase 1: Raw Data Ingestion** — currently implemented.

---

## Quickstart

### Prerequisites
- Python 3.11+
- Docker Desktop (running)
- Git Bash or WSL (for `make` commands on Windows)

### 1. Clone and set up the environment

```bash
# Copy the environment template and fill in your values
cp .env.example .env
# Edit .env with your preferred credentials

# Create the virtual environment and install dependencies
make setup
```

### 2. Place the raw data

Download `heart_disease_uci.csv` from [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) and place it at:

```
data/raw/heart_disease_uci.csv
```

### 3. Start the database

```bash
make db-up       # Starts PostgreSQL 16 via Docker
make db-check    # Verifies the connection (should print ✓)
```

### 4. Ingest the data

```bash
make ingest      # Loads 920 rows into patients_raw; writes artifacts/reports/ingest_report.md
```

### 5. Run tests

```bash
make test        # Runs pytest — no Docker or network required
```

---

## Project Structure

```
heart-disease-risk/
├── AGENTS.md              ← Rules for all agents working on this repo
├── Makefile               ← All commands live here
├── pyproject.toml         ← Dependencies and package metadata
├── docker-compose.yml     ← PostgreSQL 16 container
├── .env.example           ← Credential template (copy to .env)
├── configs/project.yaml   ← Non-secret settings (paths, column names, seed)
├── docs/
│   ├── decisions.md       ← Design decisions and their rationale
│   └── phases/01_ingest.md← Phase 1 spec and acceptance criteria
├── sql/schema.sql         ← DB schema (applied by Docker automatically)
├── src/heart_risk/
│   ├── config.py          ← Loads .env + YAML into a typed Settings object
│   ├── cli.py             ← `heart-risk db-check` and `heart-risk ingest`
│   ├── db/
│   │   ├── engine.py      ← SQLAlchemy engine factory
│   │   ├── tables.py      ← ORM table definitions
│   │   └── repository.py  ← All DB reads/writes (no SQL anywhere else)
│   └── data/
│       └── ingest.py      ← Full ingest pipeline
├── data/raw/              ← CSV lives here (gitignored)
├── artifacts/reports/     ← Generated reports (ingest_report.md)
└── tests/                 ← pytest suite (runs without Docker)
```

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Raw data ingestion | ✅ Complete |
| 2 | Cleaning & EDA | 🔜 Planned |
| 3 | Feature engineering | 🔜 Planned |
| 4 | Model training & evaluation | 🔜 Planned |
| 5 | Streamlit serving | 🔜 Planned |
