# One entry point per pipeline step; `make reproduce` runs everything from a fresh clone.
UV ?= uv
DBT = cd dbt && $(UV) run dbt

.PHONY: setup data dbt connections features train evaluate policy failure monitor app test lint reproduce clean

setup:
	$(UV) sync

data:            ## download HF subset + weather, validate, load into DuckDB
	$(UV) run python -m cre.ingest.hf_loader
	$(UV) run python -m cre.ingest.weather

dbt:             ## staging + marts + dbt tests
	$(UV) run python -m cre.config
	$(DBT) build --profiles-dir .

collect:         ## one live collection cycle (needs DB_CLIENT_ID / DB_API_KEY)
	$(UV) run python -m cre.ingest.db_api

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests notebooks app
	$(UV) run ruff format --check src tests notebooks app

clean:
	rm -rf data/cre.duckdb dbt/target dbt/logs reports/*.json
