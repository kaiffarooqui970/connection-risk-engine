# One entry point per pipeline step; `make reproduce` runs everything from a fresh clone.
UV ?= uv
PY = $(UV) run python -m
DBT = cd dbt && $(UV) run dbt

.PHONY: setup data dbt collect connections features train fallback policy failure monitor app-bundle app \
        test lint format reproduce clean help

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:           ## install the locked environment
	$(UV) sync

data:            ## download HF subset + weather, validate, load into DuckDB
	$(PY) cre.ingest.hf_loader
	$(PY) cre.ingest.weather

dbt:             ## staging + marts + dbt tests
	$(PY) cre.config
	$(DBT) build --profiles-dir .

collect:         ## one live collection cycle (needs DB_CLIENT_ID / DB_API_KEY; skips cleanly without)
	$(PY) cre.ingest.db_api

connections:     ## Phase 4: candidate connections, labels, lost time, base rates
	$(PY) cre.connections.build

features:        ## Phase 5: leakage-free features, OOF historical rates, splits
	$(PY) cre.features.build

train:           ## Phase 6: baselines + LightGBM (3 seeds) + calibration -> reports/model_metrics.json
	$(PY) cre.models.train

fallback:        ## static fallback model (no live delays) -> reports/fallback_metrics.json
	$(PY) cre.models.fallback

policy:          ## Phase 7: decision simulation vs baselines -> reports/policy_metrics.json
	$(PY) cre.evaluate.policy_sim

failure:         ## Phase 8: failure slices -> reports/failure_analysis.json
	$(PY) cre.evaluate.failure

monitor:         ## Phase 9: monitoring + shadow replay on the test month -> reports/monitoring/
	$(PY) cre.monitor.run --mode replay

app-bundle:      ## derived data for the Streamlit app (no warehouse needed at runtime)
	$(PY) cre.evaluate.export_app

app:             ## run the dashboard locally
	$(UV) run streamlit run app/streamlit_app.py

test:            ## unit + leakage tests (warehouse-dependent tests skip without data)
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests notebooks app
	$(UV) run ruff format --check src tests notebooks app

format:
	$(UV) run ruff check --fix src tests notebooks app
	$(UV) run ruff format src tests notebooks app

reproduce: setup data dbt connections features train fallback policy failure monitor app-bundle test  ## everything, in order

clean:
	rm -rf data/cre.duckdb dbt/target dbt/logs reports/*.json
