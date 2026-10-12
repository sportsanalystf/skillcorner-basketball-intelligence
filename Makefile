.PHONY: help data build app test quality docs clean all

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

data:  ## Download the SkillCorner open data (~380 MB, includes Git LFS objects)
	python scripts/fetch_data.py

build:  ## Full pipeline: ingest -> tracking -> features -> models -> warehouse
	python scripts/01_ingest.py
	python scripts/02_tracking.py
	python scripts/03_spatial_features.py
	python scripts/04_shots.py
	python scripts/05_drives_picks.py
	python scripts/06_defense.py
	python scripts/07_players.py
	python scripts/08_team_transition.py
	python scripts/09_warehouse.py
	python scripts/10_quality.py

docs:  ## Regenerate findings.json and the data dictionary
	python scripts/11_findings.py
	python scripts/12_gen_data_dictionary.py

app:  ## Run the Streamlit application
	streamlit run app/main.py

test:  ## Run the test suite
	pytest -q

quality:  ## Run the data-quality suite (exits 1 on any FAIL)
	python scripts/10_quality.py

all: data build docs test  ## Everything

clean:  ## Remove derived artefacts (keeps data/raw)
	rm -rf data/processed data/derived reports/findings.json
