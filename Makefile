PY ?= .venv/bin/python

.PHONY: help setup eda run status submission test hooks submit pull check-public

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create .venv with the pinned packages.
	uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt

eda:  ## Regenerate the charts in docs/eda/.
	$(PY) notebooks/eda.py

run:  ## Run every spec that has no result yet.
	$(PY) scripts/exp.py run

status:  ## List experiments and the current best.
	$(PY) scripts/exp.py status

submission:  ## Write a validated submission CSV for the best run.
	$(PY) scripts/exp.py submission

test:  ## Run the tests.
	$(PY) -m pytest -q tests

submit:  ## Queue GPU specs on Hugging Face for molab (commit and push first).
	$(PY) scripts/exp.py submit

pull:  ## Fetch, score and record molab results.
	$(PY) scripts/exp.py pull

hooks:  ## Install the pre-commit check that keeps competition data out of the public repository.
	printf '#!/bin/sh\nexec .venv/bin/python scripts/check_public.py\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

check-public:  ## Scan every tracked file for competition data.
	$(PY) scripts/check_public.py --all
