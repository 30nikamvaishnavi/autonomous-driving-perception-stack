.PHONY: install test lint fmt demo demo-video eda clean
install:
	pip install -e ".[dev]"
test:
	pytest --cov=adpstack --cov-report=term-missing --cov-fail-under=70
lint:
	ruff check src tests
fmt:
	ruff check --fix src tests
demo:
	python scripts/synthetic_demo.py
eda:
	python -m adpstack.data.eda --labels data/bdd100k/labels/det_20/det_train.json --out docs/figures
demo-video:
	python -m adpstack.cli run --source $(SRC) --out outputs/demo.mp4
clean:
	rm -rf outputs .pytest_cache .ruff_cache build *.egg-info
