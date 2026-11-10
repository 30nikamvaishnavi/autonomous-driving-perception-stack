.PHONY: install test lint fmt demo-video eda clean
install:
	pip install -e ".[dev]"
test:
	pytest --cov=adpstack --cov-report=term-missing
lint:
	ruff check src tests
fmt:
	ruff check --fix src tests
eda:
	python -m adpstack.data.eda --labels data/bdd100k/labels/det_20/det_train.json --out docs/figures
demo-video:
	python -m adpstack.cli run --source $(SRC) --out outputs/demo.mp4
clean:
	rm -rf outputs .pytest_cache .ruff_cache build *.egg-info
