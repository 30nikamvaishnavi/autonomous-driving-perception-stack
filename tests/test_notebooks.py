import ast
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
REPO_URL = "https://github.com/30nikamvaishnavi/autonomous-driving-perception-stack.git"


def test_notebooks_do_not_detect_colab_via_import_state() -> None:
    for notebook_path in NOTEBOOKS:
        notebook = nbformat.read(notebook_path, as_version=4)
        assert all(
            '"google.colab" in sys.modules' not in cell.source
            for cell in notebook.cells
            if cell.cell_type == "code"
        )


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_cloud_bootstrap(notebook_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)

    assert len(notebook.cells) >= 2
    parameters, bootstrap = notebook.cells[:2]
    assert parameters.cell_type == "code"
    assert bootstrap.cell_type == "code"

    tree = ast.parse(parameters.source)
    repo_urls = [
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REPO_URL" for target in node.targets)
    ]
    assert repo_urls == [REPO_URL]

    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
