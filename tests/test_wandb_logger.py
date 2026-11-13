import logging
import sys
from types import SimpleNamespace

from adpstack.utils import wandb_logger


def test_wandb_logger_records_config_and_git_commit(monkeypatch):
    class FakeWandb:
        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return object()

    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(wandb_logger, "_git_commit_hash", lambda: "abc123")

    wandb_logger.WandbLogger(config={"epochs": 3})

    assert fake_wandb.init_kwargs["config"] == {"epochs": 3, "git_commit": "abc123"}


def test_wandb_logger_active_methods_forward_values(monkeypatch):
    class FakeRun:
        def __init__(self):
            self.artifacts = []

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path):
            self.files.append(path)

    class FakeWandb:
        def __init__(self):
            self.run = FakeRun()
            self.logged = []
            self.finished = False

        def init(self, **_kwargs):
            return self.run

        def log(self, data, step=None):
            self.logged.append((data, step))

        def Image(self, path, caption=None):
            return (path, caption)

        def Table(self, columns, data):
            return (columns, data)

        def Artifact(self, name, type):
            return FakeArtifact(name, type)

        def finish(self):
            self.finished = True

    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(wandb_logger, "_git_commit_hash", lambda: None)
    logger = wandb_logger.WandbLogger(config={"epochs": 1})

    assert logger.active
    logger.log({"loss": 0.5}, step=1)
    logger.log_image("sample", "frame.jpg", caption="example")
    logger.log_table("metrics", ["epoch"], [[1]])
    logger.log_artifact("best.pt", "model")
    logger.finish()

    assert fake_wandb.logged == [
        ({"loss": 0.5}, 1),
        ({"sample": ("frame.jpg", "example")}, None),
        ({"metrics": (["epoch"], [[1]])}, None),
    ]
    assert fake_wandb.run.artifacts[0].files == ["best.pt"]
    assert fake_wandb.finished


def test_wandb_logger_disabled_methods_are_safe(caplog):
    logger = wandb_logger.WandbLogger(enabled=False)

    assert not logger.active
    with caplog.at_level(logging.INFO, logger="adpstack.utils.wandb_logger"):
        logger.log({"loss": 0.5})
        logger.log_image("sample", "frame.jpg")
        logger.log_table("metrics", ["epoch"], [[1]])
        logger.log_artifact("best.pt", "model")
        logger.finish()

    assert "metrics" in caplog.text
    assert "table metrics" in caplog.text


def test_missing_git_metadata_is_reported(monkeypatch, caplog):
    monkeypatch.setattr(
        wandb_logger.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="not a checkout"),
    )

    with caplog.at_level(logging.WARNING, logger="adpstack.utils.wandb_logger"):
        assert wandb_logger._git_commit_hash() is None

    assert "Could not determine Git commit hash" in caplog.text
