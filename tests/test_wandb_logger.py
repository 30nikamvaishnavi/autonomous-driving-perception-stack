from adpstack.utils import wandb_logger


def test_wandb_logger_records_config_and_git_commit(monkeypatch):
    class FakeWandb:
        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return object()

    fake_wandb = FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setattr(wandb_logger, "_git_commit_hash", lambda: "abc123")

    wandb_logger.WandbLogger(config={"epochs": 3})

    assert fake_wandb.init_kwargs["config"] == {"epochs": 3, "git_commit": "abc123"}
