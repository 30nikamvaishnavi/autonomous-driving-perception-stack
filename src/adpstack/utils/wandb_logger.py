"""Thin Weights & Biases wrapper that degrades gracefully when W&B is unavailable."""
from __future__ import annotations

from typing import Any

from .logging import get_logger

log = get_logger(__name__)


class WandbLogger:
    """Logs metrics/images/tables to W&B; no-ops if disabled or not installed/logged in."""

    def __init__(
        self, project: str = "adpstack", name: str | None = None,
        config: dict[str, Any] | None = None, enabled: bool = True, tags: list[str] | None = None,
    ) -> None:
        self.run = None
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb

            self._wandb = wandb
            self.run = wandb.init(project=project, name=name, config=config, tags=tags, reinit=True)
        except Exception as exc:  # noqa: BLE001 - offline / missing key must not crash experiments
            log.warning("W&B disabled (%s). Run `wandb login` to enable.", exc)

    @property
    def active(self) -> bool:
        return self.run is not None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.active:
            self._wandb.log(data, step=step)
        else:
            log.info("metrics %s", data)

    def log_image(self, key: str, path_or_array: Any, caption: str | None = None) -> None:
        if self.active:
            self._wandb.log({key: self._wandb.Image(path_or_array, caption=caption)})

    def log_table(self, key: str, columns: list[str], rows: list[list[Any]]) -> None:
        if self.active:
            self._wandb.log({key: self._wandb.Table(columns=columns, data=rows)})
        else:
            log.info("table %s: %s", key, rows)

    def log_artifact(self, path: str, name: str, kind: str = "model") -> None:
        if self.active:
            art = self._wandb.Artifact(name, type=kind)
            art.add_file(path)
            self.run.log_artifact(art)

    def finish(self) -> None:
        if self.active:
            self._wandb.finish()
