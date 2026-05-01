from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import modal
except ImportError:  # pragma: no cover - local environments need not install Modal.
    modal = None


APP_NAME = "latent-space-topology-confirmatory-h1"
VOLUME_NAME = "latent-space-topology-confirmatory-h1"
REMOTE_ROOT = Path("/root")
REMOTE_OUT = REMOTE_ROOT / "outputs" / "confirmatory_h1_v1"
REMOTE_CONFIG = REMOTE_ROOT / "experiment_configs" / "confirmatory_h1_v1.json"
REMOTE_RUNNER = REMOTE_ROOT / "scripts" / "run_confirmatory_sweep.py"
MAX_CONTAINERS = int(os.environ.get("MODAL_CONFIRMATORY_MAX_CONTAINERS", "20"))


if modal is not None:
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_sync(".", frozen=True)
        .add_local_python_source("notebook_utils")
        .add_local_file("scripts/run_confirmatory_sweep.py", str(REMOTE_RUNNER))
        .add_local_file("experiment_configs/confirmatory_h1_v1.json", str(REMOTE_CONFIG))
    )
    app = modal.App(APP_NAME, image=image)
else:
    volume = None
    app = None


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_confirmatory_sweep", REMOTE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {REMOTE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _namespace(stage: str, out: str, smoke: bool, datasets: str | None, max_conditions: int | None, conditions_file: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        config=REMOTE_CONFIG,
        stage=stage,
        executor="local",
        out=Path(out),
        smoke=smoke,
        datasets=datasets,
        max_conditions=max_conditions,
        conditions_file=Path(conditions_file) if conditions_file else None,
        force_cpu=False,
        skip_existing=True,
        modal_max_containers=MAX_CONTAINERS,
    )


if modal is not None:

    @app.function(volumes={str(REMOTE_ROOT / "outputs"): volume}, timeout=21600, cpu=8, memory=32768)
    def run_stage_remote(stage: str, out: str, smoke: bool, datasets: str | None, max_conditions: int | None) -> str:
        runner = _load_runner()
        config = runner.read_json(REMOTE_CONFIG)
        args = _namespace(stage, out, smoke, datasets, max_conditions)
        if stage == "plan":
            runner.stage_plan(config, args)
        elif stage == "encode":
            runner.stage_encode(config, args)
        elif stage == "aggregate":
            runner.stage_aggregate(config, args)
        else:
            raise ValueError(f"run_stage_remote only supports plan/encode/aggregate, got {stage}")
        volume.commit()
        return f"{stage} complete"

    @app.function(volumes={str(REMOTE_ROOT / "outputs"): volume}, timeout=21600, cpu=8, memory=32768)
    def load_conditions_remote(out: str, max_conditions: int | None, conditions_file: str | None) -> list[dict[str, Any]]:
        runner = _load_runner()
        args = _namespace("run", out, False, None, max_conditions, conditions_file)
        rows = runner.condition_rows_to_run(args)
        return rows.to_dict("records")

    @app.function(volumes={str(REMOTE_ROOT / "outputs"): volume}, timeout=21600, cpu=8, memory=32768, max_containers=MAX_CONTAINERS)
    def run_condition_remote(row: dict[str, Any], out: str, smoke: bool) -> str:
        runner = _load_runner()
        config = runner.read_json(REMOTE_CONFIG)
        args = _namespace("run", out, smoke, None, None)
        model_cfg = config["model"]
        data_cfg = config["data"]
        fit_images = int(data_cfg["smoke_fit_images"] if smoke else data_cfg["fit_images"])
        eval_images = int(data_cfg["smoke_eval_images"] if smoke else data_cfg["eval_images"])
        total = fit_images + eval_images
        result_dir = Path(out) / "runs"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{row['result_id']}.json"
        if result_path.exists():
            return f"skip {row['result_id']}"
        if row["stage"] == "positive_control":
            import numpy as np
            import pandas as pd

            cloud = runner.TokenCloud("synthetic", "synthetic", "Synthetic", "circle", np.zeros((1, 2), dtype=np.float32), pd.DataFrame({"split": ["eval"]}), (1, 1), 2, {})
        else:
            cloud = runner.load_any_cache(Path(out), config, row["dataset"], total, int(model_cfg["image_size"]))
            if cloud is None:
                raise FileNotFoundError(f"missing token cache for {row['dataset']}; run Modal encode stage first")
            cloud = runner.ensure_split_metadata(cloud, fit_images=fit_images, eval_images=eval_images)
        payload = runner.run_condition(row, cloud)
        runner.write_json(result_path, payload)
        volume.commit()
        return f"{row['result_id']} {payload['result']['status']}"

    @app.local_entrypoint()
    def main(
        stage: str = "plan",
        out: str = str(REMOTE_OUT),
        smoke: bool = False,
        datasets: str | None = None,
        max_conditions: int | None = None,
        conditions_file: str | None = None,
    ) -> None:
        if stage in {"plan", "encode", "aggregate"}:
            print(run_stage_remote.remote(stage, out, smoke, datasets, max_conditions))
            return
        if stage != "run":
            raise ValueError("Modal entrypoint supports stage=plan|encode|run|aggregate")
        rows = load_conditions_remote.remote(out, max_conditions, conditions_file)
        print(f"loaded {len(rows)} run conditions with max_containers={MAX_CONTAINERS}")
        for message in run_condition_remote.map(rows, kwargs={"out": out, "smoke": smoke}, return_exceptions=True):
            print(message)

else:

    def main(*_: Any, **__: Any) -> None:
        raise SystemExit("Modal is not installed. Install the `modal` package to use this launcher.")
