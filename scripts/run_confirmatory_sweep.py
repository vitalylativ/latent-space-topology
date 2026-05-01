#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebook_utils.encoder_explorer import (  # noqa: E402
    DEFAULT_IMAGE_DIR,
    TokenCloud,
    build_token_metadata,
    center_crop_resize,
    choose_device,
    chunked,
    flatten_spatial,
    l2_normalize,
    load_project_images,
    pil_to_vae_tensor,
    pipeline_scaled_latents,
    rgb,
    safe_to_device,
    sample_indices,
)
from notebook_utils.flux_tda import farthest_point_indices  # noqa: E402


CONFIG_PATH = ROOT / "experiment_configs" / "confirmatory_h1_v1.json"
DEFAULT_OUT = ROOT / "outputs" / "confirmatory_h1_v1"
REMOTE_MODAL_ROOT = Path("/root") / "outputs"
LEGACY_CACHE_DIR = ROOT / "outputs" / "cycle_hunt" / "data_sweep" / "cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confirmatory FLUX latent-token H1 sweep.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--stage", choices=["plan", "encode", "run", "aggregate", "all"], default="plan")
    parser.add_argument("--executor", choices=["local", "modal"], default="local")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="Use smoke data/seeds and only the first dataset.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset override.")
    parser.add_argument("--max-conditions", type=int, default=None, help="Limit run stage for local smoke/debug.")
    parser.add_argument("--conditions-file", type=Path, default=None, help="Optional CSV of conditions to run instead of <out>/conditions.csv.")
    parser.add_argument("--modal-max-containers", type=int, default=20, help="Cap Modal run-condition concurrency.")
    parser.add_argument("--force-cpu", action="store_true", default=os.environ.get("TOKENIZER_FORCE_CPU", "0") == "1")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def config_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(config)
    cleaned.get("metadata", {}).pop("config_sha256", None)
    return cleaned


def config_hash(config: dict[str, Any]) -> str:
    text = json.dumps(config_for_hash(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(payload: dict[str, Any], length: int = 16) -> str:
    text = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def value_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    if pd.isna(value):
        return default
    return float(value)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ["numpy", "pandas", "sklearn", "scipy", "ripser", "gudhi", "torch"]:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "not-installed"
    return versions


def bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def balanced_indices(targets: list[int] | np.ndarray, n: int, seed: int, exclude: set[int] | None = None) -> np.ndarray:
    targets = np.asarray(targets)
    exclude = exclude or set()
    rng = np.random.default_rng(seed)
    classes = sorted(np.unique(targets).tolist())
    per_class = max(1, int(np.ceil(n / max(len(classes), 1))))
    chosen: list[int] = []
    for class_id in classes:
        idx = [int(i) for i in np.flatnonzero(targets == class_id) if int(i) not in exclude]
        rng.shuffle(idx)
        chosen.extend(idx[:per_class])
    rng.shuffle(chosen)
    if len(chosen) < n:
        remaining = [int(i) for i in range(len(targets)) if int(i) not in exclude and int(i) not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: n - len(chosen)])
    return np.asarray(chosen[:n], dtype=int)


def load_beans_images(total: int, fit_n: int, seed: int) -> tuple[list[Image.Image], pd.DataFrame]:
    try:
        from datasets import load_dataset

        ds = load_dataset("beans", split="train")
        labels = np.asarray(ds["labels"], dtype=int)
        # Current local cache was built from the first 48 rows; reserve them from held-out full runs when possible.
        exclude = set(range(min(48, len(labels)))) if len(labels) > total + 48 else set()
        fit_idx = balanced_indices(labels, fit_n, seed, exclude=exclude)
        eval_idx = balanced_indices(labels, total - fit_n, seed + 17, exclude=exclude | set(fit_idx.tolist()))
        names = ds.features["labels"].names
        rows: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        split_indices = [("fit", int(i)) for i in fit_idx] + [("eval", int(i)) for i in eval_idx]
        for image_id, (split, dataset_index) in enumerate(split_indices):
            item = ds[dataset_index]
            image = rgb(item["image"])
            images.append(image)
            rows.append(
                {
                    "image_id": image_id,
                    "dataset": "beans",
                    "source": "hf_beans",
                    "split": split,
                    "dataset_index": dataset_index,
                    "label": names[int(item["labels"])],
                    "path": None,
                    "width": image.width,
                    "height": image.height,
                }
            )
        return images, pd.DataFrame(rows)
    except Exception:
        images, meta = load_project_images(total, DEFAULT_IMAGE_DIR)
        meta = meta.copy()
        meta.insert(0, "dataset", "beans")
        meta["split"] = ["fit" if i < fit_n else "eval" for i in range(len(meta))]
        meta["source"] = "beans_local_fallback"
        return images, meta


def torchvision_root() -> Path:
    root = ROOT / "outputs" / "confirmatory_h1_v1" / "torchvision_data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_torchvision_images(name: str, total: int, fit_n: int, seed: int) -> tuple[list[Image.Image], pd.DataFrame]:
    if name == "cifar10":
        from torchvision.datasets import CIFAR10

        ds = CIFAR10(root=str(torchvision_root()), train=True, download=True)
        labels = np.asarray(ds.targets)
        classes = ds.classes
    elif name == "fashion_mnist":
        from torchvision.datasets import FashionMNIST

        ds = FashionMNIST(root=str(torchvision_root()), train=True, download=True)
        labels = ds.targets.numpy() if hasattr(ds.targets, "numpy") else np.asarray(ds.targets)
        classes = ds.classes
    elif name == "stl10":
        from torchvision.datasets import STL10

        ds = STL10(root=str(torchvision_root()), split="train", download=True)
        labels = np.asarray(ds.labels)
        classes = ds.classes
    else:
        raise ValueError(f"unknown torchvision dataset: {name}")

    fit_idx = balanced_indices(labels, fit_n, seed)
    eval_idx = balanced_indices(labels, total - fit_n, seed + 17, exclude=set(fit_idx.tolist()))
    images: list[Image.Image] = []
    rows: list[dict[str, Any]] = []
    split_indices = [("fit", int(i)) for i in fit_idx] + [("eval", int(i)) for i in eval_idx]
    for image_id, (split, dataset_index) in enumerate(split_indices):
        image, label_id = ds[dataset_index]
        image = rgb(image)
        images.append(image)
        rows.append(
            {
                "image_id": image_id,
                "dataset": name,
                "source": name,
                "split": split,
                "dataset_index": dataset_index,
                "label": classes[int(label_id)] if classes else str(label_id),
                "path": None,
                "width": image.width,
                "height": image.height,
            }
        )
    return images, pd.DataFrame(rows)


def load_dataset_images(name: str, total: int, fit_n: int, seed: int) -> tuple[list[Image.Image], pd.DataFrame]:
    if name == "beans":
        return load_beans_images(total, fit_n, seed)
    if name in {"cifar10", "fashion_mnist", "stl10"}:
        return load_torchvision_images(name, total, fit_n, seed)
    raise ValueError(f"unknown dataset: {name}")


def cache_stem(out: Path, dataset: str, n_images: int, image_size: int) -> Path:
    return out / "tokens" / f"flux_vae_{dataset}_n{n_images}_px{image_size}"


def load_cloud_cache(path: Path, dataset: str) -> TokenCloud | None:
    npz_path = path.with_suffix(".npz")
    meta_path = path.with_name(path.name + "_token_metadata.csv")
    notes_path = path.with_name(path.name + "_notes.json")
    if not (npz_path.exists() and meta_path.exists() and notes_path.exists()):
        return None
    data = np.load(npz_path)
    tokens = data["tokens"].astype(np.float32)
    metadata = pd.read_csv(meta_path)
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    return TokenCloud(
        name="flux_vae",
        model_id="diffusers/FLUX.1-vae",
        family="AutoencoderKL",
        token_kind="posterior_mean",
        tokens=tokens,
        token_metadata=metadata,
        grid_shape=tuple(int(x) for x in data["grid_shape"].tolist()),
        channel_dim=int(data["channel_dim"][0]),
        notes={**notes, "dataset": dataset},
    )


def save_cloud_cache(out: Path, dataset: str, cloud: TokenCloud, n_images: int, image_size: int) -> None:
    stem = cache_stem(out, dataset, n_images, image_size)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stem.with_suffix(".npz"),
        tokens=cloud.tokens.astype(np.float32),
        grid_shape=np.asarray(cloud.grid_shape, dtype=np.int32),
        channel_dim=np.asarray([cloud.channel_dim], dtype=np.int32),
    )
    cloud.token_metadata.to_csv(stem.with_name(stem.name + "_token_metadata.csv"), index=False)
    stem.with_name(stem.name + "_notes.json").write_text(json.dumps(json_ready(cloud.notes), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_any_cache(out: Path, config: dict[str, Any], dataset: str, n_images: int, image_size: int) -> TokenCloud | None:
    current = load_cloud_cache(cache_stem(out, dataset, n_images, image_size), dataset)
    if current is not None:
        return current

    aliases = config.get("data", {}).get("existing_cache_aliases", {})
    alias = aliases.get(dataset, dataset)
    legacy_stems = [
        LEGACY_CACHE_DIR / f"flux_vae_{alias}_n48_px{image_size}",
        ROOT / "outputs" / "cycle_hunt" / "beans_param_sweep" / "flux_cloud_48_images",
    ]
    for stem in legacy_stems:
        cloud = load_cloud_cache(stem, dataset)
        if cloud is not None:
            return cloud
    return None


def encode_flux(images: list[Image.Image], metadata: pd.DataFrame, image_size: int, batch_size: int, force_cpu: bool) -> TokenCloud:
    import torch
    from diffusers import AutoencoderKL

    device = choose_device(force_cpu=force_cpu)
    model = AutoencoderKL.from_pretrained("diffusers/FLUX.1-vae", torch_dtype=torch.float32, use_safetensors=True)
    model, used_device = safe_to_device(model.eval(), device)
    latent_chunks: list[Any] = []
    scaled_means: list[float] = []
    scaled_stds: list[float] = []
    with torch.inference_mode():
        for _, batch_images in chunked(images, batch_size):
            batch = pil_to_vae_tensor(batch_images, image_size).to(used_device, dtype=torch.float32)
            z = model.encode(batch).latent_dist.mean
            z_scaled = pipeline_scaled_latents(model, z)
            latent_chunks.append(z.detach().cpu())
            scaled_means.append(float(z_scaled.detach().cpu().mean()))
            scaled_stds.append(float(z_scaled.detach().cpu().std()))
    z_all = torch.cat(latent_chunks, dim=0)
    tokens, grid_shape, channel_dim = flatten_spatial(z_all)
    token_metadata = build_token_metadata(metadata, "diffusers/FLUX.1-vae", "AutoencoderKL", "posterior_mean", grid_shape)
    token_metadata = token_metadata.merge(metadata[["image_id", "dataset", "split", "dataset_index", "path"]], on="image_id", how="left", suffixes=("", "_image"))
    notes = {
        "device": used_device,
        "latent_shape_bchw": tuple(int(x) for x in z_all.shape),
        "scaling_factor": float(getattr(model.config, "scaling_factor", 1.0) or 1.0),
        "shift_factor": getattr(model.config, "shift_factor", None),
        "scaled_latent_mean": float(np.mean(scaled_means)),
        "scaled_latent_std": float(np.mean(scaled_stds)),
    }
    return TokenCloud("flux_vae", "diffusers/FLUX.1-vae", "AutoencoderKL", "posterior_mean", tokens.astype(np.float32), token_metadata, grid_shape, channel_dim, notes)


def ensure_split_metadata(cloud: TokenCloud, fit_images: int, eval_images: int) -> TokenCloud:
    metadata = cloud.token_metadata.copy()
    if "dataset" not in metadata.columns:
        metadata["dataset"] = cloud.notes.get("dataset", metadata.get("source", pd.Series(["unknown"] * len(metadata))).astype(str))
    if "split" not in metadata.columns:
        image_ids = sorted(metadata["image_id"].drop_duplicates().astype(int).tolist())
        split_by_image = {image_id: ("fit" if i < fit_images else "eval") for i, image_id in enumerate(image_ids[: fit_images + eval_images])}
        metadata["split"] = metadata["image_id"].map(split_by_image)
    keep_images = sorted(metadata[metadata["split"].isin(["fit", "eval"])]["image_id"].drop_duplicates().astype(int).tolist())[: fit_images + eval_images]
    keep = metadata["image_id"].isin(keep_images)
    return TokenCloud(
        cloud.name,
        cloud.model_id,
        cloud.family,
        cloud.token_kind,
        cloud.tokens[keep.to_numpy()].astype(np.float32, copy=False),
        metadata.loc[keep].reset_index(drop=True),
        cloud.grid_shape,
        cloud.channel_dim,
        cloud.notes,
    )


def split_tokens(cloud: TokenCloud) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    meta = cloud.token_metadata.reset_index(drop=True)
    if "split" not in meta.columns:
        raise ValueError("token cache is missing split metadata")
    fit_mask = meta["split"].astype(str) == "fit"
    eval_mask = meta["split"].astype(str) == "eval"
    fit = cloud.tokens[fit_mask.to_numpy()].astype(np.float32, copy=False)
    eval_tokens = cloud.tokens[eval_mask.to_numpy()].astype(np.float32, copy=False)
    eval_meta = meta.loc[eval_mask].reset_index(drop=True)
    if len(fit) == 0 or len(eval_tokens) == 0:
        raise ValueError("fit/eval split produced an empty token set")
    return fit, eval_tokens, eval_meta


def kth_density_order_exact(x: np.ndarray, k: int, max_candidates: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    candidate_idx = sample_indices(len(x), min(max_candidates, len(x)), seed=seed)
    candidates = x[candidate_idx]
    n_neighbors = min(k + 1, len(candidates))
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(candidates).kneighbors(candidates)
    kth = distances[:, -1]
    order = np.argsort(kth)
    return candidate_idx[order], kth[order]


def kth_density_order_anchors(x: np.ndarray, k: int, max_candidates: int, seed: int, n_probes: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    candidate_idx = sample_indices(len(x), min(max_candidates, len(x)), seed=seed)
    candidates = x[candidate_idx].astype(np.float32, copy=False)
    probe_count = min(n_probes, len(candidates))
    probe_local = sample_indices(len(candidates), probe_count, seed=seed + 997)
    probes = candidates[probe_local].astype(np.float32, copy=False)
    probe_norms = np.sum(probes * probes, axis=1)
    kth_index = min(k, probe_count - 1)
    kth_sq = np.empty(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        chunk = candidates[start:stop]
        chunk_norms = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances_sq = np.maximum(chunk_norms + probe_norms[None, :] - 2.0 * (chunk @ probes.T), 0.0)
        kth_sq[start:stop] = np.partition(distances_sq, kth_index, axis=1)[:, kth_index]
    order = np.argsort(kth_sq)
    return candidate_idx[order], np.sqrt(kth_sq[order])


def dense_fps_indices(x: np.ndarray, n_dense: int, n_landmarks: int, k_density: int, max_candidates: int, seed: int) -> np.ndarray:
    density_order, _ = kth_density_order_exact(x, k=k_density, max_candidates=max_candidates, seed=seed)
    dense_idx = density_order[: min(n_dense, len(density_order))]
    local_landmarks = farthest_point_indices(x[dense_idx], n_landmarks=n_landmarks, seed=seed)
    return dense_idx[local_landmarks]


def shuffle_columns(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = x.copy()
    for col in range(out.shape[1]):
        rng.shuffle(out[:, col])
    return out


def gaussian_pool(fit_raw: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = sample_indices(len(fit_raw), min(12000, len(fit_raw)), seed=seed)
    x = fit_raw[idx].astype(np.float32)
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-5
    return rng.multivariate_normal(mean, cov, size=n).astype(np.float32)


def fit_projection(fit_raw: np.ndarray, params: dict[str, Any], seed: int):
    projection = params.get("projection")
    if projection == "pca_sphere":
        dim = int(params["pca_dim"])
        idx = sample_indices(len(fit_raw), min(50000, len(fit_raw)), seed=seed)
        return PCA(n_components=min(dim, fit_raw.shape[1], len(idx) - 1), random_state=seed).fit(fit_raw[idx].astype(np.float32))
    if projection == "random_projection_sphere":
        dim = int(params.get("projection_dim", 8))
        idx = sample_indices(len(fit_raw), min(50000, len(fit_raw)), seed=seed)
        return GaussianRandomProjection(n_components=dim, random_state=seed).fit(fit_raw[idx].astype(np.float32))
    return None


def apply_projection(raw: np.ndarray, model: Any, params: dict[str, Any]) -> np.ndarray:
    projection = params.get("projection")
    raw = raw.astype(np.float32, copy=False)
    if projection in {"pca_sphere", "random_projection_sphere"}:
        z = model.transform(raw).astype(np.float32)
        return l2_normalize(z).astype(np.float32)
    if projection == "sphere":
        return l2_normalize(raw).astype(np.float32)
    return raw


def uniform_pool(dim: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=(n, dim)).astype(np.float32)).astype(np.float32)


def rips_diagram(points: np.ndarray, maxdim: int, distance_quantile: float) -> dict[str, Any]:
    import ripser

    if len(points) < 2:
        diagrams = [np.empty((0, 2), dtype=float) for _ in range(maxdim + 1)]
        return {"diagrams": diagrams, "threshold": 0.0, "normalizer": 0.0}
    distances = pdist(points, metric="euclidean")
    threshold = float(np.max(distances) if distance_quantile >= 1.0 else np.quantile(distances, distance_quantile))
    result = ripser.ripser(points, maxdim=maxdim, thresh=threshold)
    return {"diagrams": result["dgms"], "threshold": threshold, "normalizer": threshold}


def witness_diagram(witnesses: np.ndarray, landmarks: np.ndarray, max_alpha_square: float, maxdim: int) -> dict[str, Any]:
    import gudhi

    complex_ = gudhi.EuclideanWitnessComplex(landmarks=landmarks, witnesses=witnesses)
    st = complex_.create_simplex_tree(max_alpha_square=max_alpha_square, limit_dimension=maxdim)
    st.compute_persistence()
    filtration_values = [float(value) for _, value in st.get_filtration()]
    filtration_max_sq = max(filtration_values) if filtration_values else 0.0
    diagrams = []
    for dim in range(maxdim + 1):
        intervals = np.asarray(st.persistence_intervals_in_dimension(dim), dtype=float)
        diagrams.append(intervals.reshape((-1, 2)) if intervals.size else np.empty((0, 2), dtype=float))
    return {
        "diagrams": diagrams,
        "threshold": filtration_max_sq,
        "normalizer": float(np.sqrt(max(filtration_max_sq, 0.0))),
        "num_vertices": st.num_vertices(),
        "num_simplices": st.num_simplices(),
    }


def summarize_diagrams(diagrams: list[np.ndarray], normalizer: float, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim, diagram in enumerate(diagrams):
        if diagram.size == 0:
            rows.append({"dim": dim, "n_features": 0, "n_finite": 0, "max_persistence": 0.0, "max_persistence_norm": 0.0, "threshold_or_filtration": threshold})
            continue
        finite = diagram[np.isfinite(diagram[:, 1])]
        if len(finite):
            if normalizer > 0 and threshold > 0 and threshold > normalizer:
                # Witness filtrations are in alpha^2; report alpha-scale persistence.
                persistence = np.sqrt(np.maximum(finite[:, 1], 0.0)) - np.sqrt(np.maximum(finite[:, 0], 0.0))
            else:
                persistence = finite[:, 1] - finite[:, 0]
            max_persistence = float(persistence.max()) if len(persistence) else 0.0
        else:
            max_persistence = 0.0
        rows.append(
            {
                "dim": dim,
                "n_features": int(len(diagram)),
                "n_finite": int(len(finite)),
                "max_persistence": max_persistence,
                "max_persistence_norm": float(max_persistence / normalizer) if normalizer > 0 else 0.0,
                "threshold_or_filtration": float(threshold),
            }
        )
    return rows


def synthetic_circle(seed: int, dim: int, n: int, noise: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.zeros((n, dim), dtype=np.float32)
    x[:, 0] = np.cos(theta)
    x[:, 1] = np.sin(theta)
    x += rng.normal(scale=noise, size=x.shape).astype(np.float32)
    return x


def build_rips_points(cloud: TokenCloud, row: dict[str, Any]) -> np.ndarray:
    if row["sample_kind"] == "positive_control":
        return synthetic_circle(
            int(row["seed"]),
            int(value_or_default(row.get("ambient_dim"), 8)),
            int(value_or_default(row.get("n_points"), 95)),
            value_or_default(row.get("noise"), 0.025),
        )

    fit_raw, eval_raw, _ = split_tokens(cloud)
    seed = int(row["seed"])
    params = row
    model = fit_projection(fit_raw, params, seed)
    sample_kind = row["sample_kind"]
    pool_n = min(len(eval_raw), max(int(row.get("max_candidates", 20000)), int(row.get("n_dense", 1600))))

    if sample_kind == "observed":
        transformed = apply_projection(eval_raw, model, params)
        idx = dense_fps_indices(transformed, int(row["n_dense"]), int(row["n_landmarks"]), int(row["k_density"]), int(row["max_candidates"]), seed)
        return transformed[idx]
    if sample_kind == "random_tokens":
        transformed = apply_projection(eval_raw, model, params)
        idx = sample_indices(len(transformed), int(row["n_landmarks"]), seed + 101)
        return transformed[idx]
    if sample_kind == "uniform_sphere":
        dim = int(row.get("pca_dim", row.get("projection_dim", cloud.channel_dim)))
        pool = uniform_pool(dim, pool_n, seed + 102)
        idx = dense_fps_indices(pool, int(row["n_dense"]), int(row["n_landmarks"]), int(row["k_density"]), min(int(row["max_candidates"]), len(pool)), seed + 102)
        return pool[idx]
    if sample_kind == "channel_shuffle":
        transformed = apply_projection(shuffle_columns(eval_raw, seed + 103), model, params)
        idx = dense_fps_indices(transformed, int(row["n_dense"]), int(row["n_landmarks"]), int(row["k_density"]), int(row["max_candidates"]), seed + 103)
        return transformed[idx]
    if sample_kind == "matched_gaussian":
        transformed = apply_projection(gaussian_pool(fit_raw, pool_n, seed + 104), model, params)
        idx = dense_fps_indices(transformed, int(row["n_dense"]), int(row["n_landmarks"]), int(row["k_density"]), min(int(row["max_candidates"]), len(transformed)), seed + 104)
        return transformed[idx]
    if sample_kind == "positive_control":
        return synthetic_circle(seed, int(row["ambient_dim"]), int(row["n_points"]), float(row["noise"]))
    raise ValueError(f"unknown sample kind for Rips: {sample_kind}")


def witness_sample(cloud: TokenCloud, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    fit_raw, eval_raw, _ = split_tokens(cloud)
    seed = int(row["seed"])
    sample_kind = row["sample_kind"]
    n_witnesses = int(row["n_witnesses"])
    n_landmarks = int(row["n_landmarks"])
    k_density = int(row["k_density"])
    max_candidates = int(row["max_candidates"])
    density_mode = str(row.get("density_mode", "exact"))
    density_probes = int(row.get("density_probes", 4096))
    density_chunk_size = int(row.get("density_chunk_size", 2048))

    if sample_kind == "observed":
        pool = l2_normalize(eval_raw).astype(np.float32)
        offset = seed
        use_density = True
    elif sample_kind == "random_tokens":
        pool = l2_normalize(eval_raw).astype(np.float32)
        idx = sample_indices(len(pool), min(n_witnesses, len(pool)), seed + 101)
        witnesses = pool[idx]
        local = farthest_point_indices(witnesses, n_landmarks=n_landmarks, seed=seed + 101)
        return witnesses, witnesses[local]
    elif sample_kind == "uniform_sphere":
        witnesses = uniform_pool(cloud.channel_dim, n_witnesses, seed + 102)
        local = farthest_point_indices(witnesses, n_landmarks=n_landmarks, seed=seed + 102)
        return witnesses, witnesses[local]
    elif sample_kind == "channel_shuffle":
        pool = l2_normalize(shuffle_columns(eval_raw, seed + 103)).astype(np.float32)
        offset = seed + 103
        use_density = True
    elif sample_kind == "matched_gaussian":
        pool = l2_normalize(gaussian_pool(fit_raw, min(len(eval_raw), max_candidates), seed + 104)).astype(np.float32)
        offset = seed + 104
        use_density = True
    else:
        raise ValueError(f"unknown sample kind for witness: {sample_kind}")

    if use_density:
        if density_mode == "anchors":
            order, _ = kth_density_order_anchors(pool, k_density, max_candidates, offset, density_probes, density_chunk_size)
        else:
            order, _ = kth_density_order_exact(pool, k_density, max_candidates, offset)
        witness_idx = order[: min(n_witnesses, len(order))]
        witnesses = pool[witness_idx]
        local = farthest_point_indices(witnesses, n_landmarks=n_landmarks, seed=offset)
        return witnesses, witnesses[local]
    raise AssertionError("unreachable")


def run_condition(row: dict[str, Any], cloud: TokenCloud) -> dict[str, Any]:
    started = time.perf_counter()
    result_base = {
        "condition_id": row["condition_id"],
        "result_id": row["result_id"],
        "dataset": row["dataset"],
        "pipeline": row["pipeline"],
        "stage": row["stage"],
        "sample_kind": row["sample_kind"],
        "control_kind": row["control_kind"],
        "seed": int(row["seed"]),
        "status": "ok",
    }
    try:
        family = row["family"]
        if family in {"rips", "positive_control"}:
            points = build_rips_points(cloud, row)
            tda = rips_diagram(points, maxdim=int(row["maxdim"]), distance_quantile=value_or_default(row.get("distance_quantile"), 1.0))
            n_points = len(points)
            ambient_dim = points.shape[1]
            extra = {}
        elif family == "witness":
            witnesses, landmarks = witness_sample(cloud, row)
            max_alpha = float("inf") if str(row.get("max_alpha_square", "inf")) == "inf" else float(row["max_alpha_square"])
            tda = witness_diagram(witnesses, landmarks, max_alpha_square=max_alpha, maxdim=int(row["maxdim"]))
            n_points = len(landmarks)
            ambient_dim = landmarks.shape[1]
            extra = {"n_witnesses_actual": len(witnesses), "num_simplices": tda.get("num_simplices"), "num_vertices": tda.get("num_vertices")}
        else:
            raise ValueError(f"unknown family: {family}")

        rows = []
        for summary in summarize_diagrams(tda["diagrams"], normalizer=float(tda["normalizer"]), threshold=float(tda["threshold"])):
            rows.append({**result_base, **summary, "n_points": int(n_points), "ambient_dim": int(ambient_dim), "runtime_seconds": time.perf_counter() - started, **extra})
        return {"result": {**result_base, "runtime_seconds": time.perf_counter() - started, "summaries": rows, "diagrams": [diagram.tolist() for diagram in tda["diagrams"]]}}
    except Exception as exc:
        return {
            "result": {
                **result_base,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - started,
                "summaries": [],
            }
        }


def make_variant(pipeline: str, stage: str, base: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    variant = copy.deepcopy(base)
    variant.update(overrides or {})
    variant["pipeline"] = overrides.get("pipeline", pipeline) if overrides else pipeline
    variant["stage"] = stage
    return variant


def condition_table(config: dict[str, Any], smoke: bool, datasets_override: list[str] | None = None) -> pd.DataFrame:
    datasets = datasets_override or config["data"]["datasets"]
    if smoke:
        datasets = datasets[:1]
    seeds = config["seeds"]["smoke"] if smoke else config["seeds"]["confirmatory"]
    sample_kinds = ["observed", *config["controls"]]
    rows: list[dict[str, Any]] = []

    variants: list[dict[str, Any]] = []
    variants.append(make_variant("pca8_sphere_rips", "primary", config["pipelines"]["pca8_sphere_rips"]))
    variants.append(make_variant("s15_witness", "primary", config["pipelines"]["s15_witness"]))

    pca_primary = config["pipelines"]["pca8_sphere_rips"]
    for key, values in config["stability"]["pca8_sphere_rips"]["one_at_a_time"].items():
        for value in values:
            variants.append(make_variant("pca8_sphere_rips", "stability", pca_primary, {key: value, "variant": f"{key}={value}"}))
    for extra in config["stability"]["pca8_sphere_rips"]["extra_variants"]:
        variants.append(make_variant(extra["pipeline"], "stability", pca_primary, {**extra, "variant": extra["pipeline"]}))

    witness_primary = config["pipelines"]["s15_witness"]
    for key, values in config["stability"]["s15_witness"]["one_at_a_time"].items():
        for value in values:
            variants.append(make_variant("s15_witness", "stability", witness_primary, {key: value, "variant": f"{key}={value}"}))
    for extra in config["stability"]["s15_witness"]["extra_variants"]:
        variants.append(make_variant(extra["pipeline"], "stability", witness_primary, {**extra, "variant": extra["pipeline"]}))

    positive = make_variant("synthetic_circle", "positive_control", config["pipelines"]["synthetic_circle"])
    positive["pipeline"] = "synthetic_circle"
    variants.append(positive)

    for dataset in datasets:
        for seed in seeds:
            for variant in variants:
                kinds = ["positive_control"] if variant["stage"] == "positive_control" else sample_kinds
                base_key = {
                    "dataset": dataset,
                    "seed": seed,
                    "pipeline": variant["pipeline"],
                    "stage": variant["stage"],
                    "variant": variant.get("variant", "primary"),
                }
                condition_id = stable_id(base_key)
                for sample_kind in kinds:
                    row = {
                        **variant,
                        "dataset": dataset,
                        "seed": int(seed),
                        "condition_id": condition_id,
                        "sample_kind": sample_kind,
                        "control_kind": "observed" if sample_kind in {"observed", "positive_control"} else sample_kind,
                    }
                    row["result_id"] = f"{condition_id}__{sample_kind}"
                    rows.append(row)
    return pd.DataFrame(rows).fillna("")


def write_manifest(out: Path, config: dict[str, Any], args: argparse.Namespace, conditions: pd.DataFrame) -> None:
    manifest = {
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": package_versions(),
        "output_dir": str(out),
        "smoke": bool(args.smoke),
        "conditions": int(len(conditions)),
        "datasets": sorted(conditions["dataset"].unique().tolist()) if not conditions.empty else [],
    }
    write_json(out / "manifest.json", manifest)


def stage_plan(config: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()] if args.datasets else None
    conditions = condition_table(config, smoke=args.smoke, datasets_override=datasets)
    args.out.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(args.out / "conditions.csv", index=False)
    write_manifest(args.out, config, args, conditions)
    print(f"wrote {args.out / 'conditions.csv'} ({len(conditions)} rows)")
    return conditions


def stage_encode(config: dict[str, Any], args: argparse.Namespace) -> None:
    data_cfg = config["data"]
    model_cfg = config["model"]
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()] if args.datasets else data_cfg["datasets"]
    if args.smoke:
        datasets = datasets[:1]
    fit_images = int(data_cfg["smoke_fit_images"] if args.smoke else data_cfg["fit_images"])
    eval_images = int(data_cfg["smoke_eval_images"] if args.smoke else data_cfg["eval_images"])
    total = fit_images + eval_images
    seed = int(config["seeds"]["smoke"][0] if args.smoke else config["seeds"]["confirmatory"][0])
    manifest_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        existing = load_any_cache(args.out, config, dataset, total, int(model_cfg["image_size"]))
        if existing is not None:
            cloud = ensure_split_metadata(existing, fit_images=fit_images, eval_images=eval_images)
            save_cloud_cache(args.out, dataset, cloud, len(cloud.token_metadata["image_id"].drop_duplicates()), int(model_cfg["image_size"]))
            print(f"cached {dataset}: {cloud.tokens.shape}")
        else:
            images, metadata = load_dataset_images(dataset, total=total, fit_n=fit_images, seed=seed)
            cloud = encode_flux(images, metadata, int(model_cfg["image_size"]), int(model_cfg["batch_size"]), args.force_cpu)
            save_cloud_cache(args.out, dataset, cloud, total, int(model_cfg["image_size"]))
            print(f"encoded {dataset}: {cloud.tokens.shape}")
        meta = cloud.token_metadata.drop_duplicates("image_id")[["image_id", "dataset", "split", "label"]].copy()
        meta["cache_dataset"] = dataset
        manifest_rows.extend(meta.to_dict("records"))
    pd.DataFrame(manifest_rows).to_csv(args.out / "dataset_manifest.csv", index=False)


def condition_rows_to_run(args: argparse.Namespace) -> pd.DataFrame:
    conditions_path = args.conditions_file or (args.out / "conditions.csv")
    if not conditions_path.exists():
        raise FileNotFoundError(f"missing {conditions_path}; run --stage plan first")
    conditions = pd.read_csv(conditions_path)
    if args.max_conditions is not None:
        conditions = conditions.head(args.max_conditions)
    return conditions


def stage_run(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.executor == "modal":
        raise SystemExit("Use `modal run scripts/modal_confirmatory_sweep.py -- --config ... --out ...` for Modal execution.")
    conditions = condition_rows_to_run(args)
    model_cfg = config["model"]
    data_cfg = config["data"]
    fit_images = int(data_cfg["smoke_fit_images"] if args.smoke else data_cfg["fit_images"])
    eval_images = int(data_cfg["smoke_eval_images"] if args.smoke else data_cfg["eval_images"])
    total = fit_images + eval_images
    result_dir = args.out / "runs"
    result_dir.mkdir(parents=True, exist_ok=True)
    cloud_cache: dict[str, TokenCloud] = {}
    for _, row in conditions.iterrows():
        result_path = result_dir / f"{row['result_id']}.json"
        if args.skip_existing and result_path.exists():
            continue
        dataset = str(row["dataset"])
        if row["stage"] == "positive_control":
            cloud = TokenCloud("synthetic", "synthetic", "Synthetic", "circle", np.zeros((1, 2), dtype=np.float32), pd.DataFrame({"split": ["eval"]}), (1, 1), 2, {})
        else:
            if dataset not in cloud_cache:
                cloud = load_any_cache(args.out, config, dataset, total, int(model_cfg["image_size"]))
                if cloud is None:
                    raise FileNotFoundError(f"missing token cache for {dataset}; run --stage encode first")
                cloud_cache[dataset] = ensure_split_metadata(cloud, fit_images=fit_images, eval_images=eval_images)
            cloud = cloud_cache[dataset]
        payload = run_condition(row.to_dict(), cloud)
        write_json(result_path, payload)
        status = payload["result"]["status"]
        print(f"{row['result_id']} {status}")


def load_run_results(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    diagram_rows: list[dict[str, Any]] = []
    barcode_rows: list[dict[str, Any]] = []
    betti_rows: list[dict[str, Any]] = []
    for path in sorted((out / "runs").glob("*.json")):
        payload = read_json(path)
        result = payload["result"]
        base = {
            "result_id": result.get("result_id"),
            "condition_id": result.get("condition_id"),
            "dataset": result.get("dataset"),
            "pipeline": result.get("pipeline"),
            "stage": result.get("stage"),
            "sample_kind": result.get("sample_kind"),
            "control_kind": result.get("control_kind"),
            "seed": result.get("seed"),
        }
        if result.get("status") != "ok":
            failures.append(
                {
                    "result_id": result.get("result_id"),
                    "condition_id": result.get("condition_id"),
                    "dataset": result.get("dataset"),
                    "pipeline": result.get("pipeline"),
                    "sample_kind": result.get("sample_kind"),
                    "error": result.get("error"),
                }
            )
        for summary in result.get("summaries", []):
            rows.append(summary)
        summaries = {int(item["dim"]): item for item in result.get("summaries", []) if "dim" in item}
        for dim, diagram in enumerate(result.get("diagrams", [])):
            arr = np.asarray(diagram, dtype=float).reshape((-1, 2)) if diagram else np.empty((0, 2), dtype=float)
            threshold = float(summaries.get(dim, {}).get("threshold_or_filtration", 0.0) or 0.0)
            finite = arr[np.isfinite(arr[:, 1])] if len(arr) else np.empty((0, 2), dtype=float)
            for feature_id, (birth, death) in enumerate(arr):
                persistence = float(death - birth) if np.isfinite(death) else float("inf")
                row = {**base, "dim": dim, "feature_id": feature_id, "birth": float(birth), "death": float(death), "persistence": persistence}
                diagram_rows.append(row)
            if len(finite):
                persistence = finite[:, 1] - finite[:, 0]
                order = np.argsort(persistence)[-20:][::-1]
                for rank, idx in enumerate(order, start=1):
                    birth, death = finite[idx]
                    barcode_rows.append({**base, "dim": dim, "rank": rank, "birth": float(birth), "death": float(death), "persistence": float(death - birth)})
            if len(arr) and threshold > 0:
                grid = np.linspace(0, threshold, 64)
                deaths = arr[:, 1].copy()
                deaths[~np.isfinite(deaths)] = threshold
                for scale in grid:
                    alive = int(((arr[:, 0] <= scale) & (deaths > scale)).sum())
                    betti_rows.append({**base, "dim": dim, "scale": float(scale), "betti": alive})
    return pd.DataFrame(rows), pd.DataFrame(failures), pd.DataFrame(diagram_rows), pd.DataFrame(barcode_rows), pd.DataFrame(betti_rows)


def paired_table(runs: pd.DataFrame, stage_filter: str | None = None) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    data = runs[runs["dim"] == 1].copy()
    if stage_filter:
        data = data[data["stage"] == stage_filter]
    obs = data[data["sample_kind"] == "observed"].copy()
    controls = data[~data["sample_kind"].isin(["observed", "positive_control"])].copy()
    if obs.empty or controls.empty:
        return pd.DataFrame()
    control_max = (
        controls.sort_values("max_persistence_norm", ascending=False)
        .drop_duplicates("condition_id")
        [["condition_id", "sample_kind", "max_persistence_norm"]]
        .rename(columns={"sample_kind": "hardest_control_kind", "max_persistence_norm": "hardest_control_norm"})
    )
    paired = obs.merge(control_max, on="condition_id", how="left")
    paired = paired.rename(columns={"max_persistence_norm": "observed_norm"})
    paired["delta"] = paired["observed_norm"] - paired["hardest_control_norm"]
    paired["ratio"] = paired["observed_norm"] / paired["hardest_control_norm"].replace(0, np.nan)
    paired["win"] = paired["delta"] > 0
    return paired


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            runs=("delta", "size"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            mean_ratio=("ratio", "mean"),
            win_rate=("win", "mean"),
            observed_mean=("observed_norm", "mean"),
            control_mean=("hardest_control_norm", "mean"),
        )
        .sort_values(group_cols)
    )


def verdict_from_primary(primary: pd.DataFrame, stability: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    rules = config["decision_rules"]
    verdict: dict[str, Any] = {"verdict": "insufficient_data", "hypothesis_resolution": "No completed primary runs were available.", "pipelines": {}}
    if primary.empty:
        return verdict
    pipeline_results: dict[str, Any] = {}
    for pipeline, df in primary.groupby("pipeline"):
        deltas = df["delta"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(deltas, int(rules["bootstrap_iterations"]), int(rules["bootstrap_seed"]))
        dataset_means = df.groupby("dataset")["delta"].mean()
        stability_df = stability[stability["pipeline"].astype(str).str.startswith(str(pipeline).split("_rips")[0])] if not stability.empty else pd.DataFrame()
        stability_positive_rate = float((stability_df["mean_delta"] > 0).mean()) if not stability_df.empty else 0.0
        metrics = {
            "runs": int(len(df)),
            "mean_delta": float(np.nanmean(deltas)),
            "mean_ratio": float(df["ratio"].mean()),
            "win_rate": float(df["win"].mean()),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "positive_datasets": int((dataset_means > 0).sum()),
            "dataset_count": int(len(dataset_means)),
            "stability_positive_rate": stability_positive_rate,
        }
        passes = {
            "mean_delta": metrics["mean_delta"] >= float(rules["mean_delta_min"]),
            "mean_ratio": metrics["mean_ratio"] >= float(rules["mean_ratio_min"]),
            "ci_lower": metrics["bootstrap_ci_low"] > float(rules["bootstrap_ci_lower_min"]),
            "win_rate": metrics["win_rate"] >= float(rules["win_rate_min"]),
            "datasets": metrics["positive_datasets"] >= int(rules["dataset_positive_min"]),
            "stability": metrics["stability_positive_rate"] >= float(rules["stability_positive_rate_min"]),
        }
        if all(passes.values()):
            status = "confirmed"
        elif metrics["mean_delta"] <= 0 or (metrics["bootstrap_ci_low"] <= 0 and metrics["win_rate"] < float(rules["rule_out_win_rate_max"])):
            status = "ruled_out"
        else:
            status = "pipeline_specific_or_inconclusive"
        pipeline_results[pipeline] = {"status": status, "metrics": metrics, "passes": passes}

    statuses = {value["status"] for value in pipeline_results.values()}
    if "confirmed" in statuses:
        final = "confirmed"
        resolution = "At least one pre-registered primary pipeline survived hardest controls, dataset checks, bootstrap uncertainty, and stability checks."
    elif statuses == {"ruled_out"}:
        final = "ruled_out"
        resolution = "The held-out primary pipelines did not survive controls; the preprocessing-overfit hypothesis is supported."
    else:
        final = "pipeline_specific"
        resolution = "The evidence is not strong enough for a robust FLUX-topology claim; any signal should be described as pipeline-specific or inconclusive."
    return {"verdict": final, "hypothesis_resolution": resolution, "pipelines": pipeline_results}


def stage_aggregate(config: dict[str, Any], args: argparse.Namespace) -> None:
    runs, failures, diagrams, barcodes, betti_curves = load_run_results(args.out)
    if runs.empty:
        runs = pd.DataFrame()
    runs.to_csv(args.out / "runs.csv", index=False)
    failures.to_csv(args.out / "failed_runs.csv", index=False)
    diagrams.to_csv(args.out / "diagrams.csv", index=False)
    barcodes.to_csv(args.out / "barcodes.csv", index=False)
    betti_curves.to_csv(args.out / "betti_curves.csv", index=False)
    primary = paired_table(runs, stage_filter="primary")
    all_paired = paired_table(runs)
    stability_paired = all_paired[all_paired["stage"] == "stability"].copy() if not all_paired.empty else pd.DataFrame()
    stability_summary = summarize_group(stability_paired, ["pipeline", "stage", "dataset"])
    pipeline_summary = summarize_group(all_paired, ["pipeline", "stage"])
    dataset_summary = summarize_group(all_paired, ["pipeline", "dataset", "stage"])
    verdict = verdict_from_primary(primary, stability_summary, config)
    primary.to_csv(args.out / "paired_primary.csv", index=False)
    pipeline_summary.to_csv(args.out / "pipeline_summary.csv", index=False)
    dataset_summary.to_csv(args.out / "dataset_summary.csv", index=False)
    stability_summary.to_csv(args.out / "stability_summary.csv", index=False)
    write_json(args.out / "verdict.json", verdict)
    write_summary_md(args.out, verdict, primary, pipeline_summary, dataset_summary, stability_summary, failures)
    print(f"wrote aggregate artifacts under {args.out}")


def write_summary_md(out: Path, verdict: dict[str, Any], primary: pd.DataFrame, pipeline_summary: pd.DataFrame, dataset_summary: pd.DataFrame, stability_summary: pd.DataFrame, failures: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        shown = df.copy()
        for col in shown.columns:
            if pd.api.types.is_float_dtype(shown[col]):
                shown[col] = shown[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
        shown = shown.astype(str)
        headers = shown.columns.tolist()
        values = shown.values.tolist()
        widths = [max(len(header), *(len(row[i]) for row in values)) for i, header in enumerate(headers)]

        def render(row: list[str]) -> str:
            return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"

        sep = "| " + " | ".join("-" * width for width in widths) + " |"
        return "\n".join([render(headers), sep, *(render(row) for row in values)])

    lines = [
        "# Confirmatory H1 Sweep Summary",
        "",
        f"Verdict: `{verdict.get('verdict', 'unknown')}`",
        "",
        verdict.get("hypothesis_resolution", ""),
        "",
        "## Primary Pipelines",
        "",
    ]
    if primary.empty:
        lines.append("No primary paired runs were available.")
    else:
        cols = ["dataset", "pipeline", "seed", "observed_norm", "hardest_control_norm", "hardest_control_kind", "delta", "ratio", "win"]
        lines.append(md_table(primary[cols].head(30).round(4)))
    lines.extend(["", "## Pipeline Summary", ""])
    lines.append(md_table(pipeline_summary.round(4)) if not pipeline_summary.empty else "No pipeline summary.")
    lines.extend(["", "## Dataset Summary", ""])
    lines.append(md_table(dataset_summary.round(4)) if not dataset_summary.empty else "No dataset summary.")
    lines.extend(["", "## Stability Summary", ""])
    lines.append(md_table(stability_summary.round(4)) if not stability_summary.empty else "No stability summary.")
    lines.extend(["", "## Failures", ""])
    lines.append(md_table(failures.head(50)) if not failures.empty else "No failed runs recorded.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def modal_remote_out(local_out: Path) -> Path:
    out_text = str(local_out)
    if out_text.startswith("/root/"):
        return local_out
    return REMOTE_MODAL_ROOT / local_out.name


def modal_remote_conditions_file(local_conditions_file: Path | None, local_out: Path, remote_out: Path) -> Path | None:
    if local_conditions_file is None:
        return None
    file_text = str(local_conditions_file)
    if file_text.startswith("/root/"):
        return local_conditions_file
    try:
        relative = local_conditions_file.resolve().relative_to(local_out.resolve())
    except ValueError:
        if local_conditions_file.name:
            return remote_out / local_conditions_file.name
        raise SystemExit("Modal conditions files must live under --out or be passed as a /root/... remote path.")
    return remote_out / relative


def stage_modal(args: argparse.Namespace) -> None:
    modal_bin = shutil.which("modal")
    if modal_bin is None:
        raise SystemExit("`--executor modal` requires the Modal CLI. Install and authenticate `modal`, then retry.")
    if args.config.resolve() != CONFIG_PATH.resolve():
        raise SystemExit("Modal execution currently uses the frozen default config packaged in scripts/modal_confirmatory_sweep.py.")

    stages = ["plan", "encode", "run", "aggregate"] if args.stage == "all" else [args.stage]
    remote_out = modal_remote_out(args.out)
    remote_conditions_file = modal_remote_conditions_file(args.conditions_file, args.out, remote_out)
    for stage in stages:
        cmd = [
            modal_bin,
            "run",
            str(ROOT / "scripts" / "modal_confirmatory_sweep.py"),
            "--stage",
            stage,
            "--out",
            str(remote_out),
        ]
        if args.smoke:
            cmd.append("--smoke")
        if args.datasets:
            cmd.extend(["--datasets", args.datasets])
        if args.max_conditions is not None and stage == "run":
            cmd.extend(["--max-conditions", str(args.max_conditions)])
        if remote_conditions_file is not None and stage == "run":
            cmd.extend(["--conditions-file", str(remote_conditions_file)])
        env = {**os.environ, "MODAL_CONFIRMATORY_MAX_CONTAINERS": str(args.modal_max_containers)}
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if args.executor == "modal":
        stage_modal(args)
        return
    args.out = args.out.resolve()
    if args.stage in {"plan", "all"}:
        stage_plan(config, args)
    if args.stage in {"encode", "all"}:
        stage_encode(config, args)
    if args.stage in {"run", "all"}:
        if not (args.out / "conditions.csv").exists():
            stage_plan(config, args)
        stage_run(config, args)
    if args.stage in {"aggregate", "all"}:
        stage_aggregate(config, args)


if __name__ == "__main__":
    main()
