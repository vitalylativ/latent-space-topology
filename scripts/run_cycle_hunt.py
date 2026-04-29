"""Run a compact FLUX latent H1 cycle hunt.

This script is intentionally more experimental than the narrative notebooks. It
tries several preprocessing and landmark-selection choices, runs full H1 Rips
filtrations where feasible, and writes a CSV/Markdown summary under
outputs/cycle_hunt/local/.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebook_utils.encoder_explorer import (
    DEFAULT_IMAGE_DIR,
    choose_device,
    extract_token_clouds,
    l2_normalize,
    load_project_images,
    sample_indices,
    seed_everything,
)
from notebook_utils.flux_tda import farthest_point_indices, kth_neighbor_distance, select_dense_indices


OUT_DIR = ROOT / "outputs" / "cycle_hunt" / "local"


def covariance_whiten(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    centered = x.astype(np.float32) - x.astype(np.float32).mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov + np.eye(cov.shape[0]) * eps)
    values = np.maximum(values, eps)
    return (centered @ vectors @ np.diag(1.0 / np.sqrt(values))).astype(np.float32)


def pca_view(x: np.ndarray, dim: int, seed: int, normalize: bool = False) -> np.ndarray:
    z = PCA(n_components=min(dim, x.shape[1], x.shape[0] - 1), random_state=seed).fit_transform(x.astype(np.float32))
    z = z.astype(np.float32)
    return l2_normalize(z).astype(np.float32) if normalize else z


def build_view(tokens: np.ndarray, view: str, seed: int) -> np.ndarray:
    tokens = tokens.astype(np.float32)
    if view == "sphere":
        return l2_normalize(tokens).astype(np.float32)
    if view == "raw":
        return tokens
    if view == "whitened":
        return covariance_whiten(tokens)
    if view == "pca3":
        return pca_view(tokens, 3, seed, normalize=False)
    if view == "pca8":
        return pca_view(tokens, 8, seed, normalize=False)
    if view == "pca8_sphere":
        return pca_view(tokens, 8, seed, normalize=True)
    if view == "norm_only":
        norms = np.linalg.norm(tokens, axis=1, keepdims=True)
        return ((norms - norms.mean()) / max(float(norms.std()), 1e-8)).astype(np.float32)
    raise ValueError(f"unknown view: {view}")


def choose_landmarks(x: np.ndarray, strategy: str, n_dense: int, n_landmarks: int, k_density: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n_landmarks = min(n_landmarks, len(x))
    if strategy == "dense_fps":
        dense_idx, dense_kth = select_dense_indices(x, n_dense=n_dense, k=k_density, max_candidates=min(20000, len(x)), seed=seed)
        local = farthest_point_indices(x[dense_idx], n_landmarks=n_landmarks, seed=seed)
        return dense_idx[local], {"dense_kth_median": float(np.median(dense_kth)), "dense_n": len(dense_idx)}
    if strategy == "all_fps":
        candidate_idx = sample_indices(len(x), min(20000, len(x)), seed=seed)
        local = farthest_point_indices(x[candidate_idx], n_landmarks=n_landmarks, seed=seed)
        return candidate_idx[local], {"candidate_n": len(candidate_idx)}
    if strategy == "random":
        idx = sample_indices(len(x), n_landmarks, seed=seed)
        return idx, {}
    if strategy == "dense_random":
        dense_idx, dense_kth = select_dense_indices(x, n_dense=n_dense, k=k_density, max_candidates=min(20000, len(x)), seed=seed)
        idx = np.sort(rng.choice(dense_idx, size=min(n_landmarks, len(dense_idx)), replace=False))
        return idx, {"dense_kth_median": float(np.median(dense_kth)), "dense_n": len(dense_idx)}
    raise ValueError(f"unknown strategy: {strategy}")


def h1_summary(points: np.ndarray, threshold: str = "full") -> dict[str, float]:
    import ripser

    pairwise = pdist(points, metric="euclidean")
    if len(pairwise) == 0:
        return {
            "h1_count": 0,
            "h1_max": 0.0,
            "h1_top3": 0.0,
            "h1_norm_max_pairwise": 0.0,
            "pairwise_max": 0.0,
            "pairwise_median": 0.0,
        }
    thresh = float(np.max(pairwise)) if threshold == "full" else float(np.quantile(pairwise, float(threshold)))
    result = ripser.ripser(points, maxdim=1, thresh=thresh)
    h1 = result["dgms"][1]
    finite = h1[np.isfinite(h1[:, 1])]
    persistence = finite[:, 1] - finite[:, 0] if len(finite) else np.array([])
    h1_max = float(persistence.max()) if len(persistence) else 0.0
    return {
        "h1_count": int(len(finite)),
        "h1_max": h1_max,
        "h1_top3": float(np.sort(persistence)[-3:].sum()) if len(persistence) else 0.0,
        "h1_norm_max_pairwise": h1_max / float(np.max(pairwise)),
        "h1_norm_median_pairwise": h1_max / float(np.median(pairwise)),
        "pairwise_max": float(np.max(pairwise)),
        "pairwise_median": float(np.median(pairwise)),
        "threshold": thresh,
    }


def uniform_control(dim: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    return l2_normalize(x).astype(np.float32) if dim > 1 else x


def shuffled_control(tokens: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = tokens.copy()
    for col in range(x.shape[1]):
        rng.shuffle(x[:, col])
    return x


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("CYCLE_HUNT_SEED", "72"))
    n_images = int(os.environ.get("CYCLE_HUNT_N_IMAGES", "48"))
    batch_size = int(os.environ.get("TOKENIZER_BATCH_SIZE", "4"))
    image_size = int(os.environ.get("TOKENIZER_AUTOENCODER_SIZE", "256"))
    seed_everything(seed)

    print(f"loading {n_images} images from {DEFAULT_IMAGE_DIR}")
    images, metadata = load_project_images(n_images, DEFAULT_IMAGE_DIR)
    print(f"loaded {len(images)} images")

    device = choose_device(force_cpu=os.environ.get("TOKENIZER_FORCE_CPU", "0") == "1")
    t0 = time.perf_counter()
    clouds, failures = extract_token_clouds(
        images,
        metadata,
        device=device,
        batch_size=batch_size,
        autoencoder_size=image_size,
        selected=["flux_vae"],
    )
    if not failures.empty:
        print(failures)
    flux = clouds["flux_vae"]
    print(f"encoded tokens={flux.tokens.shape} in {time.perf_counter() - t0:.2f}s on {device}")

    views = ["sphere", "raw", "whitened", "pca3", "pca8", "pca8_sphere", "norm_only"]
    strategies = ["dense_fps", "all_fps", "random", "dense_random"]
    landmark_sizes = [70, 95, 140, 200]
    dense_sizes = [400, 800, 1600]
    k_values = [8, 16, 32]
    seeds = [seed, seed + 1, seed + 2]

    rows: list[dict[str, Any]] = []
    total = len(views) * len(strategies) * len(landmark_sizes) * len(seeds)
    run_i = 0
    view_cache = {view: build_view(flux.tokens, view, seed=seed) for view in views}
    for view, x in view_cache.items():
        for strategy in strategies:
            for n_landmarks in landmark_sizes:
                for s in seeds:
                    n_dense = dense_sizes[min(dense_sizes.index(800) if 800 in dense_sizes else 0, len(dense_sizes) - 1)]
                    k_density = 16
                    if strategy.startswith("dense"):
                        n_dense = dense_sizes[(s + n_landmarks) % len(dense_sizes)]
                        k_density = k_values[(s + n_landmarks) % len(k_values)]
                    run_i += 1
                    try:
                        idx, notes = choose_landmarks(x, strategy, n_dense, n_landmarks, k_density, s)
                        points = x[idx]
                        observed = h1_summary(points, threshold="full")
                        random_idx = sample_indices(len(x), len(points), seed=s + 1001)
                        random_same_view = h1_summary(x[random_idx], threshold="full")
                        uniform = h1_summary(uniform_control(points.shape[1], len(points), seed=s + 1002), threshold="full")
                        shuffled_x = shuffled_control(x, seed=s + 1003)
                        shuffle_idx, _ = choose_landmarks(shuffled_x, strategy, n_dense, len(points), k_density, s)
                        shuffled = h1_summary(shuffled_x[shuffle_idx], threshold="full")
                        row = {
                            "view": view,
                            "strategy": strategy,
                            "seed": s,
                            "n_landmarks": len(points),
                            "n_dense": n_dense,
                            "k_density": k_density,
                            "dim": points.shape[1],
                            **notes,
                            **{f"observed_{k}": v for k, v in observed.items()},
                            **{f"random_{k}": v for k, v in random_same_view.items()},
                            **{f"uniform_{k}": v for k, v in uniform.items()},
                            **{f"shuffled_{k}": v for k, v in shuffled.items()},
                        }
                        row["delta_vs_best_control"] = row["observed_h1_norm_max_pairwise"] - max(
                            row["random_h1_norm_max_pairwise"],
                            row["uniform_h1_norm_max_pairwise"],
                            row["shuffled_h1_norm_max_pairwise"],
                        )
                        row["delta_raw_vs_best_control"] = row["observed_h1_max"] - max(
                            row["random_h1_max"],
                            row["uniform_h1_max"],
                            row["shuffled_h1_max"],
                        )
                        rows.append(row)
                    except Exception as exc:  # keep the sweep moving
                        rows.append(
                            {
                                "view": view,
                                "strategy": strategy,
                                "seed": s,
                                "n_landmarks": n_landmarks,
                                "n_dense": n_dense,
                                "k_density": k_density,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    if run_i % 25 == 0:
                        print(f"{run_i}/{total} sweep configs")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "cycle_hunt_local_sweep.csv"
    df.to_csv(csv_path, index=False)

    ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()].copy()
    top_observed = ok.sort_values("observed_h1_norm_max_pairwise", ascending=False).head(20)
    top_delta = ok.sort_values("delta_vs_best_control", ascending=False).head(20)
    md_path = OUT_DIR / "cycle_hunt_local_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Local FLUX Cycle Hunt Summary\n\n")
        handle.write(f"- images: {len(images)}\n")
        handle.write(f"- tokens: {tuple(flux.tokens.shape)}\n")
        handle.write(f"- completed configs: {len(ok)} / {len(df)}\n\n")
        handle.write("## Top Observed Normalized H1\n\n")
        handle.write(
            top_observed[
                [
                    "view",
                    "strategy",
                    "seed",
                    "n_landmarks",
                    "n_dense",
                    "k_density",
                    "observed_h1_norm_max_pairwise",
                    "random_h1_norm_max_pairwise",
                    "uniform_h1_norm_max_pairwise",
                    "shuffled_h1_norm_max_pairwise",
                    "delta_vs_best_control",
                ]
            ].to_string(index=False)
        )
        handle.write("\n\n## Top Observed Minus Best Control\n\n")
        handle.write(
            top_delta[
                [
                    "view",
                    "strategy",
                    "seed",
                    "n_landmarks",
                    "n_dense",
                    "k_density",
                    "observed_h1_norm_max_pairwise",
                    "random_h1_norm_max_pairwise",
                    "uniform_h1_norm_max_pairwise",
                    "shuffled_h1_norm_max_pairwise",
                    "delta_vs_best_control",
                ]
            ].to_string(index=False)
        )
        handle.write("\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(top_delta.head(10).to_string(index=False))


if __name__ == "__main__":
    run()
