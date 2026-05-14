from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from notebook_utils.encoder_explorer import (
    TokenCloud,
    approximate_patch,
    l2_normalize,
    load_project_images,
    make_cloud_views,
    participation_ratio,
    run_raw_patches,
    sample_indices,
    spatial_neighbor_cosine,
    twonn_intrinsic_dimension,
)
from notebook_utils.mapper_explorer import (
    MapperGraph,
    build_mapper_graph,
    discover_cached_flux_clouds,
    load_token_cloud_cache,
    make_lens_catalog,
    mapper_stats,
    node_metadata_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostic_workbench"


@dataclass
class MapperRun:
    run_id: str
    seed: int
    graph: MapperGraph
    x: np.ndarray
    metadata: pd.DataFrame
    source_indices: np.ndarray
    lens_name: str
    config: dict[str, Any]


def _dataset_from_cache_path(path: str | Path) -> str:
    stem = Path(path).stem
    if stem.startswith("flux_vae_"):
        return stem.removeprefix("flux_vae_").split("_n", 1)[0]
    return stem


def load_cached_flux_clouds(max_clouds: int | None = None) -> dict[str, TokenCloud]:
    """Load all local FLUX token caches that follow this project's cache format."""
    rows = discover_cached_flux_clouds()
    clouds: dict[str, TokenCloud] = {}
    if rows.empty:
        return clouds
    available_datasets = set(rows["dataset"].astype(str).tolist()) if "dataset" in rows.columns else set()
    for _, row in rows.sort_values("dataset").iterrows():
        dataset = str(row.get("dataset") or _dataset_from_cache_path(row["cloud_path"]))
        if dataset == "beans_param_sweep" and "beans_local" in available_datasets:
            continue
        if dataset in clouds:
            continue
        cloud = load_token_cloud_cache(
            row["cloud_path"],
            row.get("metadata_path") if pd.notna(row.get("metadata_path")) else None,
            name=f"{dataset}:flux_vae",
        )
        cloud.notes = dict(cloud.notes)
        cloud.notes["dataset"] = dataset
        clouds[dataset] = cloud
        if max_clouds is not None and len(clouds) >= max_clouds:
            break
    return clouds


def maybe_load_raw_patch_cloud(
    n_images: int = 48,
    image_size: int = 256,
    image_dir: str | Path | None = None,
) -> TokenCloud | None:
    """Build the local raw-patch baseline when the image folder or HF fallback is available."""
    try:
        images, metadata = load_project_images(n_images=n_images, image_dir=image_dir)
        cloud = run_raw_patches(images, metadata, image_size=image_size, patch_size=16)
        cloud.notes = dict(cloud.notes)
        cloud.notes["dataset"] = "beans_local"
        return cloud
    except Exception:
        return None


def _safe_label_silhouette(x: np.ndarray, metadata: pd.DataFrame, label_col: str) -> float:
    if label_col not in metadata.columns:
        return float("nan")
    labels = metadata[label_col].astype(str).to_numpy()
    valid = np.asarray([label not in {"None", "nan", ""} for label in labels])
    if valid.sum() < 10 or len(np.unique(labels[valid])) < 2:
        return float("nan")
    if min(pd.Series(labels[valid]).value_counts()) < 2:
        return float("nan")
    try:
        return float(silhouette_score(x[valid], labels[valid], metric="euclidean"))
    except Exception:
        return float("nan")


def _same_image_nn_rate(x: np.ndarray, metadata: pd.DataFrame, k: int = 6) -> float:
    if "image_id" not in metadata.columns or len(x) <= 2:
        return float("nan")
    n_neighbors = min(k + 1, len(x))
    indices = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors(x, return_distance=False)[:, 1:]
    images = metadata["image_id"].astype(str).to_numpy()
    rates = []
    for i, neigh in enumerate(indices):
        rates.append(float(np.mean(images[neigh] == images[i])))
    return float(np.mean(rates))


def _pca_summary(x: np.ndarray, seed: int) -> dict[str, float | int]:
    n_components = min(32, x.shape[0] - 1, x.shape[1])
    if n_components < 1:
        return {
            "pc1": float("nan"),
            "pc2": float("nan"),
            "pca_80_components": 0,
            "participation_ratio": float("nan"),
        }
    pca = PCA(n_components=n_components, random_state=seed).fit(x)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    return {
        "pc1": float(ratios[0]),
        "pc2": float(ratios[1]) if len(ratios) > 1 else 0.0,
        "pca_80_components": int(np.searchsorted(cumulative, 0.80) + 1),
        "participation_ratio": participation_ratio(pca),
    }


def geometry_feature_row(
    cloud: TokenCloud,
    dataset: str,
    representation: str,
    view_key: str,
    max_points: int = 1600,
    seed: int = 72,
    label_col: str = "label",
) -> dict[str, Any]:
    views = make_cloud_views({representation: cloud}, max_whiten_dim=min(64, cloud.channel_dim), seed=seed)
    view = views[f"{representation}:{view_key}"]
    idx = sample_indices(len(view.tokens), min(max_points, len(view.tokens)), seed=seed)
    x = view.tokens[idx].astype(np.float32)
    metadata = view.token_metadata.iloc[idx].reset_index(drop=True)
    norms = np.linalg.norm(x, axis=1)
    kth = NearestNeighbors(n_neighbors=min(17, len(x))).fit(x).kneighbors(x)[0][:, -1] if len(x) > 2 else np.zeros(len(x))
    nearest = (
        NearestNeighbors(n_neighbors=min(2, len(x))).fit(x).kneighbors(x)[0][:, -1]
        if len(x) > 1
        else np.full(len(x), np.nan)
    )
    duplicate_eps = 1e-8 * max(1.0, float(np.nanmedian(norms)))
    n_images = int(cloud.token_metadata["image_id"].nunique()) if "image_id" in cloud.token_metadata.columns else np.nan

    row: dict[str, Any] = {
        "dataset": dataset,
        "representation": representation,
        "view": view_key,
        "feature_family": "geometry",
        "seed": seed,
        "n_images": n_images,
        "n_tokens": int(len(x)),
        "ambient_dim": int(x.shape[1]),
        "grid_h": int(cloud.grid_shape[0]),
        "grid_w": int(cloud.grid_shape[1]),
        "norm_mean": float(norms.mean()) if len(norms) else float("nan"),
        "norm_cv": float(norms.std() / max(norms.mean(), 1e-12)) if len(norms) else float("nan"),
        "density_q90_q10": float(np.quantile(kth, 0.90) / max(np.quantile(kth, 0.10), 1e-12)) if len(kth) else float("nan"),
        "near_duplicate_fraction": float(np.mean(nearest <= duplicate_eps)) if len(nearest) else float("nan"),
        "same_image_nn_rate": _same_image_nn_rate(x, metadata),
        "twonn_id": twonn_intrinsic_dimension(x),
        "label_silhouette": _safe_label_silhouette(x, metadata, label_col=label_col),
    }
    row.update(_pca_summary(x, seed))
    row.update(spatial_neighbor_cosine(view, n_images if not pd.isna(n_images) else 0))
    return row


def topology_feature_row(
    cloud: TokenCloud,
    dataset: str,
    representation: str,
    view_key: str = "unit",
    max_points: int = 220,
    seed: int = 72,
    distance_quantile: float = 0.85,
) -> dict[str, Any]:
    try:
        from ripser import ripser
    except Exception as exc:
        return {
            "dataset": dataset,
            "representation": representation,
            "view": view_key,
            "feature_family": "tda",
            "seed": seed,
            "error": f"ripser unavailable: {type(exc).__name__}: {exc}",
        }

    views = make_cloud_views({representation: cloud}, max_whiten_dim=min(64, cloud.channel_dim), seed=seed)
    x = views[f"{representation}:{view_key}"].tokens
    idx = sample_indices(len(x), min(max_points, len(x)), seed=seed)
    xs = x[idx].astype(np.float32)
    if len(xs) < 5:
        return {
            "dataset": dataset,
            "representation": representation,
            "view": view_key,
            "feature_family": "tda",
            "seed": seed,
            "n_points": int(len(xs)),
        }
    distances = NearestNeighbors(n_neighbors=min(16, len(xs))).fit(xs).kneighbors(xs)[0][:, -1]
    threshold = float(np.quantile(distances, distance_quantile))
    threshold = max(threshold, 1e-8)
    try:
        result = ripser(xs, maxdim=1, thresh=threshold)
        diagrams = result["dgms"]
        h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))
        finite = h1[np.isfinite(h1[:, 1])] if len(h1) else h1
        lifetimes = finite[:, 1] - finite[:, 0] if len(finite) else np.asarray([], dtype=float)
        top = np.sort(lifetimes)[::-1][:5]
        row: dict[str, Any] = {
            "dataset": dataset,
            "representation": representation,
            "view": view_key,
            "feature_family": "tda",
            "seed": seed,
            "n_points": int(len(xs)),
            "ambient_dim": int(xs.shape[1]),
            "threshold": threshold,
            "h1_features": int(len(h1)),
            "h1_finite": int(len(finite)),
            "h1_max_persistence": float(lifetimes.max()) if len(lifetimes) else 0.0,
            "h1_total_persistence": float(lifetimes.sum()) if len(lifetimes) else 0.0,
            "h1_max_persistence_norm": float(lifetimes.max() / threshold) if len(lifetimes) else 0.0,
        }
        for i in range(5):
            row[f"h1_top{i + 1}_persistence"] = float(top[i]) if i < len(top) else 0.0
        return row
    except Exception as exc:
        return {
            "dataset": dataset,
            "representation": representation,
            "view": view_key,
            "feature_family": "tda",
            "seed": seed,
            "n_points": int(len(xs)),
            "ambient_dim": int(xs.shape[1]),
            "threshold": threshold,
            "error": f"{type(exc).__name__}: {exc}",
        }


def mapper_feature_row(
    cloud: TokenCloud,
    dataset: str,
    representation: str,
    lens_name: str = "norm_density",
    config: dict[str, Any] | None = None,
    max_points: int = 1400,
    seed: int = 72,
    label_col: str = "label",
) -> dict[str, Any]:
    config = config or {
        "config": "baseline",
        "n_intervals": 8,
        "overlap": 0.35,
        "min_bin_points": 12,
        "min_cluster_size": 5,
        "eps_quantile": 0.35,
    }
    idx = sample_indices(len(cloud.tokens), min(max_points, len(cloud.tokens)), seed=seed)
    x = cloud.tokens[idx].astype(np.float32)
    metadata = cloud.token_metadata.iloc[idx].reset_index(drop=True)
    lens_catalog = make_lens_catalog(x, seed=seed)
    if lens_name not in lens_catalog:
        raise KeyError(f"lens {lens_name!r} is not available")
    graph = build_mapper_graph(
        x,
        lens_catalog[lens_name],
        name=f"{dataset}:{representation}:{lens_name}:{config.get('config', 'config')}",
        source_indices=idx,
        n_intervals=int(config.get("n_intervals", 8)),
        overlap=float(config.get("overlap", 0.35)),
        min_bin_points=int(config.get("min_bin_points", 12)),
        clusterer=str(config.get("clusterer", "dbscan")),
        min_cluster_size=int(config.get("min_cluster_size", 5)),
        eps_quantile=float(config.get("eps_quantile", 0.35)),
    )
    row = {
        "dataset": dataset,
        "representation": representation,
        "view": "mapper_input_raw",
        "feature_family": "mapper",
        "pipeline_or_lens": lens_name,
        "config": str(config.get("config", "config")),
        "seed": seed,
    }
    row.update(mapper_stats(graph, metadata, label_col=label_col))
    return row


def compute_feature_table(
    clouds: dict[str, TokenCloud],
    geometry_views: tuple[str, ...] = ("raw", "unit", "whitened"),
    include_tda: bool = True,
    include_mapper: bool = True,
    max_geometry_points: int = 1600,
    max_tda_points: int = 220,
    max_mapper_points: int = 1400,
    seed: int = 72,
) -> pd.DataFrame:
    """Build one mixed feature table for cached representations."""
    rows: list[dict[str, Any]] = []
    mapper_config = {
        "config": "baseline",
        "n_intervals": 8,
        "overlap": 0.35,
        "min_bin_points": 12,
        "min_cluster_size": 5,
        "eps_quantile": 0.35,
    }
    for dataset_key, cloud in clouds.items():
        dataset = str(cloud.notes.get("dataset", dataset_key)) if hasattr(cloud, "notes") else str(dataset_key)
        representation = cloud.name.split(":", 1)[-1] if ":" in cloud.name else cloud.name
        for view in geometry_views:
            rows.append(
                geometry_feature_row(
                    cloud,
                    dataset=dataset,
                    representation=representation,
                    view_key=view,
                    max_points=max_geometry_points,
                    seed=seed,
                )
            )
        if include_tda:
            rows.append(
                topology_feature_row(
                    cloud,
                    dataset=dataset,
                    representation=representation,
                    view_key="unit",
                    max_points=max_tda_points,
                    seed=seed,
                )
            )
        if include_mapper:
            try:
                rows.append(
                    mapper_feature_row(
                        cloud,
                        dataset=dataset,
                        representation=representation,
                        lens_name="norm_density",
                        config=mapper_config,
                        max_points=max_mapper_points,
                        seed=seed,
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "representation": representation,
                        "view": "mapper_input_raw",
                        "feature_family": "mapper",
                        "pipeline_or_lens": "norm_density",
                        "config": "baseline",
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(rows).reset_index(drop=True)


def save_feature_table(table: pd.DataFrame, output_dir: str | Path | None = None, stem: str = "features") -> dict[str, Path]:
    output = Path(output_dir) if output_dir is not None else FEATURE_OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    paths = {"csv": output / f"{stem}.csv"}
    table.to_csv(paths["csv"], index=False)
    try:
        paths["parquet"] = output / f"{stem}.parquet"
        table.to_parquet(paths["parquet"], index=False)
    except Exception:
        pass
    return paths


def plot_feature_heatmap(table: pd.DataFrame, metrics: list[str] | None = None) -> None:
    if table.empty:
        print("No feature rows to plot.")
        return
    metrics = metrics or [
        "norm_cv",
        "pc1",
        "pca_80_components",
        "participation_ratio",
        "twonn_id",
        "density_q90_q10",
        "spatial_cosine_mean",
        "same_image_nn_rate",
        "label_silhouette",
        "h1_max_persistence_norm",
        "nodes",
        "graph_h1_rank",
        "coverage_fraction",
        "weighted_label_purity",
    ]
    metrics = [metric for metric in metrics if metric in table.columns]
    if not metrics:
        print("No requested metric columns are present.")
        return
    display_table = table.copy()
    display_table["row"] = (
        display_table["dataset"].astype(str)
        + " | "
        + display_table["representation"].astype(str)
        + " | "
        + display_table["feature_family"].astype(str)
        + " | "
        + display_table["view"].astype(str)
    )
    values = display_table.set_index("row")[metrics].apply(pd.to_numeric, errors="coerce")
    z = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)
    height = max(4.0, min(14.0, 0.42 * len(z) + 1.5))
    plt.figure(figsize=(12, height))
    sns.heatmap(z, cmap="vlag", center=0, linewidths=0.3, linecolor="white")
    plt.title("Standardized diagnostic feature table")
    plt.tight_layout()
    plt.show()


def build_mapper_stability_runs(
    cloud: TokenCloud,
    seeds: list[int] | tuple[int, ...] = (72, 73, 74),
    lens_name: str = "norm_density",
    config: dict[str, Any] | None = None,
    pool_points: int = 2200,
    sample_fraction: float = 0.75,
    pool_seed: int = 72,
) -> dict[str, MapperRun]:
    """Build Mapper graphs on overlapping bootstrap samples from one fixed token pool."""
    config = config or {
        "config": "baseline",
        "n_intervals": 8,
        "overlap": 0.35,
        "min_bin_points": 12,
        "min_cluster_size": 5,
        "eps_quantile": 0.35,
    }
    pool = sample_indices(len(cloud.tokens), min(pool_points, len(cloud.tokens)), seed=pool_seed)
    run_size = max(1, int(round(len(pool) * sample_fraction)))
    runs: dict[str, MapperRun] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        source_indices = np.sort(rng.choice(pool, size=run_size, replace=False))
        x = cloud.tokens[source_indices].astype(np.float32)
        metadata = cloud.token_metadata.iloc[source_indices].reset_index(drop=True)
        lenses = make_lens_catalog(x, seed=seed)
        graph = build_mapper_graph(
            x,
            lenses[lens_name],
            name=f"{lens_name}:{config.get('config', 'config')}:seed{seed}",
            source_indices=source_indices,
            n_intervals=int(config.get("n_intervals", 8)),
            overlap=float(config.get("overlap", 0.35)),
            min_bin_points=int(config.get("min_bin_points", 12)),
            clusterer=str(config.get("clusterer", "dbscan")),
            min_cluster_size=int(config.get("min_cluster_size", 5)),
            eps_quantile=float(config.get("eps_quantile", 0.35)),
        )
        run_id = f"seed{seed}"
        runs[run_id] = MapperRun(run_id, int(seed), graph, x, metadata, source_indices, lens_name, config)
    return runs


def node_source_sets(graph: MapperGraph) -> dict[int, set[int]]:
    if graph.memberships.empty:
        return {}
    return {
        int(node_id): set(group["source_index"].astype(int).tolist())
        for node_id, group in graph.memberships.groupby("node_id")
    }


def node_match_table(
    graph_a: MapperGraph,
    graph_b: MapperGraph,
    run_a: str = "a",
    run_b: str = "b",
    min_jaccard: float = 0.0,
) -> pd.DataFrame:
    sets_a = node_source_sets(graph_a)
    sets_b = node_source_sets(graph_b)
    rows = []
    for node_a, values_a in sets_a.items():
        for node_b, values_b in sets_b.items():
            intersection = len(values_a & values_b)
            if intersection == 0:
                continue
            union = len(values_a | values_b)
            jaccard = intersection / max(union, 1)
            if jaccard < min_jaccard:
                continue
            rows.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "node_a": int(node_a),
                    "node_b": int(node_b),
                    "size_a": int(len(values_a)),
                    "size_b": int(len(values_b)),
                    "intersection": int(intersection),
                    "jaccard": float(jaccard),
                    "containment_a": float(intersection / max(len(values_a), 1)),
                    "containment_b": float(intersection / max(len(values_b), 1)),
                }
            )
    columns = [
        "run_a",
        "run_b",
        "node_a",
        "node_b",
        "size_a",
        "size_b",
        "intersection",
        "jaccard",
        "containment_a",
        "containment_b",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["jaccard", "intersection"], ascending=False).reset_index(drop=True)


def greedy_node_matches(matches: pd.DataFrame, min_jaccard: float = 0.05) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()
    chosen = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for _, row in matches.sort_values(["jaccard", "intersection"], ascending=False).iterrows():
        node_a = int(row["node_a"])
        node_b = int(row["node_b"])
        if row["jaccard"] < min_jaccard or node_a in used_a or node_b in used_b:
            continue
        chosen.append(row.to_dict())
        used_a.add(node_a)
        used_b.add(node_b)
    if not chosen:
        return matches.head(0).copy()
    return pd.DataFrame(chosen)


def hungarian_node_matches(matches: pd.DataFrame, min_jaccard: float = 0.05) -> pd.DataFrame:
    """One-to-one node matching that maximizes exact source-overlap Jaccard."""
    if matches.empty:
        return matches.copy()
    nodes_a = sorted(matches["node_a"].astype(int).unique())
    nodes_b = sorted(matches["node_b"].astype(int).unique())
    a_pos = {node: i for i, node in enumerate(nodes_a)}
    b_pos = {node: i for i, node in enumerate(nodes_b)}
    score = np.zeros((len(nodes_a), len(nodes_b)), dtype=float)
    row_by_pair = {}
    for _, row in matches.iterrows():
        a = int(row["node_a"])
        b = int(row["node_b"])
        score[a_pos[a], b_pos[b]] = max(score[a_pos[a], b_pos[b]], float(row["jaccard"]))
        row_by_pair[(a, b)] = row.to_dict()
    rows_a, rows_b = linear_sum_assignment(-score)
    chosen = []
    for i, j in zip(rows_a, rows_b):
        if score[i, j] >= min_jaccard:
            chosen.append(row_by_pair[(nodes_a[i], nodes_b[j])])
    if not chosen:
        return matches.head(0).copy()
    return pd.DataFrame(chosen).sort_values(["jaccard", "intersection"], ascending=False).reset_index(drop=True)


def reference_node_stability(
    runs: dict[str, MapperRun],
    reference_run: str | None = None,
    min_jaccard: float = 0.05,
    matching: str = "hungarian",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not runs:
        return pd.DataFrame(), pd.DataFrame()
    reference_run = reference_run or sorted(runs)[0]
    reference = runs[reference_run]
    match_rows = []
    matcher = hungarian_node_matches if matching == "hungarian" else greedy_node_matches
    for run_id, run in runs.items():
        if run_id == reference_run:
            continue
        all_matches = node_match_table(reference.graph, run.graph, run_a=reference_run, run_b=run_id)
        chosen = matcher(all_matches, min_jaccard=min_jaccard)
        if not chosen.empty:
            match_rows.append(chosen)
    matches = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()

    base_nodes = reference.graph.nodes[["node_id", "size"]].copy()
    base_nodes = base_nodes.rename(columns={"node_id": "reference_node_id", "size": "reference_size"})
    rows = []
    other_runs = [run_id for run_id in runs if run_id != reference_run]
    for _, node in base_nodes.iterrows():
        node_id = int(node["reference_node_id"])
        node_matches = matches[matches["node_a"] == node_id] if not matches.empty else pd.DataFrame()
        best_by_run = node_matches.groupby("run_b")["jaccard"].max() if not node_matches.empty else pd.Series(dtype=float)
        row = {
            "reference_run": reference_run,
            "reference_node_id": node_id,
            "reference_size": int(node["reference_size"]),
            "eligible_runs": int(len(other_runs)),
            "matched_runs": int((best_by_run >= min_jaccard).sum()),
            "recurrence": float((best_by_run >= min_jaccard).sum() / max(len(other_runs), 1)),
            "mean_best_jaccard": float(best_by_run.mean()) if len(best_by_run) else 0.0,
            "median_best_jaccard": float(best_by_run.median()) if len(best_by_run) else 0.0,
            "max_best_jaccard": float(best_by_run.max()) if len(best_by_run) else 0.0,
        }
        rows.append(row)
    stability = pd.DataFrame(rows).sort_values(
        ["recurrence", "median_best_jaccard", "reference_size"], ascending=False
    )
    return stability.reset_index(drop=True), matches.reset_index(drop=True)


def annotate_stability_with_node_summary(
    stability: pd.DataFrame,
    run: MapperRun,
    label_col: str = "label",
) -> pd.DataFrame:
    if stability.empty:
        return stability.copy()
    summary = node_metadata_summary(run.graph, run.x, run.metadata, label_col=label_col)
    if summary.empty:
        return stability.copy()
    summary = summary.rename(columns={"node_id": "reference_node_id"})
    out = stability.merge(summary, on="reference_node_id", how="left")
    if f"{label_col}_purity" in out.columns:
        baseline = _label_baseline_purity(run.metadata, label_col=label_col)
        out[f"{label_col}_purity_excess"] = out[f"{label_col}_purity"] - baseline
    return out


def _label_baseline_purity(metadata: pd.DataFrame, label_col: str = "label") -> float:
    if label_col not in metadata.columns:
        return float("nan")
    counts = metadata[label_col].astype(str).value_counts(normalize=True)
    return float(counts.iloc[0]) if len(counts) else float("nan")


def summarize_stability_runs(runs: dict[str, MapperRun]) -> pd.DataFrame:
    rows = []
    for run_id, run in runs.items():
        row = {"run_id": run_id, "seed": run.seed, "lens": run.lens_name, "config": run.config.get("config", "config")}
        row.update(mapper_stats(run.graph, run.metadata))
        rows.append(row)
    return pd.DataFrame(rows)


def select_stable_atlas_nodes(
    stability_summary: pd.DataFrame,
    max_nodes: int = 6,
    min_recurrence: float = 0.5,
    min_size: int = 16,
) -> list[int]:
    if stability_summary.empty:
        return []
    table = stability_summary.copy()
    table = table[table["reference_size"] >= min_size]
    if "recurrence" in table.columns:
        table = table[table["recurrence"] >= min_recurrence]
    sort_cols = [col for col in ["recurrence", "median_best_jaccard", "label_purity_excess", "reference_size"] if col in table.columns]
    if table.empty:
        table = stability_summary.copy()
        sort_cols = [col for col in ["recurrence", "median_best_jaccard", "reference_size"] if col in table.columns]
    return table.sort_values(sort_cols, ascending=False)["reference_node_id"].head(max_nodes).astype(int).tolist()


def _node_representative_sources(
    graph: MapperGraph,
    cloud: TokenCloud,
    node_id: int,
    patches_per_node: int,
    diversify_images: bool = True,
) -> list[int]:
    rows = graph.memberships[graph.memberships["node_id"] == node_id]
    if rows.empty:
        return []
    sources = rows["source_index"].astype(int).drop_duplicates().to_numpy()
    x = cloud.tokens[sources].astype(np.float32)
    centroid = x.mean(axis=0, keepdims=True)
    order = np.argsort(np.linalg.norm(x - centroid, axis=1))
    ranked = [int(sources[i]) for i in order]
    if not diversify_images or "image_id" not in cloud.token_metadata.columns:
        return ranked[:patches_per_node]
    selected = []
    used_images: set[str] = set()
    for source in ranked:
        image_id = str(cloud.token_metadata.iloc[source].get("image_id", "unknown"))
        if image_id in used_images:
            continue
        selected.append(source)
        used_images.add(image_id)
        if len(selected) >= patches_per_node:
            return selected
    for source in ranked:
        if source not in selected:
            selected.append(source)
        if len(selected) >= patches_per_node:
            break
    return selected


def patch_atlas_records(
    graph: MapperGraph,
    cloud: TokenCloud,
    node_ids: list[int],
    patches_per_node: int = 6,
) -> pd.DataFrame:
    rows = []
    for node_id in node_ids:
        for rank, source_index in enumerate(_node_representative_sources(graph, cloud, node_id, patches_per_node), start=1):
            meta = cloud.token_metadata.iloc[source_index].to_dict()
            rows.append(
                {
                    "node_id": int(node_id),
                    "patch_rank": int(rank),
                    "source_index": int(source_index),
                    "image_id": meta.get("image_id"),
                    "label": meta.get("label"),
                    "h": meta.get("h"),
                    "w": meta.get("w"),
                }
            )
    return pd.DataFrame(rows)


def plot_patch_atlas(
    graph: MapperGraph,
    cloud: TokenCloud,
    images: list[Any],
    node_ids: list[int],
    node_summary: pd.DataFrame | None = None,
    patches_per_node: int = 6,
    image_size: int = 256,
    title: str = "Stable Mapper-node patch atlas",
) -> None:
    if not node_ids:
        print("No nodes selected for the patch atlas.")
        return
    context = 2 if cloud.grid_shape[0] >= 16 else 1
    fig, axes = plt.subplots(len(node_ids), patches_per_node, figsize=(1.75 * patches_per_node, 1.95 * len(node_ids)))
    axes = np.asarray(axes).reshape(len(node_ids), patches_per_node)
    summary_lookup: dict[int, pd.Series] = {}
    if node_summary is not None and not node_summary.empty and "reference_node_id" in node_summary.columns:
        summary_lookup = {int(row["reference_node_id"]): row for _, row in node_summary.iterrows()}
    elif node_summary is not None and not node_summary.empty and "node_id" in node_summary.columns:
        summary_lookup = {int(row["node_id"]): row for _, row in node_summary.iterrows()}

    for row_i, node_id in enumerate(node_ids):
        sources = _node_representative_sources(graph, cloud, node_id, patches_per_node)
        for col_i in range(patches_per_node):
            ax = axes[row_i, col_i]
            if col_i < len(sources):
                ax.imshow(approximate_patch(cloud, images, sources[col_i], image_size=image_size, context_cells=context))
            ax.axis("off")
            if col_i == 0:
                label = f"node {node_id}"
                if node_id in summary_lookup:
                    row = summary_lookup[node_id]
                    parts = []
                    if "recurrence" in row and pd.notna(row["recurrence"]):
                        parts.append(f"rec={row['recurrence']:.2f}")
                    if "median_best_jaccard" in row and pd.notna(row["median_best_jaccard"]):
                        parts.append(f"J={row['median_best_jaccard']:.2f}")
                    if "label_purity" in row and pd.notna(row["label_purity"]):
                        parts.append(f"pur={row['label_purity']:.2f}")
                    if parts:
                        label = f"{label}\n" + ", ".join(parts)
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_stability_overview(stability: pd.DataFrame) -> None:
    if stability.empty:
        print("No stability rows to plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.scatterplot(
        data=stability,
        x="reference_size",
        y="median_best_jaccard",
        hue="recurrence",
        size="recurrence",
        sizes=(30, 160),
        palette="viridis",
        ax=axes[0],
    )
    axes[0].set_title("Reference nodes by size and match quality")
    axes[0].set_xscale("log")
    sns.histplot(stability["median_best_jaccard"], bins=min(20, max(5, int(math.sqrt(len(stability))))), ax=axes[1])
    axes[1].set_title("Best-match Jaccard distribution")
    axes[1].set_xlabel("median best Jaccard to other seeds")
    plt.tight_layout()
    plt.show()
