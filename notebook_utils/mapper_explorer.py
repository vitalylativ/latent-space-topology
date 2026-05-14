from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, SpectralEmbedding, TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection

from notebook_utils.encoder_explorer import (
    TokenCloud,
    approximate_patch,
    l2_normalize,
    sample_indices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLUX_CACHE = PROJECT_ROOT / "outputs" / "cycle_hunt" / "beans_param_sweep"
DATA_SWEEP_CACHE = PROJECT_ROOT / "outputs" / "cycle_hunt" / "data_sweep" / "cache"


@dataclass
class LensResult:
    name: str
    values: np.ndarray
    columns: list[str]
    notes: dict[str, Any]


@dataclass
class MapperGraph:
    name: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    memberships: pd.DataFrame
    lens_values: np.ndarray
    lens_columns: list[str]
    source_indices: np.ndarray
    notes: dict[str, Any]


def load_cached_flux_cloud(cache_dir: str | Path | None = None) -> TokenCloud | None:
    """Load the ignored local FLUX token cache if a previous cycle hunt created it."""
    folder = Path(cache_dir) if cache_dir is not None else DEFAULT_FLUX_CACHE
    cloud_path = folder / "flux_cloud_48_images.npz"
    metadata_path = folder / "flux_token_metadata_48_images.csv"
    if not cloud_path.exists() or not metadata_path.exists():
        return None

    payload = np.load(cloud_path)
    tokens = payload["tokens"].astype(np.float32)
    grid_shape = tuple(int(x) for x in np.ravel(payload["grid_shape"])[:2])
    channel_dim = int(np.ravel(payload["channel_dim"])[0])
    metadata = pd.read_csv(metadata_path)
    notes = {
        "source": "outputs/cycle_hunt/beans_param_sweep",
        "cache_path": str(cloud_path),
        "metadata_path": str(metadata_path),
    }
    return TokenCloud(
        name="flux_vae_cached",
        model_id="diffusers/FLUX.1-vae",
        family="AutoencoderKL",
        token_kind="posterior_mean",
        tokens=tokens,
        token_metadata=metadata,
        grid_shape=grid_shape,
        channel_dim=channel_dim,
        notes=notes,
    )


def load_token_cloud_cache(
    cloud_path: str | Path,
    metadata_path: str | Path | None = None,
    name: str | None = None,
    model_id: str = "diffusers/FLUX.1-vae",
    family: str = "AutoencoderKL",
    token_kind: str = "posterior_mean",
) -> TokenCloud:
    """Load a TokenCloud from the npz/csv cache format used by sweep scripts."""
    cloud_path = Path(cloud_path)
    if metadata_path is None:
        metadata_path = cloud_path.with_name(f"{cloud_path.stem}_token_metadata.csv")
    metadata_path = Path(metadata_path)
    payload = np.load(cloud_path)
    tokens = payload["tokens"].astype(np.float32)
    grid_shape = tuple(int(x) for x in np.ravel(payload["grid_shape"])[:2])
    channel_dim = int(np.ravel(payload["channel_dim"])[0])
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame(index=np.arange(len(tokens)))
    notes_path = cloud_path.with_name(f"{cloud_path.stem}_notes.json")
    notes = {
        "source": str(cloud_path),
        "cache_path": str(cloud_path),
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "notes_path": str(notes_path) if notes_path.exists() else None,
    }
    return TokenCloud(
        name=name or cloud_path.stem,
        model_id=model_id,
        family=family,
        token_kind=token_kind,
        tokens=tokens,
        token_metadata=metadata,
        grid_shape=grid_shape,
        channel_dim=channel_dim,
        notes=notes,
    )


def discover_cached_flux_clouds(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Find available FLUX token caches that can support offline Mapper notebooks."""
    folder = Path(cache_dir) if cache_dir is not None else DATA_SWEEP_CACHE
    rows = []
    if folder.exists():
        for cloud_path in sorted(folder.glob("flux_vae_*_n*_px*.npz")):
            metadata_path = cloud_path.with_name(f"{cloud_path.stem}_token_metadata.csv")
            notes_path = cloud_path.with_name(f"{cloud_path.stem}_notes.json")
            dataset = cloud_path.stem.removeprefix("flux_vae_").split("_n", 1)[0]
            row = {
                "dataset": dataset,
                "cloud_path": str(cloud_path),
                "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                "notes_path": str(notes_path) if notes_path.exists() else None,
            }
            try:
                payload = np.load(cloud_path)
                row["tokens"] = int(payload["tokens"].shape[0])
                row["channels"] = int(payload["tokens"].shape[1])
                row["grid_shape"] = tuple(int(x) for x in np.ravel(payload["grid_shape"])[:2])
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    legacy = DEFAULT_FLUX_CACHE / "flux_cloud_48_images.npz"
    legacy_meta = DEFAULT_FLUX_CACHE / "flux_token_metadata_48_images.csv"
    if legacy.exists() and legacy_meta.exists():
        payload = np.load(legacy)
        rows.append(
            {
                "dataset": "beans_param_sweep",
                "cloud_path": str(legacy),
                "metadata_path": str(legacy_meta),
                "notes_path": str(DEFAULT_FLUX_CACHE / "flux_cloud_48_images_notes.json"),
                "tokens": int(payload["tokens"].shape[0]),
                "channels": int(payload["tokens"].shape[1]),
                "grid_shape": tuple(int(x) for x in np.ravel(payload["grid_shape"])[:2]),
            }
        )
    return pd.DataFrame(rows)


def subsample_cloud(
    cloud: TokenCloud,
    max_points: int = 5000,
    seed: int = 72,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    idx = sample_indices(len(cloud.tokens), min(max_points, len(cloud.tokens)), seed=seed)
    x = cloud.tokens[idx].astype(np.float32)
    metadata = cloud.token_metadata.iloc[idx].reset_index(drop=True).copy()
    return x, metadata, idx


def standardize_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(scale, eps)


def kth_neighbor_distance(x: np.ndarray, k: int = 16) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros(len(x), dtype=np.float32)
    n_neighbors = min(k + 1, len(x))
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors(x)
    return distances[:, -1].astype(np.float32)


def _safe_pca(x: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, PCA | None]:
    n_components = min(n_components, x.shape[0] - 1, x.shape[1])
    if n_components < 1:
        return np.zeros((len(x), 1), dtype=np.float32), None
    pca = PCA(n_components=n_components, random_state=seed)
    values = pca.fit_transform(x).astype(np.float32)
    if values.shape[1] == 1:
        values = np.column_stack([values[:, 0], np.zeros(len(values), dtype=np.float32)])
    return values, pca


def make_lens_catalog(
    x: np.ndarray,
    seed: int = 72,
    include_slow: bool = False,
    n_neighbors: int = 16,
) -> dict[str, LensResult]:
    """Create several 2D filter functions for Mapper exploration."""
    x = x.astype(np.float32)
    unit = l2_normalize(x).astype(np.float32)
    norms = np.linalg.norm(x, axis=1)
    density = kth_neighbor_distance(unit, k=n_neighbors)

    catalog: dict[str, LensResult] = {}
    raw_pca, raw_model = _safe_pca(x, 2, seed)
    catalog["pca2_raw"] = LensResult(
        "pca2_raw",
        standardize_columns(raw_pca),
        ["pc1_raw", "pc2_raw"],
        {
            "view": "raw tokens",
            "explained_variance": None
            if raw_model is None
            else [float(v) for v in raw_model.explained_variance_ratio_[:2]],
        },
    )

    unit_pca, unit_model = _safe_pca(unit, 2, seed)
    catalog["pca2_unit"] = LensResult(
        "pca2_unit",
        standardize_columns(unit_pca),
        ["pc1_unit", "pc2_unit"],
        {
            "view": "L2-normalized directions",
            "explained_variance": None
            if unit_model is None
            else [float(v) for v in unit_model.explained_variance_ratio_[:2]],
        },
    )

    radial_density = np.column_stack([norms, density]).astype(np.float32)
    catalog["norm_density"] = LensResult(
        "norm_density",
        standardize_columns(radial_density),
        ["raw_norm", f"unit_k{n_neighbors}_distance"],
        {"view": "token norm plus local distance in unit-direction space"},
    )

    pca8, pca8_model = _safe_pca(x, min(8, x.shape[1]), seed)
    pca8_unit = l2_normalize(pca8).astype(np.float32)
    pca8_lens, pca8_lens_model = _safe_pca(pca8_unit, 2, seed)
    catalog["pca8_sphere_pca2"] = LensResult(
        "pca8_sphere_pca2",
        standardize_columns(pca8_lens),
        ["pc1_pca8_sphere", "pc2_pca8_sphere"],
        {
            "view": "PCA8 followed by L2 normalization",
            "pca8_explained_variance": None
            if pca8_model is None
            else float(np.sum(pca8_model.explained_variance_ratio_)),
            "lens_explained_variance": None
            if pca8_lens_model is None
            else [float(v) for v in pca8_lens_model.explained_variance_ratio_[:2]],
        },
    )

    if include_slow and len(x) >= 5:
        n_neighbors_eff = min(max(3, n_neighbors), len(x) - 1)
        try:
            isomap = Isomap(n_neighbors=n_neighbors_eff, n_components=2)
            values = isomap.fit_transform(unit).astype(np.float32)
            catalog["isomap2_unit"] = LensResult(
                "isomap2_unit",
                standardize_columns(values),
                ["isomap1_unit", "isomap2_unit"],
                {"view": "Isomap on L2-normalized directions", "n_neighbors": n_neighbors_eff},
            )
        except Exception as exc:
            catalog["isomap2_unit_failed"] = LensResult(
                "isomap2_unit_failed",
                np.zeros((len(x), 2), dtype=np.float32),
                ["failed_0", "failed_1"],
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    return catalog


def cover_intervals(values: np.ndarray, n_intervals: int, overlap: float) -> list[tuple[float, float]]:
    if n_intervals < 1:
        raise ValueError("n_intervals must be at least 1")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    lo = float(np.min(values))
    hi = float(np.max(values))
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    if n_intervals == 1:
        return [(lo, hi + 1e-8)]
    width = (hi - lo) / (n_intervals - (n_intervals - 1) * overlap)
    step = width * (1.0 - overlap)
    intervals = [(lo + i * step, lo + i * step + width) for i in range(n_intervals)]
    intervals[-1] = (intervals[-1][0], hi + 1e-8)
    return intervals


def _point_interval_memberships(values: np.ndarray, intervals: list[tuple[float, float]]) -> list[list[int]]:
    memberships: list[list[int]] = []
    for value in values:
        bins = [
            i
            for i, (lo, hi) in enumerate(intervals)
            if (lo <= value < hi) or (i == len(intervals) - 1 and lo <= value <= hi)
        ]
        memberships.append(bins)
    return memberships


def _cluster_preimage(
    x: np.ndarray,
    min_cluster_size: int,
    eps_quantile: float,
    clusterer: str,
) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=int)
    if clusterer not in {"dbscan", "single"}:
        raise ValueError(f"unknown clusterer {clusterer!r}; expected 'dbscan' or 'single'")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1")
    if not 0 <= eps_quantile <= 1:
        raise ValueError("eps_quantile must be in [0, 1]")
    if clusterer == "single" or len(x) < max(min_cluster_size * 2, 6):
        return np.zeros(len(x), dtype=int)

    n_neighbors = min(max(2, min_cluster_size), len(x))
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors(x)
    eps = float(np.quantile(distances[:, -1], eps_quantile))
    eps = max(eps, 1e-8)
    labels = DBSCAN(eps=eps, min_samples=min_cluster_size).fit_predict(x)
    if np.all(labels < 0):
        return np.zeros(len(x), dtype=int)
    return labels.astype(int)


def build_mapper_graph(
    x: np.ndarray,
    lens: LensResult | np.ndarray,
    name: str = "mapper",
    source_indices: np.ndarray | None = None,
    n_intervals: int = 8,
    overlap: float = 0.35,
    min_bin_points: int = 12,
    clusterer: str = "dbscan",
    min_cluster_size: int = 5,
    eps_quantile: float = 0.35,
) -> MapperGraph:
    x = np.asarray(x, dtype=np.float32)
    if min_bin_points < 1:
        raise ValueError("min_bin_points must be at least 1")
    if isinstance(lens, LensResult):
        lens_values = np.asarray(lens.values, dtype=np.float32)
        lens_columns = lens.columns
        lens_notes = lens.notes
    else:
        lens_values = np.asarray(lens, dtype=np.float32)
        lens_columns = [f"lens_{i}" for i in range(lens_values.shape[1] if lens_values.ndim > 1 else 1)]
        lens_notes = {}
    if lens_values.ndim == 1:
        lens_values = lens_values[:, None]
    if len(lens_values) != len(x):
        raise ValueError("lens must have one row per point")
    if source_indices is None:
        source_indices = np.arange(len(x), dtype=int)
    else:
        source_indices = np.asarray(source_indices, dtype=int)
        if len(source_indices) != len(x):
            raise ValueError("source_indices must have one entry per point")

    intervals_by_dim = [cover_intervals(lens_values[:, dim], n_intervals, overlap) for dim in range(lens_values.shape[1])]
    memberships_by_dim = [_point_interval_memberships(lens_values[:, dim], intervals) for dim, intervals in enumerate(intervals_by_dim)]
    cell_members: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for point_idx in range(len(x)):
        choices = [dim_members[point_idx] for dim_members in memberships_by_dim]
        if any(len(choice) == 0 for choice in choices):
            continue
        for bin_id in itertools.product(*choices):
            cell_members[tuple(int(v) for v in bin_id)].append(point_idx)

    node_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, int]] = []
    node_id = 0
    for bin_id, members in sorted(cell_members.items()):
        if len(members) < min_bin_points:
            continue
        members_arr = np.asarray(members, dtype=int)
        labels = _cluster_preimage(x[members_arr], min_cluster_size, eps_quantile, clusterer)
        for cluster_id in sorted(v for v in np.unique(labels) if v >= 0):
            local = np.flatnonzero(labels == cluster_id)
            if len(local) == 0:
                continue
            point_indices = members_arr[local]
            center = lens_values[point_indices].mean(axis=0)
            row = {
                "node_id": node_id,
                "bin_id": str(bin_id),
                "cluster_id": int(cluster_id),
                "size": int(len(point_indices)),
            }
            for dim, value in enumerate(center):
                row[f"lens_{dim}"] = float(value)
            node_rows.append(row)
            for point_index in point_indices:
                membership_rows.append(
                    {
                        "node_id": node_id,
                        "point_index": int(point_index),
                        "source_index": int(source_indices[point_index]),
                    }
                )
            node_id += 1

    node_columns = ["node_id", "bin_id", "cluster_id", "size", *[f"lens_{dim}" for dim in range(lens_values.shape[1])]]
    nodes = pd.DataFrame(node_rows, columns=node_columns)
    memberships = pd.DataFrame(membership_rows, columns=["node_id", "point_index", "source_index"])

    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    if not memberships.empty:
        for _, group in memberships.groupby("point_index"):
            node_ids = sorted(group["node_id"].unique().tolist())
            for a, b in itertools.combinations(node_ids, 2):
                edge_counts[(int(a), int(b))] += 1

    edges = pd.DataFrame(
        [
            {"source": a, "target": b, "shared_points": shared}
            for (a, b), shared in sorted(edge_counts.items())
            if a != b
        ],
        columns=["source", "target", "shared_points"],
    )

    notes = {
        "n_input_points": int(len(x)),
        "n_lens_dimensions": int(lens_values.shape[1]),
        "n_intervals": int(n_intervals),
        "overlap": float(overlap),
        "min_bin_points": int(min_bin_points),
        "clusterer": clusterer,
        "min_cluster_size": int(min_cluster_size),
        "eps_quantile": float(eps_quantile),
        "lens_notes": lens_notes,
    }
    return MapperGraph(name, nodes, edges, memberships, lens_values, lens_columns, source_indices, notes)


def _component_labels(n_nodes: int, edges: pd.DataFrame) -> np.ndarray:
    parent = np.arange(n_nodes, dtype=int)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if not edges.empty:
        for source, target in edges[["source", "target"]].itertuples(index=False):
            union(int(source), int(target))
    roots = np.asarray([find(i) for i in range(n_nodes)], dtype=int)
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def node_label_summary(graph: MapperGraph, metadata: pd.DataFrame | None, label_col: str = "label") -> pd.DataFrame:
    if metadata is None or label_col not in metadata.columns or graph.memberships.empty:
        return pd.DataFrame()
    expected = int(graph.notes.get("n_input_points", len(graph.lens_values)))
    if len(metadata) != expected:
        raise ValueError("metadata must be aligned to graph point_index rows; pass the subsampled metadata, not the full cloud metadata")
    labels = metadata[label_col].astype(str).to_numpy()
    rows = []
    for node_id, group in graph.memberships.groupby("node_id"):
        values = labels[group["point_index"].to_numpy(dtype=int)]
        counts = pd.Series(values).value_counts()
        total = int(counts.sum())
        rows.append(
            {
                "node_id": int(node_id),
                f"dominant_{label_col}": str(counts.index[0]),
                f"{label_col}_purity": float(counts.iloc[0] / total),
                f"{label_col}_entropy": float(-(counts / total * np.log2(counts / total)).sum()),
            }
        )
    return pd.DataFrame(rows)


def node_metadata_summary(
    graph: MapperGraph,
    x: np.ndarray,
    metadata: pd.DataFrame | None = None,
    label_col: str = "label",
) -> pd.DataFrame:
    """Summarize what each Mapper node contains in token, image, label, and grid terms."""
    if graph.nodes.empty:
        return pd.DataFrame()
    if graph.memberships.empty:
        return graph.nodes.copy()
    if metadata is not None and not metadata.empty and len(metadata) != len(x):
        raise ValueError("metadata must be aligned to x rows; pass the subsampled metadata, not the full cloud metadata")
    rows = []
    norms = np.linalg.norm(x, axis=1)
    component_labels = _component_labels(len(graph.nodes), graph.edges)
    component_lookup = dict(zip(graph.nodes["node_id"].astype(int), component_labels))
    degrees = np.zeros(len(graph.nodes), dtype=int)
    if not graph.edges.empty:
        for source, target in graph.edges[["source", "target"]].itertuples(index=False):
            degrees[int(source)] += 1
            degrees[int(target)] += 1
    degree_lookup = dict(zip(graph.nodes["node_id"].astype(int), degrees))

    for node_id, group in graph.memberships.groupby("node_id"):
        point_idx = group["point_index"].to_numpy(dtype=int)
        row: dict[str, Any] = {
            "node_id": int(node_id),
            "component": int(component_lookup.get(int(node_id), -1)),
            "degree": int(degree_lookup.get(int(node_id), 0)),
            "membership_rows": int(len(group)),
            "unique_points": int(len(np.unique(point_idx))),
            "token_norm_mean": float(norms[point_idx].mean()),
            "token_norm_std": float(norms[point_idx].std()),
        }
        if metadata is not None and not metadata.empty:
            meta = metadata.iloc[point_idx]
            if label_col in meta.columns:
                labels = meta[label_col].astype(str)
                counts = labels.value_counts()
                total = int(counts.sum())
                row[f"dominant_{label_col}"] = str(counts.index[0])
                row[f"{label_col}_purity"] = float(counts.iloc[0] / max(total, 1))
                row[f"{label_col}_entropy"] = float(-(counts / total * np.log2(counts / total)).sum()) if total else 0.0
            if "image_id" in meta.columns:
                image_counts = meta["image_id"].astype(str).value_counts()
                total = int(image_counts.sum())
                row["image_count"] = int(len(image_counts))
                row["dominant_image_fraction"] = float(image_counts.iloc[0] / max(total, 1))
                row["image_entropy"] = float(-(image_counts / total * np.log2(image_counts / total)).sum()) if total else 0.0
            if {"h", "w"}.issubset(meta.columns):
                h = meta["h"].to_numpy(dtype=float)
                w = meta["w"].to_numpy(dtype=float)
                row["h_mean"] = float(np.mean(h))
                row["w_mean"] = float(np.mean(w))
                row["h_std"] = float(np.std(h))
                row["w_std"] = float(np.std(w))
                row["spatial_radius"] = float(np.mean(np.sqrt((h - h.mean()) ** 2 + (w - w.mean()) ** 2)))
        rows.append(row)

    out = graph.nodes.merge(pd.DataFrame(rows), on="node_id", how="left")
    return out.sort_values(["degree", "size"], ascending=False).reset_index(drop=True)


def edge_overlap_summary(graph: MapperGraph) -> pd.DataFrame:
    if graph.edges.empty or graph.nodes.empty:
        return pd.DataFrame(columns=["source", "target", "shared_points", "jaccard", "source_size", "target_size"])
    sizes = graph.nodes.set_index("node_id")["size"].to_dict()
    rows = []
    for source, target, shared in graph.edges[["source", "target", "shared_points"]].itertuples(index=False):
        source_size = int(sizes.get(int(source), 0))
        target_size = int(sizes.get(int(target), 0))
        denom = max(source_size + target_size - int(shared), 1)
        rows.append(
            {
                "source": int(source),
                "target": int(target),
                "shared_points": int(shared),
                "source_size": source_size,
                "target_size": target_size,
                "jaccard": float(int(shared) / denom),
            }
        )
    return pd.DataFrame(rows).sort_values(["shared_points", "jaccard"], ascending=False).reset_index(drop=True)


def mapper_stats(graph: MapperGraph, metadata: pd.DataFrame | None = None, label_col: str = "label") -> dict[str, Any]:
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    if n_nodes == 0:
        base = {
            "nodes": 0,
            "edges": 0,
            "components": 0,
            "graph_h1_rank": 0,
            "coverage_fraction": 0.0,
            "mean_memberships_per_covered_point": 0.0,
            "mean_node_size": 0.0,
            "median_node_size": 0.0,
            "mean_degree": 0.0,
            "max_degree": 0,
            "largest_component_fraction": 0.0,
            "weighted_label_purity": np.nan,
        }
        base.update({k: v for k, v in graph.notes.items() if k != "lens_notes"})
        return base

    component_labels = _component_labels(n_nodes, graph.edges)
    components = int(len(np.unique(component_labels)))
    degrees = np.zeros(n_nodes, dtype=int)
    if not graph.edges.empty:
        for source, target in graph.edges[["source", "target"]].itertuples(index=False):
            degrees[int(source)] += 1
            degrees[int(target)] += 1
    component_sizes = pd.Series(component_labels).value_counts()
    covered = int(graph.memberships["point_index"].nunique()) if not graph.memberships.empty else 0
    n_input = int(graph.notes.get("n_input_points", len(graph.lens_values)))

    stats: dict[str, Any] = {
        "nodes": int(n_nodes),
        "edges": int(n_edges),
        "components": components,
        "graph_h1_rank": int(max(0, n_edges - n_nodes + components)),
        "coverage_fraction": float(covered / max(n_input, 1)),
        "mean_memberships_per_covered_point": float(len(graph.memberships) / max(covered, 1)),
        "mean_node_size": float(graph.nodes["size"].mean()),
        "median_node_size": float(graph.nodes["size"].median()),
        "mean_degree": float(degrees.mean()),
        "max_degree": int(degrees.max(initial=0)),
        "largest_component_fraction": float(component_sizes.max() / max(n_nodes, 1)),
    }
    label_summary = node_label_summary(graph, metadata, label_col=label_col)
    if not label_summary.empty:
        purity_col = f"{label_col}_purity"
        weights = graph.nodes.set_index("node_id").loc[label_summary["node_id"], "size"].to_numpy()
        stats[f"weighted_{label_col}_purity"] = float(np.average(label_summary[purity_col], weights=weights))
    else:
        stats[f"weighted_{label_col}_purity"] = np.nan
    stats.update({k: v for k, v in graph.notes.items() if k != "lens_notes"})
    return stats


def stability_summary(
    table: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    if metric_cols is None:
        metric_cols = [
            "nodes",
            "edges",
            "components",
            "graph_h1_rank",
            "coverage_fraction",
            "mean_degree",
            "largest_component_fraction",
            "weighted_label_purity",
        ]
    metric_cols = [col for col in metric_cols if col in table.columns]
    rows = []
    for keys, group in table.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["runs"] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_cv"] = float(std / abs(mean)) if abs(mean) > 1e-12 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def component_table(graph: MapperGraph) -> pd.DataFrame:
    if graph.nodes.empty:
        return pd.DataFrame(columns=["component", "nodes", "node_size_sum", "unique_points", "edges", "graph_h1_rank"])
    labels = _component_labels(len(graph.nodes), graph.edges)
    node_components = pd.DataFrame({"node_id": graph.nodes["node_id"].to_numpy(dtype=int), "component": labels})
    node_sizes = graph.nodes[["node_id", "size"]].merge(node_components, on="node_id")
    rows = []
    for component, group in node_sizes.groupby("component"):
        node_set = set(group["node_id"].tolist())
        edge_count = 0
        if not graph.edges.empty:
            edge_count = int(
                graph.edges.apply(lambda row: int(row["source"]) in node_set and int(row["target"]) in node_set, axis=1).sum()
            )
        unique_points = 0
        if not graph.memberships.empty:
            component_node_ids = set(group["node_id"].tolist())
            unique_points = int(graph.memberships[graph.memberships["node_id"].isin(component_node_ids)]["point_index"].nunique())
        rows.append(
            {
                "component": int(component),
                "nodes": int(len(group)),
                "node_size_sum": int(group["size"].sum()),
                "unique_points": unique_points,
                "edges": edge_count,
                "graph_h1_rank": int(max(0, edge_count - len(group) + 1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["nodes", "node_size_sum"], ascending=False).reset_index(drop=True)


def plot_lens_scatter(
    lens: LensResult | np.ndarray,
    metadata: pd.DataFrame | None = None,
    color_col: str = "label",
    max_points: int = 6000,
    seed: int = 72,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if isinstance(lens, LensResult):
        values = lens.values
        columns = lens.columns
        title = lens.name
    else:
        values = np.asarray(lens)
        columns = ["lens_0", "lens_1"]
        title = "lens"
    if values.ndim == 1:
        values = np.column_stack([values, np.zeros(len(values))])
    if values.shape[1] == 1:
        values = np.column_stack([values[:, 0], np.zeros(len(values))])
    idx = sample_indices(len(values), min(max_points, len(values)), seed=seed)
    plot_df = pd.DataFrame({"lens_0": values[idx, 0], "lens_1": values[idx, 1]})
    if metadata is not None and color_col in metadata.columns:
        plot_df[color_col] = metadata.iloc[idx][color_col].astype(str).to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 4.8))
    if color_col in plot_df.columns:
        sns.scatterplot(data=plot_df, x="lens_0", y="lens_1", hue=color_col, s=8, alpha=0.45, linewidth=0, ax=ax)
        ax.legend(fontsize=8, markerscale=2)
    else:
        ax.scatter(plot_df["lens_0"], plot_df["lens_1"], s=7, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(columns[0] if columns else "lens_0")
    ax.set_ylabel(columns[1] if len(columns) > 1 else "0")
    return ax


def plot_mapper_graph(
    graph: MapperGraph,
    metadata: pd.DataFrame | None = None,
    label_col: str = "label",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 4.8))
    if graph.nodes.empty:
        ax.set_title(title or f"{graph.name}: empty graph")
        return ax

    nodes = graph.nodes.copy()
    if "lens_1" not in nodes.columns:
        nodes["lens_1"] = 0.0
    label_summary = node_label_summary(graph, metadata, label_col=label_col)
    color_values: Any = nodes["size"]
    palette = None
    if not label_summary.empty:
        nodes = nodes.merge(label_summary, on="node_id", how="left")
        color_col = f"dominant_{label_col}"
        labels = sorted(nodes[color_col].fillna("unknown").unique())
        colors = sns.color_palette("tab10", n_colors=max(1, len(labels)))
        palette = {label: colors[i % len(colors)] for i, label in enumerate(labels)}
        color_values = nodes[color_col].fillna("unknown").map(palette)

    positions = nodes.set_index("node_id")[["lens_0", "lens_1"]]
    if not graph.edges.empty:
        segments = []
        widths = []
        for source, target, shared in graph.edges[["source", "target", "shared_points"]].itertuples(index=False):
            if source in positions.index and target in positions.index:
                segments.append([positions.loc[source].to_numpy(), positions.loc[target].to_numpy()])
                widths.append(0.4 + math.log1p(float(shared)) * 0.25)
        if segments:
            ax.add_collection(LineCollection(segments, colors="0.35", linewidths=widths, alpha=0.28, zorder=1))

    sizes = 28 + 9 * np.sqrt(nodes["size"].to_numpy(dtype=float))
    if palette is None:
        scatter = ax.scatter(nodes["lens_0"], nodes["lens_1"], s=sizes, c=color_values, cmap="viridis", alpha=0.88, edgecolor="white", linewidth=0.45, zorder=2)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, label="node size")
    else:
        ax.scatter(nodes["lens_0"], nodes["lens_1"], s=sizes, c=list(color_values), alpha=0.88, edgecolor="white", linewidth=0.45, zorder=2)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=7, label=label)
            for label, color in palette.items()
        ]
        ax.legend(handles=handles, title=label_col, fontsize=8)

    stats = mapper_stats(graph, metadata, label_col=label_col)
    ax.set_title(title or f"{graph.name}: V={stats['nodes']} E={stats['edges']} H1={stats['graph_h1_rank']}")
    ax.set_xlabel(graph.lens_columns[0] if graph.lens_columns else "lens_0")
    ax.set_ylabel(graph.lens_columns[1] if len(graph.lens_columns) > 1 else "0")
    ax.autoscale()
    return ax


def plot_mapper_diagnostics(
    graph: MapperGraph,
    metadata: pd.DataFrame | None = None,
    label_col: str = "label",
    max_points: int = 6000,
    seed: int = 72,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    plot_lens_scatter(graph.lens_values, metadata, label_col, max_points=max_points, seed=seed, ax=axes[0])
    axes[0].set_title(f"{graph.name}: lens scatter")
    plot_mapper_graph(graph, metadata, label_col=label_col, ax=axes[1])
    if graph.nodes.empty:
        axes[2].text(0.5, 0.5, "empty graph", ha="center", va="center")
        axes[2].set_axis_off()
    else:
        sns.histplot(graph.nodes["size"], bins=min(30, max(5, len(graph.nodes) // 2)), ax=axes[2])
        axes[2].set_title("node size distribution")
        axes[2].set_xlabel("points in node")
    plt.tight_layout()
    plt.show()


def plot_node_patches(
    graph: MapperGraph,
    cloud: TokenCloud,
    images: list[Any],
    node_ids: list[int] | None = None,
    patches_per_node: int = 4,
    max_nodes: int = 6,
    image_size: int = 256,
) -> None:
    if graph.memberships.empty:
        print("No node memberships to show.")
        return
    if node_ids is None:
        node_ids = graph.nodes.sort_values("size", ascending=False)["node_id"].head(max_nodes).astype(int).tolist()
    node_ids = node_ids[:max_nodes]
    fig, axes = plt.subplots(len(node_ids), patches_per_node, figsize=(2.0 * patches_per_node, 2.2 * len(node_ids)))
    axes = np.asarray(axes).reshape(len(node_ids), patches_per_node)
    for row_i, node_id in enumerate(node_ids):
        rows = graph.memberships[graph.memberships["node_id"] == node_id].head(patches_per_node)
        for col_i in range(patches_per_node):
            ax = axes[row_i, col_i]
            if col_i < len(rows):
                source_index = int(rows.iloc[col_i]["source_index"])
                context = 2 if cloud.grid_shape[0] >= 16 else 1
                ax.imshow(approximate_patch(cloud, images, source_index, image_size=image_size, context_cells=context))
                node_size = int(graph.nodes.loc[graph.nodes["node_id"] == node_id, "size"].iloc[0])
                ax.set_title(f"node {node_id} n={node_size}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def run_mapper_sweep(
    x: np.ndarray,
    lenses: dict[str, LensResult],
    configs: list[dict[str, Any]],
    source_indices: np.ndarray | None = None,
    metadata: pd.DataFrame | None = None,
    label_col: str = "label",
) -> tuple[pd.DataFrame, dict[tuple[str, str], MapperGraph]]:
    rows = []
    graphs: dict[tuple[str, str], MapperGraph] = {}
    for lens_name, lens in lenses.items():
        if lens_name.endswith("_failed"):
            continue
        for config in configs:
            config_name = str(config.get("config", f"n{config.get('n_intervals', 'x')}"))
            graph = build_mapper_graph(
                x,
                lens,
                name=f"{lens_name}:{config_name}",
                source_indices=source_indices,
                n_intervals=int(config.get("n_intervals", 8)),
                overlap=float(config.get("overlap", 0.35)),
                min_bin_points=int(config.get("min_bin_points", 12)),
                clusterer=str(config.get("clusterer", "dbscan")),
                min_cluster_size=int(config.get("min_cluster_size", 5)),
                eps_quantile=float(config.get("eps_quantile", 0.35)),
            )
            row = {"lens": lens_name, "config": config_name}
            row.update(mapper_stats(graph, metadata, label_col=label_col))
            rows.append(row)
            graphs[(lens_name, config_name)] = graph
    return pd.DataFrame(rows), graphs


def make_control_clouds(x: np.ndarray, seed: int = 72) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = l2_normalize(x).astype(np.float32)
    shuffled = x.copy()
    for dim in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, dim])
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
    gaussian = rng.multivariate_normal(mean, cov, size=len(x)).astype(np.float32)
    sphere = rng.normal(size=x.shape).astype(np.float32)
    sphere = l2_normalize(sphere).astype(np.float32)
    norm_random_directions = sphere * norms
    shuffled_norms = unit * rng.permutation(norms.reshape(-1)).reshape(-1, 1)
    return {
        "observed": x,
        "channel_shuffle": shuffled,
        "matched_gaussian": gaussian,
        "uniform_sphere": sphere,
        "norm_random_directions": norm_random_directions.astype(np.float32),
        "shuffled_norms": shuffled_norms.astype(np.float32),
    }


def make_projection_catalog(
    x: np.ndarray,
    seed: int = 72,
    include_slow: bool = False,
    n_neighbors: int = 16,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    x = x.astype(np.float32)
    unit = l2_normalize(x).astype(np.float32)
    projections: dict[str, np.ndarray] = {}
    failures = []

    def add(name: str, fn) -> None:
        try:
            values = fn()
            projections[name] = standardize_columns(np.asarray(values, dtype=np.float32))
        except Exception as exc:
            failures.append({"projection": name, "error_type": type(exc).__name__, "error": str(exc)})

    add("pca_raw", lambda: PCA(n_components=2, random_state=seed).fit_transform(x))
    add("pca_unit", lambda: PCA(n_components=2, random_state=seed).fit_transform(unit))
    add("gaussian_random_projection", lambda: GaussianRandomProjection(n_components=2, random_state=seed).fit_transform(unit))

    n_neighbors_eff = min(max(3, n_neighbors), len(x) - 1)
    add("isomap_unit", lambda: Isomap(n_neighbors=n_neighbors_eff, n_components=2).fit_transform(unit))
    add(
        "spectral_unit",
        lambda: SpectralEmbedding(n_components=2, n_neighbors=n_neighbors_eff, random_state=seed).fit_transform(unit),
    )
    if include_slow:
        perplexity = min(30, max(5, (len(x) - 1) // 3))
        add(
            "tsne_unit",
            lambda: TSNE(
                n_components=2,
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
                random_state=seed,
                max_iter=750,
            ).fit_transform(unit),
        )
    return projections, pd.DataFrame(failures)


def knn_jaccard(x: np.ndarray, y: np.ndarray, n_neighbors: int = 12) -> float:
    if len(x) <= 2:
        return float("nan")
    k = min(n_neighbors, len(x) - 1)
    x_idx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)[:, 1:]
    y_idx = NearestNeighbors(n_neighbors=k + 1).fit(y).kneighbors(y, return_distance=False)[:, 1:]
    scores = []
    for a, b in zip(x_idx, y_idx):
        sa = set(int(v) for v in a)
        sb = set(int(v) for v in b)
        scores.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(scores))


def projection_quality_table(
    x: np.ndarray,
    projections: dict[str, np.ndarray],
    metadata: pd.DataFrame | None = None,
    label_col: str = "label",
    n_neighbors: int = 12,
    max_distance_points: int = 900,
    seed: int = 72,
) -> pd.DataFrame:
    rows = []
    x = x.astype(np.float32)
    distance_idx = sample_indices(len(x), min(max_distance_points, len(x)), seed=seed)
    original_distances = pdist(x[distance_idx], metric="euclidean")
    labels = None
    if metadata is not None and label_col in metadata.columns:
        labels = metadata[label_col].astype(str).to_numpy()
    for name, values in projections.items():
        y = np.asarray(values, dtype=np.float32)
        k = min(n_neighbors, len(x) - 1)
        row = {
            "projection": name,
            "trustworthiness": float(trustworthiness(x, y, n_neighbors=k)),
            "knn_jaccard": knn_jaccard(x, y, n_neighbors=k),
            "pairwise_distance_spearman": float(spearmanr(original_distances, pdist(y[distance_idx], metric="euclidean")).statistic),
        }
        if labels is not None and len(np.unique(labels)) >= 2 and min(pd.Series(labels).value_counts()) >= 2:
            try:
                row[f"{label_col}_silhouette_2d"] = float(silhouette_score(y, labels, metric="euclidean"))
            except Exception:
                row[f"{label_col}_silhouette_2d"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trustworthiness", ascending=False).reset_index(drop=True)


def plot_projection_catalog(
    projections: dict[str, np.ndarray],
    metadata: pd.DataFrame | None = None,
    label_col: str = "label",
    max_points: int = 5000,
    seed: int = 72,
) -> None:
    if not projections:
        print("No projections to plot.")
        return
    cols = min(3, len(projections))
    rows = math.ceil(len(projections) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, values) in zip(axes, projections.items()):
        lens = LensResult(name, values, ["dim1", "dim2"], {})
        plot_lens_scatter(lens, metadata, color_col=label_col, max_points=max_points, seed=seed, ax=ax)
        ax.set_title(name)
    for ax in axes[len(projections) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
