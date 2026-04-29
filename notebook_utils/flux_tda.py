from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from notebook_utils.encoder_explorer import TokenCloud, approximate_patch, l2_normalize, sample_indices
from notebook_utils.flux_deep_dive import latent_tensor


@dataclass
class TDASample:
    name: str
    tokens: np.ndarray
    source_indices: np.ndarray
    metadata: pd.DataFrame
    notes: dict[str, Any]


def kth_neighbor_distance(x: np.ndarray, k: int = 16) -> np.ndarray:
    k_eff = min(k + 1, len(x))
    distances, _ = NearestNeighbors(n_neighbors=k_eff).fit(x).kneighbors(x)
    return distances[:, -1]


def select_dense_indices(
    x: np.ndarray,
    n_dense: int = 700,
    k: int = 16,
    max_candidates: int = 7000,
    seed: int = 72,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_idx = sample_indices(len(x), min(max_candidates, len(x)), seed=seed)
    candidate_x = x[candidate_idx]
    kth = kth_neighbor_distance(candidate_x, k=k)
    order = np.argsort(kth)[: min(n_dense, len(candidate_idx))]
    return candidate_idx[order], kth[order]


def farthest_point_indices(x: np.ndarray, n_landmarks: int = 120, seed: int = 72) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n <= n_landmarks:
        return np.arange(n)
    first = int(rng.integers(0, n))
    selected = [first]
    min_dist = np.linalg.norm(x - x[first], axis=1)
    for _ in range(1, n_landmarks):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist = np.linalg.norm(x - x[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return np.asarray(selected, dtype=int)


def build_observed_tda_sample(
    cloud: TokenCloud,
    n_dense: int = 700,
    n_landmarks: int = 120,
    k_density: int = 16,
    seed: int = 72,
) -> tuple[TDASample, pd.DataFrame]:
    sphere = l2_normalize(cloud.tokens.astype(np.float32)).astype(np.float32)
    dense_global_idx, dense_kth = select_dense_indices(sphere, n_dense=n_dense, k=k_density, seed=seed)
    dense_tokens = sphere[dense_global_idx]
    landmark_local_idx = farthest_point_indices(dense_tokens, n_landmarks=n_landmarks, seed=seed)
    landmark_global_idx = dense_global_idx[landmark_local_idx]
    sample = TDASample(
        name="observed_dense_sphere_landmarks",
        tokens=dense_tokens[landmark_local_idx],
        source_indices=landmark_global_idx,
        metadata=cloud.token_metadata.iloc[landmark_global_idx].reset_index(drop=True),
        notes={
            "view": "unit sphere",
            "density": "small kth-neighbor distance",
            "n_dense": len(dense_global_idx),
            "n_landmarks": len(landmark_global_idx),
            "k_density": k_density,
        },
    )
    dense_table = cloud.token_metadata.iloc[dense_global_idx].reset_index(drop=True).copy()
    dense_table["source_index"] = dense_global_idx
    dense_table["kth_distance"] = dense_kth
    dense_table["is_landmark"] = dense_table["source_index"].isin(set(landmark_global_idx.tolist()))
    return sample, dense_table


def build_random_sphere_sample(cloud: TokenCloud, n_landmarks: int = 120, seed: int = 72) -> TDASample:
    sphere = l2_normalize(cloud.tokens.astype(np.float32)).astype(np.float32)
    idx = sample_indices(len(sphere), n_landmarks, seed=seed)
    return TDASample(
        name="random_sphere_tokens",
        tokens=sphere[idx],
        source_indices=idx,
        metadata=cloud.token_metadata.iloc[idx].reset_index(drop=True),
        notes={"view": "unit sphere", "selection": "random token subset"},
    )


def build_uniform_sphere_sample(dim: int, n_landmarks: int = 120, seed: int = 72) -> TDASample:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_landmarks, dim)).astype(np.float32)
    x = l2_normalize(x).astype(np.float32)
    return TDASample(
        name="uniform_sphere_control",
        tokens=x,
        source_indices=np.arange(n_landmarks),
        metadata=pd.DataFrame({"image_id": ["control"] * n_landmarks, "label": ["uniform"] * n_landmarks, "h": 0, "w": 0}),
        notes={"control": "uniform random directions on the same ambient sphere"},
    )


def build_channel_shuffle_dense_sample(
    cloud: TokenCloud,
    n_dense: int = 700,
    n_landmarks: int = 120,
    k_density: int = 16,
    seed: int = 72,
) -> TDASample:
    rng = np.random.default_rng(seed)
    x = cloud.tokens.astype(np.float32).copy()
    for j in range(x.shape[1]):
        rng.shuffle(x[:, j])
    sphere = l2_normalize(x).astype(np.float32)
    dense_global_idx, _ = select_dense_indices(sphere, n_dense=n_dense, k=k_density, seed=seed)
    dense_tokens = sphere[dense_global_idx]
    landmark_local_idx = farthest_point_indices(dense_tokens, n_landmarks=n_landmarks, seed=seed)
    landmark_global_idx = dense_global_idx[landmark_local_idx]
    return TDASample(
        name="channel_shuffle_dense_sphere",
        tokens=dense_tokens[landmark_local_idx],
        source_indices=landmark_global_idx,
        metadata=cloud.token_metadata.iloc[landmark_global_idx].reset_index(drop=True),
        notes={"control": "shuffle each channel marginal, then sphere-project and density-select"},
    )


def build_matched_gaussian_sphere_sample(cloud: TokenCloud, n_landmarks: int = 120, seed: int = 72) -> TDASample:
    rng = np.random.default_rng(seed)
    raw = cloud.tokens.astype(np.float32)
    idx = sample_indices(len(raw), min(5000, len(raw)), seed=seed)
    x = raw[idx]
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
    g = rng.multivariate_normal(mean, cov, size=n_landmarks).astype(np.float32)
    g = l2_normalize(g).astype(np.float32)
    return TDASample(
        name="matched_gaussian_then_sphere",
        tokens=g,
        source_indices=np.arange(n_landmarks),
        metadata=pd.DataFrame({"image_id": ["control"] * n_landmarks, "label": ["gaussian"] * n_landmarks, "h": 0, "w": 0}),
        notes={"control": "Gaussian matched to raw mean/covariance, then projected to sphere"},
    )


def ripser_diagrams(sample: TDASample, maxdim: int = 2, distance_quantile: float = 0.85) -> dict[str, Any]:
    import ripser

    distances = pdist(sample.tokens, metric="euclidean")
    thresh = float(np.quantile(distances, distance_quantile))
    result = ripser.ripser(sample.tokens, maxdim=maxdim, thresh=thresh)
    return {
        "sample": sample.name,
        "diagrams": result["dgms"],
        "threshold": thresh,
        "distance_quantile": distance_quantile,
        "n_points": len(sample.tokens),
        "maxdim": maxdim,
    }


def diagram_summary(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for dim, diagram in enumerate(result["diagrams"]):
        finite = diagram[np.isfinite(diagram[:, 1])]
        persistence = finite[:, 1] - finite[:, 0] if len(finite) else np.array([])
        rows.append(
            {
                "sample": result["sample"],
                "dim": dim,
                "n_features": len(diagram),
                "n_finite": len(finite),
                "max_persistence": float(persistence.max()) if len(persistence) else 0.0,
                "mean_persistence": float(persistence.mean()) if len(persistence) else 0.0,
                "threshold": result["threshold"],
            }
        )
    return pd.DataFrame(rows)


def top_persistence_table(result: dict[str, Any], top_n: int = 8) -> pd.DataFrame:
    rows = []
    for dim, diagram in enumerate(result["diagrams"]):
        finite = diagram[np.isfinite(diagram[:, 1])]
        if not len(finite):
            continue
        persistence = finite[:, 1] - finite[:, 0]
        order = np.argsort(persistence)[-top_n:][::-1]
        for rank, idx in enumerate(order, start=1):
            birth, death = finite[idx]
            rows.append(
                {
                    "sample": result["sample"],
                    "dim": dim,
                    "rank": rank,
                    "birth": float(birth),
                    "death": float(death),
                    "persistence": float(death - birth),
                    "threshold": result["threshold"],
                }
            )
    return pd.DataFrame(rows)


def betti_curves(result: dict[str, Any], n_grid: int = 120) -> pd.DataFrame:
    threshold = result["threshold"]
    grid = np.linspace(0, threshold, n_grid)
    rows = []
    for dim, diagram in enumerate(result["diagrams"]):
        for t in grid:
            alive = (diagram[:, 0] <= t) & (diagram[:, 1] > t)
            rows.append({"sample": result["sample"], "dim": dim, "scale": t, "betti": int(alive.sum())})
    return pd.DataFrame(rows)


def plot_pipeline_projection(
    cloud: TokenCloud,
    dense_table: pd.DataFrame,
    sample: TDASample,
    max_points: int = 5000,
    seed: int = 72,
) -> None:
    sphere = l2_normalize(cloud.tokens.astype(np.float32)).astype(np.float32)
    idx = sample_indices(len(sphere), min(max_points, len(sphere)), seed=seed)
    xy = PCA(n_components=2, random_state=seed).fit_transform(sphere[idx])
    plot_df = cloud.token_metadata.iloc[idx].reset_index(drop=True).copy()
    plot_df["pc1"] = xy[:, 0]
    plot_df["pc2"] = xy[:, 1]
    plot_df["norm"] = np.linalg.norm(cloud.tokens[idx], axis=1)
    dense_set = set(dense_table["source_index"].tolist())
    landmark_set = set(sample.source_indices.tolist())
    plot_df["stage"] = "all sampled sphere tokens"
    plot_df.loc[[i in dense_set for i in idx], "stage"] = "dense tokens"
    plot_df.loc[[i in landmark_set for i in idx], "stage"] = "landmarks"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    sc = axes[0].scatter(plot_df["pc1"], plot_df["pc2"], c=plot_df["norm"], s=5, alpha=0.45, cmap="viridis")
    axes[0].set_title("sphere tokens colored by raw norm")
    plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.03)
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="stage", s=10, alpha=0.7, ax=axes[1])
    axes[1].set_title("pipeline stages in PCA view")
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="label", s=8, alpha=0.55, ax=axes[2])
    axes[2].set_title("same view colored by label")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_dense_maps(cloud: TokenCloud, dense_table: pd.DataFrame, image_ids: list[int]) -> None:
    z = latent_tensor(cloud)
    h, w = cloud.grid_shape
    fig, axes = plt.subplots(2, len(image_ids), figsize=(3.3 * len(image_ids), 6.2))
    axes = np.asarray(axes).reshape(2, len(image_ids))
    dense_lookup = dense_table.groupby("image_id")
    for col, image_id in enumerate(image_ids):
        norm = np.linalg.norm(z[image_id], axis=-1)
        axes[0, col].imshow(norm, cmap="magma")
        axes[0, col].set_title(f"image {image_id}: norm")
        axes[0, col].axis("off")
        mask = np.zeros((h, w), dtype=float)
        if image_id in dense_lookup.groups:
            rows = dense_lookup.get_group(image_id)
            mask[rows["h"].to_numpy(dtype=int), rows["w"].to_numpy(dtype=int)] = 1.0
        axes[1, col].imshow(mask, cmap="Greens", vmin=0, vmax=1)
        axes[1, col].set_title("dense-token mask")
        axes[1, col].axis("off")
    plt.tight_layout()
    plt.show()


def plot_landmark_patches(sample: TDASample, cloud: TokenCloud, images: list[Image.Image], image_size: int = 256, n: int = 12) -> None:
    n = min(n, len(sample.source_indices))
    cols = min(6, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, token_idx in zip(axes, sample.source_indices[:n]):
        ax.imshow(approximate_patch(cloud, images, int(token_idx), image_size=image_size, context_cells=2))
        row = cloud.token_metadata.iloc[int(token_idx)]
        ax.set_title(f"img {row['image_id']} ({row['h']},{row['w']})", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("source patches for selected TDA landmarks", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_landmark_distance_matrix(sample: TDASample) -> None:
    dist = squareform(pdist(sample.tokens, metric="euclidean"))
    order = np.argsort(PCA(n_components=1, random_state=72).fit_transform(sample.tokens).reshape(-1))
    ordered = dist[order][:, order]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    sns.heatmap(dist, cmap="mako", square=True, cbar_kws={"label": "distance"}, ax=axes[0])
    axes[0].set_title("landmark distance matrix")
    sns.heatmap(ordered, cmap="mako", square=True, cbar_kws={"label": "distance"}, ax=axes[1])
    axes[1].set_title("same matrix ordered by PC1")
    for ax in axes:
        ax.set_xlabel("landmark")
        ax.set_ylabel("landmark")
    fig.suptitle(sample.name, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_filtration_graph_snapshots(
    sample: TDASample,
    result: dict[str, Any],
    scales: list[float] | None = None,
    max_edges: int = 1800,
    seed: int = 72,
) -> None:
    xy = PCA(n_components=2, random_state=seed).fit_transform(sample.tokens)
    dist = squareform(pdist(sample.tokens, metric="euclidean"))
    if scales is None:
        pairwise = dist[np.triu_indices_from(dist, k=1)]
        quantiles = [0.18, 0.35, 0.55, min(0.78, result["distance_quantile"])]
        scales = [float(np.quantile(pairwise, q)) for q in quantiles]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(scales), figsize=(4.2 * len(scales), 4.2))
    axes = np.asarray([axes]).reshape(-1)
    for ax, scale in zip(axes, scales):
        edge_idx = np.argwhere(np.triu(dist <= scale, k=1))
        if len(edge_idx) > max_edges:
            edge_idx = edge_idx[rng.choice(len(edge_idx), size=max_edges, replace=False)]
        segments = np.stack([xy[edge_idx[:, 0]], xy[edge_idx[:, 1]]], axis=1) if len(edge_idx) else []
        if len(edge_idx):
            ax.add_collection(LineCollection(segments, colors="0.55", linewidths=0.35, alpha=0.22))
        ax.scatter(xy[:, 0], xy[:, 1], s=18, c="C0", alpha=0.88, edgecolor="white", linewidth=0.25)
        ax.set_title(f"scale {scale:.3f}\nedges shown: {len(edge_idx)}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale()
    fig.suptitle(f"Rips filtration snapshots: {sample.name}", y=1.05)
    plt.tight_layout()
    plt.show()


def plot_pairwise_distance_hist(samples: list[TDASample]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for sample in samples:
        distances = pdist(sample.tokens, metric="euclidean")
        sns.kdeplot(distances, ax=ax, label=sample.name)
    ax.set_title("pairwise landmark distance distributions")
    ax.set_xlabel("Euclidean chord distance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_persistence_lifetimes(results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        for dim, diagram in enumerate(result["diagrams"]):
            finite = diagram[np.isfinite(diagram[:, 1])]
            for birth, death in finite:
                rows.append(
                    {
                        "sample": result["sample"],
                        "dim": dim,
                        "birth": float(birth),
                        "death": float(death),
                        "persistence": float(death - birth),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        print("No finite persistence features to plot.")
        return
    g = sns.relplot(
        data=df,
        x="birth",
        y="persistence",
        hue="sample",
        col="dim",
        kind="scatter",
        height=3.8,
        aspect=1.15,
        facet_kws={"sharey": False},
        alpha=0.7,
        s=24,
    )
    g.fig.suptitle("Birth scale versus persistence", y=1.05)
    plt.show()


def plot_diagrams(results: list[dict[str, Any]]) -> None:
    import persim

    fig, axes = plt.subplots(len(results), 1, figsize=(6.5, 4.2 * len(results)))
    axes = np.asarray([axes]).reshape(-1)
    for ax, result in zip(axes, results):
        persim.plot_diagrams(result["diagrams"], ax=ax, show=False)
        ax.set_title(f"{result['sample']} (n={result['n_points']}, thresh={result['threshold']:.3f})")
    plt.tight_layout()
    plt.show()


def plot_betti_curves(results: list[dict[str, Any]]) -> None:
    curves = pd.concat([betti_curves(result) for result in results], ignore_index=True)
    g = sns.relplot(
        data=curves,
        x="scale",
        y="betti",
        hue="sample",
        col="dim",
        kind="line",
        facet_kws={"sharey": False},
        height=3.6,
        aspect=1.2,
    )
    g.fig.suptitle("Betti curves across filtration scale", y=1.05)
    plt.show()


def plot_barcode(result: dict[str, Any], max_bars_per_dim: int = 45) -> None:
    dims = len(result["diagrams"])
    fig, axes = plt.subplots(dims, 1, figsize=(9, 2.4 * dims))
    axes = np.asarray([axes]).reshape(-1)
    threshold = result["threshold"]
    for dim, (ax, diagram) in enumerate(zip(axes, result["diagrams"])):
        finite = diagram[np.isfinite(diagram[:, 1])]
        if not len(finite):
            ax.set_title(f"H{dim}: no finite bars")
            continue
        pers = finite[:, 1] - finite[:, 0]
        order = np.argsort(pers)[-max_bars_per_dim:]
        chosen = finite[order]
        for y, (birth, death) in enumerate(chosen):
            ax.plot([birth, min(death, threshold)], [y, y], color=f"C{dim}", linewidth=1.6)
        ax.set_title(f"H{dim}: longest finite bars")
        ax.set_xlim(0, threshold)
        ax.set_ylabel("bars")
    axes[-1].set_xlabel("filtration scale")
    fig.suptitle(result["sample"], y=1.02)
    plt.tight_layout()
    plt.show()


def plot_persistence_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.barplot(data=summary, x="sample", y="n_finite", hue="dim", ax=axes[0])
    axes[0].set_title("finite feature counts")
    axes[0].tick_params(axis="x", rotation=25)
    sns.barplot(data=summary, x="sample", y="max_persistence", hue="dim", ax=axes[1])
    axes[1].set_title("maximum finite persistence")
    axes[1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.show()
