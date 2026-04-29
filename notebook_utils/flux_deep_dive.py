from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from IPython.display import display
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from notebook_utils.encoder_explorer import (
    TokenCloud,
    approximate_patch,
    center_crop_resize,
    l2_normalize,
    pca_whiten,
    sample_indices,
    safe_hist,
)


def latent_tensor(cloud: TokenCloud) -> np.ndarray:
    h, w = cloud.grid_shape
    c = cloud.channel_dim
    b = cloud.tokens.shape[0] // (h * w)
    return cloud.tokens.reshape(b, h, w, c)


def channel_summary(cloud: TokenCloud) -> pd.DataFrame:
    x = cloud.tokens
    rows = []
    for i in range(x.shape[1]):
        values = x[:, i]
        rows.append(
            {
                "channel": i,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "skew": float(skew(values)),
                "kurtosis": float(kurtosis(values)),
            }
        )
    return pd.DataFrame(rows)


def plot_original_reconstruction(images: list[Image.Image], cloud: TokenCloud, image_size: int = 256, max_images: int = 4) -> None:
    if not cloud.reconstructions:
        print("No reconstructions stored on this cloud.")
        return
    n = min(max_images, len(images), len(cloud.reconstructions))
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.0))
    axes = np.asarray(axes).reshape(2, n)
    for j in range(n):
        axes[0, j].imshow(center_crop_resize(images[j], image_size))
        axes[0, j].set_title("input", fontsize=10)
        axes[0, j].axis("off")
        axes[1, j].imshow(cloud.reconstructions[j])
        axes[1, j].set_title("decoded latent", fontsize=10)
        axes[1, j].axis("off")
    plt.tight_layout()
    plt.show()


def plot_latent_channel_maps(cloud: TokenCloud, image_id: int = 0, max_abs_percentile: float = 99.0) -> None:
    z = latent_tensor(cloud)[image_id]
    vmax = float(np.percentile(np.abs(z), max_abs_percentile))
    cols = 4
    rows = math.ceil(cloud.channel_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows))
    axes = np.asarray(axes).reshape(-1)
    for c, ax in enumerate(axes[: cloud.channel_dim]):
        im = ax.imshow(z[:, :, c], cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"channel {c}", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes[cloud.channel_dim :]:
        ax.axis("off")
    fig.suptitle(f"FLUX latent channel maps for image {image_id}", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_latent_norm_and_rgb_pca_maps(cloud: TokenCloud, image_id: int = 0) -> None:
    z = latent_tensor(cloud)[image_id]
    norm = np.linalg.norm(z, axis=-1)
    flat = z.reshape(-1, z.shape[-1])
    pca = PCA(n_components=3, random_state=72)
    rgb = pca.fit_transform(flat).reshape(z.shape[0], z.shape[1], 3)
    rgb = (rgb - rgb.min(axis=(0, 1), keepdims=True)) / np.maximum(np.ptp(rgb, axis=(0, 1), keepdims=True), 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im = axes[0].imshow(norm, cmap="magma")
    axes[0].set_title("token norm map")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].imshow(rgb)
    axes[1].set_title("first 3 latent PCs as RGB")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def plot_channel_distributions(cloud: TokenCloud) -> None:
    x = cloud.tokens
    cols = 4
    rows = math.ceil(x.shape[1] / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.5 * rows))
    axes = np.asarray(axes).reshape(-1)
    for c, ax in enumerate(axes[: x.shape[1]]):
        safe_hist(ax, x[:, c], bins=50, alpha=0.85)
        ax.set_title(f"channel {c}", fontsize=10)
    for ax in axes[x.shape[1] :]:
        ax.axis("off")
    fig.suptitle("Per-channel latent value distributions", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_channel_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    sns.barplot(data=summary, x="channel", y="mean", ax=axes[0, 0])
    axes[0, 0].set_title("channel means")
    sns.barplot(data=summary, x="channel", y="std", ax=axes[0, 1])
    axes[0, 1].set_title("channel std")
    sns.barplot(data=summary, x="channel", y="skew", ax=axes[1, 0])
    axes[1, 0].set_title("channel skew")
    sns.barplot(data=summary, x="channel", y="kurtosis", ax=axes[1, 1])
    axes[1, 1].set_title("channel excess kurtosis")
    for ax in axes.reshape(-1):
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()


def plot_channel_correlation(cloud: TokenCloud) -> None:
    corr = np.corrcoef(cloud.tokens, rowvar=False)
    cov = np.cov(cloud.tokens, rowvar=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(corr, cmap="vlag", center=0, ax=axes[0])
    axes[0].set_title("channel correlation")
    sns.heatmap(cov, cmap="mako", ax=axes[1])
    axes[1].set_title("channel covariance")
    plt.tight_layout()
    plt.show()


def plot_norm_maps_many(images: list[Image.Image], cloud: TokenCloud, image_ids: list[int], image_size: int = 256) -> None:
    z = latent_tensor(cloud)
    n = len(image_ids)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.0))
    axes = np.asarray(axes).reshape(2, n)
    for col, image_id in enumerate(image_ids):
        axes[0, col].imshow(center_crop_resize(images[image_id], image_size))
        axes[0, col].set_title(f"image {image_id}", fontsize=10)
        axes[0, col].axis("off")
        im = axes[1, col].imshow(np.linalg.norm(z[image_id], axis=-1), cmap="magma")
        axes[1, col].set_title("latent norm", fontsize=10)
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def pca_spectrum(cloud: TokenCloud, max_points: int = 6000, seed: int = 72) -> tuple[PCA, np.ndarray]:
    idx = sample_indices(len(cloud.tokens), max_points, seed=seed)
    x = cloud.tokens[idx]
    pca = PCA(n_components=min(cloud.channel_dim, x.shape[0] - 1), random_state=seed).fit(x)
    return pca, idx


def plot_pca_spectrum_and_loadings(cloud: TokenCloud, max_points: int = 6000, seed: int = 72) -> PCA:
    pca, _ = pca_spectrum(cloud, max_points=max_points, seed=seed)
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(components, pca.explained_variance_ratio_)
    axes[0].plot(components, np.cumsum(pca.explained_variance_ratio_), marker="o", color="black")
    axes[0].set_title("PCA spectrum and cumulative variance")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("explained variance ratio")
    sns.heatmap(pd.DataFrame(pca.components_[:6], columns=[f"ch{i}" for i in range(cloud.channel_dim)]), cmap="vlag", center=0, ax=axes[1])
    axes[1].set_title("first six PCA loading vectors")
    axes[1].set_ylabel("PC")
    plt.tight_layout()
    plt.show()
    return pca


def projection_dataframe(
    cloud: TokenCloud,
    method: str = "pca",
    view: str = "raw",
    max_points: int = 2500,
    seed: int = 72,
) -> pd.DataFrame:
    idx = sample_indices(len(cloud.tokens), max_points, seed=seed + hash((method, view)) % 1000)
    x = cloud.tokens[idx].astype(np.float32)
    if view == "unit":
        x = l2_normalize(x)
    elif view == "whitened":
        x, _ = pca_whiten(x, max_dim=min(16, x.shape[1]), seed=seed)

    if method == "pca":
        xy = PCA(n_components=2, random_state=seed).fit_transform(x)
    elif method == "tsne":
        pre = PCA(n_components=min(12, x.shape[1], x.shape[0] - 1), random_state=seed).fit_transform(x)
        perplexity = max(5, min(30, (len(pre) - 1) // 3))
        xy = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(pre)
    elif method == "spectral":
        pre = PCA(n_components=min(12, x.shape[1], x.shape[0] - 1), random_state=seed).fit_transform(x)
        n_neighbors = max(5, min(20, len(pre) // 8))
        xy = SpectralEmbedding(n_components=2, n_neighbors=n_neighbors, random_state=seed).fit_transform(pre)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    meta = cloud.token_metadata.iloc[idx].reset_index(drop=True)
    norms = np.linalg.norm(cloud.tokens[idx], axis=1)
    return pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "norm": norms,
            "image_id": meta["image_id"].astype(str),
            "label": meta["label"].astype(str),
            "h": meta["h"].astype(int),
            "w": meta["w"].astype(int),
            "method": method,
            "view": view,
        }
    )


def plot_projection_grid(cloud: TokenCloud, methods: list[str], views: list[str], max_points: int = 2000, seed: int = 72) -> None:
    fig, axes = plt.subplots(len(views), len(methods), figsize=(4.6 * len(methods), 4.2 * len(views)))
    axes = np.asarray(axes).reshape(len(views), len(methods))
    for row, view in enumerate(views):
        for col, method in enumerate(methods):
            df = projection_dataframe(cloud, method=method, view=view, max_points=max_points, seed=seed)
            sc = axes[row, col].scatter(df["x"], df["y"], c=df["norm"], s=7, alpha=0.65, cmap="viridis")
            axes[row, col].set_title(f"{method.upper()} / {view} / color=norm")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(sc, ax=axes[row, col], fraction=0.046, pad=0.03)
    plt.tight_layout()
    plt.show()


def plot_projection_colorings(cloud: TokenCloud, method: str = "pca", view: str = "raw", max_points: int = 2500, seed: int = 72) -> None:
    df = projection_dataframe(cloud, method=method, view=view, max_points=max_points, seed=seed)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    sns.scatterplot(data=df, x="x", y="y", hue="label", s=10, alpha=0.65, ax=axes[0, 0])
    axes[0, 0].set_title("color = label")
    sns.scatterplot(data=df, x="x", y="y", hue="image_id", s=10, alpha=0.65, ax=axes[0, 1], legend=False)
    axes[0, 1].set_title("color = image id")
    sc = axes[0, 2].scatter(df["x"], df["y"], c=df["norm"], s=10, alpha=0.65, cmap="viridis")
    axes[0, 2].set_title("color = token norm")
    plt.colorbar(sc, ax=axes[0, 2], fraction=0.046, pad=0.03)
    sc = axes[1, 0].scatter(df["x"], df["y"], c=df["h"], s=10, alpha=0.65, cmap="magma")
    axes[1, 0].set_title("color = latent row")
    plt.colorbar(sc, ax=axes[1, 0], fraction=0.046, pad=0.03)
    sc = axes[1, 1].scatter(df["x"], df["y"], c=df["w"], s=10, alpha=0.65, cmap="magma")
    axes[1, 1].set_title("color = latent column")
    plt.colorbar(sc, ax=axes[1, 1], fraction=0.046, pad=0.03)
    axes[1, 2].hist2d(df["x"], df["y"], bins=60, cmap="mako")
    axes[1, 2].set_title("2D density")
    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{cloud.name}: {method.upper()} projection of {view} tokens", y=1.01)
    plt.tight_layout()
    plt.show()


def distance_effect_dataframe(cloud: TokenCloud, max_points: int = 800, seed: int = 72) -> pd.DataFrame:
    idx = sample_indices(len(cloud.tokens), max_points, seed=seed)
    raw = cloud.tokens[idx]
    unit = l2_normalize(raw)
    white, _ = pca_whiten(raw, max_dim=raw.shape[1], seed=seed)
    distances = {
        "raw_euclidean": pdist(raw, metric="euclidean"),
        "unit_euclidean": pdist(unit, metric="euclidean"),
        "whitened_euclidean": pdist(white, metric="euclidean"),
        "raw_cosine": pdist(raw, metric="cosine"),
    }
    rows = []
    keys = list(distances)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            rows.append({"a": a, "b": b, "spearman": float(spearmanr(distances[a], distances[b]).statistic)})
    return pd.DataFrame(rows), distances


def plot_distance_effects(cloud: TokenCloud, max_points: int = 800, seed: int = 72) -> None:
    corr_df, distances = distance_effect_dataframe(cloud, max_points=max_points, seed=seed)
    display(corr_df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, values in distances.items():
        sns.kdeplot(values, ax=axes[0], label=key)
    axes[0].set_title("pairwise distance distributions")
    axes[0].legend(fontsize=8)
    axes[1].scatter(distances["raw_euclidean"], distances["unit_euclidean"], s=3, alpha=0.25)
    axes[1].set_xlabel("raw Euclidean")
    axes[1].set_ylabel("unit-normalized Euclidean")
    axes[1].set_title("how unit normalization rewrites distances")
    plt.tight_layout()
    plt.show()


def spatial_offset_profile(cloud: TokenCloud, max_offset: int = 8) -> pd.DataFrame:
    z = latent_tensor(cloud)
    rows = []
    for dy in range(0, max_offset + 1):
        for dx in range(0, max_offset + 1):
            if dy == 0 and dx == 0:
                continue
            a = z[:, : z.shape[1] - dy or None, : z.shape[2] - dx or None, :]
            b = z[:, dy:, dx:, :]
            h = min(a.shape[1], b.shape[1])
            w = min(a.shape[2], b.shape[2])
            aa = a[:, :h, :w, :].reshape(-1, z.shape[-1])
            bb = b[:, :h, :w, :].reshape(-1, z.shape[-1])
            denom = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
            cos = np.sum(aa * bb, axis=1) / np.maximum(denom, 1e-8)
            rows.append({"dx": dx, "dy": dy, "manhattan": dx + dy, "mean_cosine": float(np.mean(cos)), "std_cosine": float(np.std(cos))})
    return pd.DataFrame(rows)


def plot_spatial_offset_profile(cloud: TokenCloud, max_offset: int = 8) -> None:
    profile = spatial_offset_profile(cloud, max_offset=max_offset)
    pivot = profile.pivot(index="dy", columns="dx", values="mean_cosine")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.heatmap(pivot, cmap="magma", ax=axes[0])
    axes[0].set_title("mean cosine by spatial offset")
    sns.lineplot(data=profile, x="manhattan", y="mean_cosine", errorbar="sd", marker="o", ax=axes[1])
    axes[1].set_title("spatial autocorrelation decay")
    axes[1].set_xlabel("Manhattan offset in latent grid")
    plt.tight_layout()
    plt.show()


def nearest_neighbor_patch_indices(cloud: TokenCloud, query_index: int, k: int = 8, exclude_same_image: bool = True) -> np.ndarray:
    q = cloud.tokens[[query_index]]
    query_image = cloud.token_metadata.iloc[query_index]["image_id"]
    candidates = np.arange(len(cloud.tokens))
    if exclude_same_image:
        candidates = candidates[cloud.token_metadata.iloc[candidates]["image_id"].to_numpy() != query_image]
    distances = pairwise_distances(q, cloud.tokens[candidates], metric="euclidean")[0]
    order = np.argsort(distances)[:k]
    return candidates[order]


def show_neighbor_patch_retrieval(
    cloud: TokenCloud,
    images: list[Image.Image],
    query_indices: list[int],
    image_size: int = 256,
    k: int = 6,
) -> None:
    for query_index in query_indices:
        neighbors = nearest_neighbor_patch_indices(cloud, query_index, k=k, exclude_same_image=True)
        fig, axes = plt.subplots(1, k + 1, figsize=(2.0 * (k + 1), 2.35))
        axes[0].imshow(approximate_patch(cloud, images, query_index, image_size=image_size, context_cells=2))
        axes[0].set_title("query", fontsize=9)
        axes[0].axis("off")
        for ax, neighbor_index in zip(axes[1:], neighbors):
            ax.imshow(approximate_patch(cloud, images, int(neighbor_index), image_size=image_size, context_cells=2))
            row = cloud.token_metadata.iloc[int(neighbor_index)]
            ax.set_title(f"img {row['image_id']}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"Nearest latent neighbors for token {query_index}", y=1.05)
        plt.tight_layout()
        plt.show()


def interesting_query_indices(cloud: TokenCloud, max_points: int = 5000, seed: int = 72) -> list[int]:
    norms = np.linalg.norm(cloud.tokens, axis=1)
    idx = sample_indices(len(cloud.tokens), max_points, seed=seed)
    xs = cloud.tokens[idx]
    kth = NearestNeighbors(n_neighbors=min(16, len(xs))).fit(xs).kneighbors(xs)[0][:, -1]
    return [
        int(np.argmax(norms)),
        int(np.argmin(norms)),
        int(idx[np.argmin(kth)]),
        int(idx[np.argmax(kth)]),
    ]


def image_level_summary(cloud: TokenCloud) -> pd.DataFrame:
    z = latent_tensor(cloud)
    rows = []
    for image_id in range(z.shape[0]):
        flat = z[image_id].reshape(-1, z.shape[-1])
        norms = np.linalg.norm(flat, axis=1)
        row = {
            "image_id": image_id,
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "max_norm": float(norms.max()),
        }
        for c in range(z.shape[-1]):
            row[f"mean_ch{c}"] = float(flat[:, c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_image_level_summary(summary: pd.DataFrame, metadata: pd.DataFrame) -> None:
    df = summary.merge(metadata[["image_id", "label"]], on="image_id", how="left")
    feature_cols = [c for c in df.columns if c.startswith("mean_ch")] + ["mean_norm", "std_norm", "max_norm"]
    n_components = min(2, len(df) - 1, len(feature_cols))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=df, x="label", y="mean_norm", ax=axes[0])
    sns.stripplot(data=df, x="label", y="mean_norm", color="black", alpha=0.5, ax=axes[0])
    axes[0].set_title("per-image mean latent norm")
    axes[0].tick_params(axis="x", rotation=25)
    if n_components >= 2:
        xy = PCA(n_components=2, random_state=72).fit_transform(StandardScaler().fit_transform(df[feature_cols]))
        plot_df = pd.DataFrame({"pc1": xy[:, 0], "pc2": xy[:, 1], "label": df["label"].astype(str), "image_id": df["image_id"].astype(str)})
        sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="label", style="label", s=80, ax=axes[1])
        for _, row in plot_df.iterrows():
            axes[1].text(row["pc1"], row["pc2"], row["image_id"], fontsize=8)
        axes[1].set_title("per-image latent summary PCA")
    else:
        axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def dense_token_indices(
    tokens: np.ndarray,
    n_dense: int = 1200,
    k: int = 16,
    max_candidates: int = 8000,
    seed: int = 72,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of tokens with smallest kth-neighbor distance."""
    candidate_idx = sample_indices(len(tokens), min(max_candidates, len(tokens)), seed=seed)
    x = tokens[candidate_idx]
    k_eff = min(k + 1, len(x))
    distances, _ = NearestNeighbors(n_neighbors=k_eff).fit(x).kneighbors(x)
    kth = distances[:, -1]
    order = np.argsort(kth)[: min(n_dense, len(candidate_idx))]
    return candidate_idx[order], kth[order]


def prepare_3d_view_data(
    cloud: TokenCloud,
    view: str,
    selection: str,
    max_points: int = 2500,
    n_dense: int = 1200,
    k_dense: int = 16,
    seed: int = 72,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Build token matrix and metadata for a 3D reduction condition.

    Views:
    - raw: original FLUX token vectors.
    - sphere: L2-normalized token directions.
    - whitened: PCA-whitened raw vectors.

    Selections:
    - all: random/all sample.
    - dense_raw: densest points selected in raw space.
    - dense_view: densest points selected after applying the view.
    """
    raw = cloud.tokens.astype(np.float32)
    if view == "raw":
        viewed = raw
        view_notes = {}
    elif view == "sphere":
        viewed = l2_normalize(raw).astype(np.float32)
        view_notes = {"view": "unit sphere directions"}
    elif view == "whitened":
        viewed, view_notes = pca_whiten(raw, max_dim=min(cloud.channel_dim, 16), seed=seed)
    else:
        raise ValueError(f"Unknown view: {view}")

    if selection == "all":
        idx = sample_indices(len(viewed), max_points, seed=seed)
        density_score = np.full(len(idx), np.nan)
    elif selection == "dense_raw":
        idx, density_score = dense_token_indices(raw, n_dense=min(n_dense, max_points), k=k_dense, seed=seed)
    elif selection == "dense_view":
        idx, density_score = dense_token_indices(viewed, n_dense=min(n_dense, max_points), k=k_dense, seed=seed)
    else:
        raise ValueError(f"Unknown selection: {selection}")

    x = viewed[idx]
    meta = cloud.token_metadata.iloc[idx].reset_index(drop=True)
    notes = {
        "view": view,
        "selection": selection,
        "n_points": len(idx),
        "k_dense": k_dense,
        **view_notes,
    }
    return x, idx, meta, {"density_score": density_score, **notes}


def reduce_3d(
    x: np.ndarray,
    method: str = "pca",
    seed: int = 72,
) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=3, random_state=seed).fit_transform(x)
    if method == "tsne":
        pre = PCA(n_components=min(12, x.shape[1], x.shape[0] - 1), random_state=seed).fit_transform(x)
        perplexity = max(5, min(30, (len(pre) - 1) // 3))
        return TSNE(n_components=3, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(pre)
    if method == "spectral":
        pre = PCA(n_components=min(12, x.shape[1], x.shape[0] - 1), random_state=seed).fit_transform(x)
        n_neighbors = max(5, min(20, len(pre) // 8))
        return SpectralEmbedding(n_components=3, n_neighbors=n_neighbors, random_state=seed).fit_transform(pre)
    raise ValueError(f"Unknown 3D method: {method}")


def make_3d_embedding_dataframe(
    cloud: TokenCloud,
    view: str,
    selection: str,
    method: str = "pca",
    max_points: int = 2500,
    n_dense: int = 1200,
    k_dense: int = 16,
    seed: int = 72,
) -> pd.DataFrame:
    x, idx, meta, notes = prepare_3d_view_data(
        cloud,
        view=view,
        selection=selection,
        max_points=max_points,
        n_dense=n_dense,
        k_dense=k_dense,
        seed=seed,
    )
    xyz = reduce_3d(x, method=method, seed=seed)
    raw_norm = np.linalg.norm(cloud.tokens[idx], axis=1)
    df = pd.DataFrame(
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "raw_norm": raw_norm,
            "image_id": meta["image_id"].astype(str),
            "label": meta["label"].astype(str),
            "h": meta["h"].astype(int),
            "w": meta["w"].astype(int),
            "view": view,
            "selection": selection,
            "method": method,
        }
    )
    if len(notes["density_score"]) == len(df):
        df["density_score"] = notes["density_score"]
    return df


def plot_3d_static_grid(
    cloud: TokenCloud,
    conditions: list[tuple[str, str]],
    method: str = "pca",
    max_points: int = 1600,
    n_dense: int = 900,
    seed: int = 72,
    color_by: str = "raw_norm",
) -> None:
    fig = plt.figure(figsize=(5.0 * len(conditions), 4.6))
    for i, (view, selection) in enumerate(conditions, start=1):
        df = make_3d_embedding_dataframe(
            cloud,
            view=view,
            selection=selection,
            method=method,
            max_points=max_points,
            n_dense=n_dense,
            seed=seed + i,
        )
        ax = fig.add_subplot(1, len(conditions), i, projection="3d")
        colors = df[color_by] if color_by in df else df["raw_norm"]
        sc = ax.scatter(df["x"], df["y"], df["z"], c=colors, s=5, alpha=0.65, cmap="viridis")
        ax.set_title(f"{method.upper()} 3D\n{view} / {selection}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    plt.tight_layout()
    plt.show()


def plot_3d_plotly(
    cloud: TokenCloud,
    view: str,
    selection: str,
    method: str = "pca",
    max_points: int = 2500,
    n_dense: int = 1200,
    seed: int = 72,
    color: str = "raw_norm",
) -> Any:
    import plotly.express as px

    df = make_3d_embedding_dataframe(
        cloud,
        view=view,
        selection=selection,
        method=method,
        max_points=max_points,
        n_dense=n_dense,
        seed=seed,
    )
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=color if color in df.columns else "raw_norm",
        hover_data=["label", "image_id", "h", "w", "view", "selection"],
        title=f"FLUX {method.upper()} 3D: {view} / {selection}",
    )
    fig.update_traces(marker=dict(size=2))
    return fig


def dense_selection_summary(
    cloud: TokenCloud,
    views: list[str] = ["raw", "sphere", "whitened"],
    n_dense: int = 1200,
    k_dense: int = 16,
    seed: int = 72,
) -> pd.DataFrame:
    rows = []
    raw_dense, _ = dense_token_indices(cloud.tokens, n_dense=n_dense, k=k_dense, seed=seed)
    raw_set = set(raw_dense.tolist())
    for view in views:
        _, idx, meta, notes = prepare_3d_view_data(
            cloud,
            view=view,
            selection="dense_view",
            max_points=n_dense,
            n_dense=n_dense,
            k_dense=k_dense,
            seed=seed,
        )
        idx_set = set(idx.tolist())
        label_counts = meta["label"].astype(str).value_counts(normalize=True).to_dict()
        rows.append(
            {
                "view_for_density": view,
                "n_dense": len(idx),
                "overlap_with_raw_dense": len(raw_set & idx_set) / max(1, len(idx_set)),
                "mean_raw_norm": float(np.linalg.norm(cloud.tokens[idx], axis=1).mean()),
                "median_raw_norm": float(np.median(np.linalg.norm(cloud.tokens[idx], axis=1))),
                **{f"label_frac_{k}": v for k, v in label_counts.items()},
            }
        )
    return pd.DataFrame(rows)
