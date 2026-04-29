"""Build the natural image patch baseline notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "10_natural_patch_baseline.ipynb"


def md(source: str) -> dict:
    text = dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": "md-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    text = dedent(source).strip()
    return {
        "cell_type": "code",
        "id": "code-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """
        # 10 - Natural Image Patch Baseline

        This notebook builds a compact raw-image-patch baseline for the beans
        images. The framing is inspired by the Carlsson natural-image-patch
        line of work: before asking whether learned latent tokens have
        interesting topology, ask what is already visible in local,
        contrast-normalized image patches.

        The central object here is the raw patch cloud. A FLUX VAE token cloud
        can be added as a side-by-side comparison when the model weights and
        runtime are available, but the readout stays cautious and patch-first.
        """
    ),
    md(
        """
        ## What This Baseline Tests

        The default pipeline is intentionally small:

        ```text
        beans images
          -> 16x16 RGB patches on a 256x256 center crop
          -> per-patch mean subtraction and L2 contrast normalization
          -> unit-sphere view
          -> dense kth-neighbor subset
          -> farthest-point landmarks
          -> Vietoris-Rips persistent homology
          -> simple null and synthetic controls
          -> landmark patches mapped back to source images
        ```

        This is not a reproduction of a natural-image-patch theorem or a claim
        about a canonical shape. It is a sanity baseline for the exact images,
        preprocessing, and TDA settings used in this project.
        """
    ),
    code(
        """
        from __future__ import annotations

        import math
        import os
        from pathlib import Path
        import sys
        import time
        import warnings

        for candidate in [Path.cwd(), Path.cwd().parent]:
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        from IPython.display import display
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from scipy.spatial.distance import pdist
        from sklearn.decomposition import PCA

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="The input point cloud has more columns than rows.*")
        plt.rcParams["figure.max_open_warning"] = 120
        sns.set_theme(style="whitegrid", context="notebook")

        from notebook_utils.encoder_explorer import (
            DEFAULT_IMAGE_DIR,
            center_crop_resize,
            choose_device,
            extract_token_clouds,
            l2_normalize,
            load_project_images,
            safe_hist,
            seed_everything,
            shape_summary,
            show_image_grid,
            token_norm_table,
        )
        from notebook_utils.flux_tda import (
            TDASample,
            build_channel_shuffle_dense_sample,
            build_observed_tda_sample,
            build_random_sphere_sample,
            build_uniform_sphere_sample,
            plot_betti_curves,
            plot_diagrams,
            plot_landmark_distance_matrix,
            plot_landmark_patches,
            plot_pipeline_projection,
            ripser_diagrams,
            top_persistence_table,
        )
        """
    ),
    md(
        """
        ## 1. Runtime Knobs

        Defaults are the requested compact run: 12 images, 16x16 raw patches,
        600 dense candidates, 95 landmarks, and `maxdim=2`. Set
        `TOKENIZER_SMOKE=1` for a short raw-only execution check.
        """
    ),
    code(
        """
        def parse_int_list(text: str) -> list[int]:
            return [int(part.strip()) for part in text.split(",") if part.strip()]


        SEED = int(os.environ.get("TOKENIZER_NOTEBOOK_SEED", "72"))
        SMOKE = os.environ.get("TOKENIZER_SMOKE", "0") == "1"

        RAW_PATCH_SIZE = 16
        IMAGE_SIZE = int(os.environ.get("TOKENIZER_PATCH_IMAGE_SIZE", "256"))
        N_IMAGES = int(os.environ.get("TOKENIZER_N_IMAGES", "4" if SMOKE else "12"))
        BATCH_SIZE = int(os.environ.get("TOKENIZER_BATCH_SIZE", "2" if SMOKE else "4"))
        IMAGE_DIR = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))

        COMPARISON_SEEDS = parse_int_list(os.environ.get("TOKENIZER_PATCH_SEEDS", "72"))
        N_DENSE = int(os.environ.get("TOKENIZER_PATCH_N_DENSE", "120" if SMOKE else "600"))
        N_LANDMARKS = int(os.environ.get("TOKENIZER_PATCH_N_LANDMARKS", "35" if SMOKE else "95"))
        K_DENSITY = int(os.environ.get("TOKENIZER_PATCH_K_DENSITY", "8" if SMOKE else "16"))
        MAXDIM = int(os.environ.get("TOKENIZER_PATCH_MAXDIM", "1" if SMOKE else "2"))
        DISTANCE_QUANTILE = float(os.environ.get("TOKENIZER_PATCH_DISTANCE_QUANTILE", "0.82"))
        PCA_POINTS = int(os.environ.get("TOKENIZER_PATCH_PCA_POINTS", "1200" if SMOKE else "4500"))

        default_compare_flux = "0" if SMOKE else "1"
        COMPARE_FLUX = os.environ.get("TOKENIZER_PATCH_COMPARE_FLUX", default_compare_flux) == "1"

        if IMAGE_SIZE % RAW_PATCH_SIZE != 0:
            raise ValueError(f"IMAGE_SIZE must be divisible by {RAW_PATCH_SIZE}; got {IMAGE_SIZE}")

        seed_everything(SEED)
        DEVICE = choose_device(force_cpu=os.environ.get("TOKENIZER_FORCE_CPU", "0") == "1")

        display(
            pd.DataFrame(
                [
                    {"knob": "seed", "value": SEED},
                    {"knob": "smoke", "value": SMOKE},
                    {"knob": "device", "value": DEVICE},
                    {"knob": "n_images", "value": N_IMAGES},
                    {"knob": "image_size", "value": IMAGE_SIZE},
                    {"knob": "raw_patch_size", "value": RAW_PATCH_SIZE},
                    {"knob": "raw_patch_grid", "value": (IMAGE_SIZE // RAW_PATCH_SIZE, IMAGE_SIZE // RAW_PATCH_SIZE)},
                    {"knob": "seeds", "value": COMPARISON_SEEDS},
                    {"knob": "n_dense", "value": N_DENSE},
                    {"knob": "n_landmarks", "value": N_LANDMARKS},
                    {"knob": "k_density", "value": K_DENSITY},
                    {"knob": "maxdim", "value": MAXDIM},
                    {"knob": "distance_quantile", "value": DISTANCE_QUANTILE},
                    {"knob": "compare_flux", "value": COMPARE_FLUX},
                    {"knob": "image_dir", "value": IMAGE_DIR},
                ]
            )
        )
        """
    ),
    md(
        """
        ## 2. Load Images and Extract Patches

        Raw patches are extracted first so the baseline is always available.
        The optional FLUX comparison is attempted afterward and skipped if the
        model cannot be loaded.
        """
    ),
    code(
        """
        images, image_metadata = load_project_images(N_IMAGES, IMAGE_DIR)
        display(image_metadata.head(N_IMAGES))
        show_image_grid(images, image_metadata, n=min(12, len(images)), title="Beans images for the patch baseline")

        t0 = time.perf_counter()
        raw_clouds, raw_failures = extract_token_clouds(
            images,
            image_metadata,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            autoencoder_size=IMAGE_SIZE,
            selected=["raw_patches"],
        )
        encode_failures = [raw_failures]
        token_clouds = dict(raw_clouds)

        if "raw_patches" not in token_clouds:
            raise RuntimeError("raw_patches extraction failed; this notebook needs the raw patch baseline.")

        if COMPARE_FLUX:
            flux_clouds, flux_failures = extract_token_clouds(
                images,
                image_metadata,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                autoencoder_size=IMAGE_SIZE,
                selected=["flux_vae"],
            )
            token_clouds.update(flux_clouds)
            encode_failures.append(flux_failures)

        encode_seconds = time.perf_counter() - t0
        failures = pd.concat([df for df in encode_failures if not df.empty], ignore_index=True) if any(not df.empty for df in encode_failures) else pd.DataFrame()

        if not failures.empty:
            display(failures)

        print(f"encoding seconds: {encode_seconds:.2f}")
        display(shape_summary(token_clouds))
        display(token_norm_table(token_clouds).round(6))
        """
    ),
    md(
        """
        ## 3. Patch Preprocessing and Sanity Checks

        The raw patch extractor stores one vector per spatial cell after
        subtracting the patch mean and normalizing by the centered L2 norm. That
        removes absolute brightness and most raw contrast, which is exactly why
        the checks below matter before reading topology.
        """
    ),
    code(
        """
        raw_cloud = token_clouds["raw_patches"]

        display(
            pd.DataFrame(
                [
                    {"field": "model_id", "value": raw_cloud.model_id},
                    {"field": "family", "value": raw_cloud.family},
                    {"field": "token_kind", "value": raw_cloud.token_kind},
                    {"field": "tokens", "value": raw_cloud.tokens.shape},
                    {"field": "grid_shape", "value": raw_cloud.grid_shape},
                    {"field": "channel_dim", "value": raw_cloud.channel_dim},
                    {"field": "notes", "value": raw_cloud.notes},
                ]
            )
        )


        def raw_patch_preprocess_table(images, metadata: pd.DataFrame, image_size: int, patch_size: int) -> pd.DataFrame:
            rows = []
            for image_id, img in enumerate(images):
                arr = np.asarray(center_crop_resize(img, image_size), dtype=np.float32) / 255.0
                label = metadata.iloc[image_id].get("label")
                for hh, yy in enumerate(range(0, image_size, patch_size)):
                    for ww, xx in enumerate(range(0, image_size, patch_size)):
                        patch = arr[yy : yy + patch_size, xx : xx + patch_size, :]
                        vec = patch.reshape(-1)
                        centered = vec - vec.mean()
                        rows.append(
                            {
                                "image_id": image_id,
                                "label": label,
                                "h": hh,
                                "w": ww,
                                "raw_mean": float(vec.mean()),
                                "raw_std": float(vec.std()),
                                "raw_min": float(vec.min()),
                                "raw_max": float(vec.max()),
                                "centered_l2": float(np.linalg.norm(centered)),
                            }
                        )
            return pd.DataFrame(rows)


        raw_pre = raw_patch_preprocess_table(images, image_metadata, IMAGE_SIZE, RAW_PATCH_SIZE)
        if len(raw_pre) != len(raw_cloud.tokens):
            raise RuntimeError(f"preprocess table/token mismatch: {len(raw_pre)} vs {len(raw_cloud.tokens)}")

        raw_pre["stored_norm"] = np.linalg.norm(raw_cloud.tokens, axis=1)
        raw_pre["stored_mean"] = raw_cloud.tokens.mean(axis=1)
        raw_pre["stored_abs_mean"] = np.abs(raw_cloud.tokens).mean(axis=1)
        raw_pre["is_nearly_flat"] = raw_pre["centered_l2"] < 1e-8

        display(raw_pre[["raw_mean", "raw_std", "centered_l2", "stored_norm", "stored_mean", "stored_abs_mean"]].describe().T.round(6))
        display(
            raw_pre.groupby("label", dropna=False)
            .agg(
                patches=("raw_std", "size"),
                raw_std_mean=("raw_std", "mean"),
                raw_std_p05=("raw_std", lambda x: float(np.quantile(x, 0.05))),
                centered_l2_mean=("centered_l2", "mean"),
                stored_norm_mean=("stored_norm", "mean"),
                nearly_flat=("is_nearly_flat", "sum"),
            )
            .round(6)
        )

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
        axes = axes.reshape(-1)
        safe_hist(axes[0], raw_pre["raw_std"].to_numpy(), bins=45, alpha=0.85)
        axes[0].set_title("raw per-patch RGB std")
        safe_hist(axes[1], raw_pre["centered_l2"].to_numpy(), bins=45, alpha=0.85)
        axes[1].set_title("centered patch L2 before normalization")
        safe_hist(axes[2], raw_pre["stored_norm"].to_numpy(), bins=45, alpha=0.85)
        axes[2].set_title("stored token L2 norm")
        safe_hist(axes[3], raw_pre["stored_mean"].to_numpy(), bins=45, alpha=0.85)
        axes[3].set_title("stored token mean")
        for ax in axes:
            ax.set_ylabel("count")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 4. Dense Sphere Landmark Sampling

        The TDA object is not every patch. We project the patch vectors to the
        unit sphere, keep a dense subset by small kth-neighbor distance, and
        choose farthest-point landmarks inside that dense subset.
        """
    ),
    code(
        """
        raw_sample, raw_dense_table = build_observed_tda_sample(
            raw_cloud,
            n_dense=N_DENSE,
            n_landmarks=N_LANDMARKS,
            k_density=K_DENSITY,
            seed=COMPARISON_SEEDS[0],
        )
        raw_sample.name = "raw_patches: observed_dense"

        display(pd.DataFrame([raw_sample.notes]))
        display(raw_dense_table.head(12))
        display(
            raw_dense_table.groupby(["image_id", "label"], dropna=False)
            .agg(
                dense_tokens=("source_index", "size"),
                landmarks=("is_landmark", "sum"),
                kth_mean=("kth_distance", "mean"),
                kth_min=("kth_distance", "min"),
                kth_max=("kth_distance", "max"),
            )
            .round(5)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        safe_hist(ax, raw_dense_table["kth_distance"].to_numpy(), bins=40, alpha=0.85)
        ax.set_title("dense subset kth-neighbor distances")
        ax.set_xlabel("kth-neighbor distance on the unit sphere")
        ax.set_ylabel("dense patches")
        plt.tight_layout()
        plt.show()

        plot_pipeline_projection(raw_cloud, raw_dense_table, raw_sample, max_points=PCA_POINTS, seed=SEED)


        def plot_dense_landmark_overlay(cloud, dense_table: pd.DataFrame, sample: TDASample, images, image_ids: list[int]) -> None:
            cols = max(1, len(image_ids))
            fig, axes = plt.subplots(1, cols, figsize=(3.25 * cols, 3.45))
            axes = np.asarray([axes]).reshape(-1)
            h, w = cloud.grid_shape
            cell_w = IMAGE_SIZE / w
            cell_h = IMAGE_SIZE / h
            landmark_meta = sample.metadata.copy()
            for ax, image_id in zip(axes, image_ids):
                ax.imshow(center_crop_resize(images[image_id], IMAGE_SIZE))
                dense_rows = dense_table[dense_table["image_id"] == image_id]
                if len(dense_rows):
                    ax.scatter(
                        (dense_rows["w"].to_numpy() + 0.5) * cell_w,
                        (dense_rows["h"].to_numpy() + 0.5) * cell_h,
                        s=9,
                        c="white",
                        alpha=0.45,
                        linewidths=0,
                    )
                landmark_rows = landmark_meta[landmark_meta["image_id"] == image_id]
                if len(landmark_rows):
                    ax.scatter(
                        (landmark_rows["w"].to_numpy() + 0.5) * cell_w,
                        (landmark_rows["h"].to_numpy() + 0.5) * cell_h,
                        s=32,
                        facecolors="none",
                        edgecolors="crimson",
                        linewidths=1.25,
                    )
                label = image_metadata.iloc[image_id].get("label")
                ax.set_title(f"image {image_id}: {label}", fontsize=9)
                ax.axis("off")
            fig.suptitle("dense raw patches (white) and landmarks (red)", y=1.02)
            plt.tight_layout()
            plt.show()


        image_ids = raw_dense_table["image_id"].value_counts().head(min(4, N_IMAGES)).index.astype(int).tolist()
        plot_dense_landmark_overlay(raw_cloud, raw_dense_table, raw_sample, images, image_ids)
        """
    ),
    md(
        """
        ## 5. Persistent Homology Setup and Controls

        Controls are deliberately simple and cheap:

        - `observed_dense`: dense sphere landmarks from the real cloud;
        - `random_tokens`: random sphere-normalized tokens from the same cloud;
        - `uniform_sphere`: random directions in the same ambient dimension;
        - `channel_shuffle`: per-channel shuffling before density selection;
        - `diag_gaussian`: independent Gaussian channels matched to raw means
          and standard deviations, then sphere-projected;
        - `synthetic_circle`: a positive-control loop on a low-dimensional
          sphere, included so the pipeline has a known H1-like target.
        """
    ),
    code(
        """
        CONTROL_KINDS = (
            ["observed_dense", "uniform_sphere"]
            if SMOKE
            else ["observed_dense", "random_tokens", "uniform_sphere", "channel_shuffle", "diag_gaussian"]
        )


        def build_diag_gaussian_sphere_sample(cloud, n_landmarks: int, seed: int) -> TDASample:
            rng = np.random.default_rng(seed)
            raw = cloud.tokens.astype(np.float32)
            mean = raw.mean(axis=0)
            std = np.maximum(raw.std(axis=0), 1e-6)
            x = rng.normal(loc=mean, scale=std, size=(n_landmarks, raw.shape[1])).astype(np.float32)
            x = l2_normalize(x).astype(np.float32)
            return TDASample(
                name=f"{cloud.name}: diag_gaussian",
                tokens=x,
                source_indices=np.arange(n_landmarks),
                metadata=pd.DataFrame({"image_id": ["control"] * n_landmarks, "label": ["diag_gaussian"] * n_landmarks, "h": 0, "w": 0}),
                notes={"control": "independent Gaussian matched to per-channel raw means/stds, then sphere-projected"},
            )


        def build_noisy_circle_sample(n_landmarks: int, seed: int, noise: float = 0.035) -> TDASample:
            rng = np.random.default_rng(seed)
            theta = np.linspace(0.0, 2.0 * np.pi, n_landmarks, endpoint=False)
            theta = theta + rng.normal(scale=0.025, size=n_landmarks)
            x = np.column_stack(
                [
                    np.cos(theta),
                    np.sin(theta),
                    noise * rng.normal(size=n_landmarks),
                ]
            ).astype(np.float32)
            x = l2_normalize(x).astype(np.float32)
            return TDASample(
                name="synthetic: noisy_circle",
                tokens=x,
                source_indices=np.arange(n_landmarks),
                metadata=pd.DataFrame({"image_id": ["synthetic"] * n_landmarks, "label": ["noisy_circle"] * n_landmarks, "h": 0, "w": 0}),
                notes={"control": "noisy circle on a 3D unit sphere"},
            )


        def pairwise_stats(sample: TDASample) -> dict[str, float]:
            distances = pdist(sample.tokens, metric="euclidean")
            if not len(distances):
                return {"pairwise_mean": 0.0, "pairwise_std": 0.0, "pairwise_cv": 0.0}
            mean = float(distances.mean())
            std = float(distances.std())
            return {"pairwise_mean": mean, "pairwise_std": std, "pairwise_cv": std / mean if mean > 1e-12 else 0.0}


        def summarize_result(result: dict, *, cloud_name: str, family: str, channel_dim: int, seed: int, sample_kind: str, sample: TDASample) -> pd.DataFrame:
            pstats = pairwise_stats(sample)
            rows = []
            for dim, diagram in enumerate(result["diagrams"]):
                finite = diagram[np.isfinite(diagram[:, 1])]
                persistence = finite[:, 1] - finite[:, 0] if len(finite) else np.array([])
                rows.append(
                    {
                        "cloud": cloud_name,
                        "family": family,
                        "channel_dim": channel_dim,
                        "seed": seed,
                        "sample_kind": sample_kind,
                        "dim": dim,
                        "n_points": len(sample.tokens),
                        "n_features": len(diagram),
                        "n_finite": len(finite),
                        "max_persistence": float(persistence.max()) if len(persistence) else 0.0,
                        "mean_persistence": float(persistence.mean()) if len(persistence) else 0.0,
                        "total_persistence": float(persistence.sum()) if len(persistence) else 0.0,
                        "top3_persistence": float(np.sort(persistence)[-3:].sum()) if len(persistence) else 0.0,
                        "threshold": result["threshold"],
                        "max_persistence_fraction": float(persistence.max() / result["threshold"]) if len(persistence) and result["threshold"] > 0 else 0.0,
                        **pstats,
                    }
                )
            return pd.DataFrame(rows)


        def samples_for_cloud(cloud, seed: int) -> tuple[list[tuple[str, TDASample]], pd.DataFrame | None]:
            samples = []
            dense_table = None
            if "observed_dense" in CONTROL_KINDS:
                observed, dense_table = build_observed_tda_sample(
                    cloud,
                    n_dense=N_DENSE,
                    n_landmarks=N_LANDMARKS,
                    k_density=K_DENSITY,
                    seed=seed,
                )
                observed.name = f"{cloud.name}: observed_dense"
                samples.append(("observed_dense", observed))
            if "random_tokens" in CONTROL_KINDS:
                random_sample = build_random_sphere_sample(cloud, n_landmarks=N_LANDMARKS, seed=seed + 101)
                random_sample.name = f"{cloud.name}: random_tokens"
                samples.append(("random_tokens", random_sample))
            if "uniform_sphere" in CONTROL_KINDS:
                uniform_sample = build_uniform_sphere_sample(cloud.channel_dim, n_landmarks=N_LANDMARKS, seed=seed + 102)
                uniform_sample.name = f"{cloud.name}: uniform_sphere"
                samples.append(("uniform_sphere", uniform_sample))
            if "channel_shuffle" in CONTROL_KINDS:
                shuffle_sample = build_channel_shuffle_dense_sample(
                    cloud,
                    n_dense=N_DENSE,
                    n_landmarks=N_LANDMARKS,
                    k_density=K_DENSITY,
                    seed=seed + 103,
                )
                shuffle_sample.name = f"{cloud.name}: channel_shuffle"
                samples.append(("channel_shuffle", shuffle_sample))
            if "diag_gaussian" in CONTROL_KINDS:
                samples.append(("diag_gaussian", build_diag_gaussian_sphere_sample(cloud, n_landmarks=N_LANDMARKS, seed=seed + 104)))
            return samples, dense_table


        display(pd.DataFrame({"control_kind": CONTROL_KINDS}))
        """
    ),
    md(
        """
        ## 6. Run Persistent Homology

        The same Rips settings are used for raw patches and, when available,
        FLUX latent tokens. Normalized persistence divides by the selected
        filtration threshold so clouds with different ambient dimensions are a
        little easier to compare.
        """
    ),
    code(
        """
        tda_cloud_order = ["raw_patches"] + [name for name in ["flux_vae"] if name in token_clouds]
        results: dict[tuple[str, int, str], dict] = {}
        sample_registry: dict[tuple[str, int, str], TDASample] = {}
        dense_tables: dict[tuple[str, int], pd.DataFrame] = {}
        summaries = []

        t0 = time.perf_counter()
        for cloud_name in tda_cloud_order:
            cloud = token_clouds[cloud_name]
            for seed in COMPARISON_SEEDS:
                samples, dense_table = samples_for_cloud(cloud, seed)
                if dense_table is not None:
                    dense_tables[(cloud_name, seed)] = dense_table
                for sample_kind, sample in samples:
                    print(f"ripser: {cloud_name:12s} {sample_kind:16s} seed={seed} n={len(sample.tokens)} dim={sample.tokens.shape[1]}")
                    result = ripser_diagrams(sample, maxdim=MAXDIM, distance_quantile=DISTANCE_QUANTILE)
                    result["sample"] = sample.name
                    key = (cloud_name, seed, sample_kind)
                    results[key] = result
                    sample_registry[key] = sample
                    summaries.append(
                        summarize_result(
                            result,
                            cloud_name=cloud_name,
                            family=cloud.family,
                            channel_dim=cloud.channel_dim,
                            seed=seed,
                            sample_kind=sample_kind,
                            sample=sample,
                        )
                    )

        for seed in COMPARISON_SEEDS:
            circle = build_noisy_circle_sample(N_LANDMARKS, seed=seed + 1001)
            print(f"ripser: {'synthetic':12s} {'noisy_circle':16s} seed={seed} n={len(circle.tokens)} dim={circle.tokens.shape[1]}")
            result = ripser_diagrams(circle, maxdim=MAXDIM, distance_quantile=DISTANCE_QUANTILE)
            result["sample"] = circle.name
            key = ("synthetic_control", seed, "noisy_circle")
            results[key] = result
            sample_registry[key] = circle
            summaries.append(
                summarize_result(
                    result,
                    cloud_name="synthetic_control",
                    family="Synthetic",
                    channel_dim=circle.tokens.shape[1],
                    seed=seed,
                    sample_kind="noisy_circle",
                    sample=circle,
                )
            )

        tda_seconds = time.perf_counter() - t0
        summary = pd.concat(summaries, ignore_index=True)

        print(f"TDA seconds: {tda_seconds:.2f}")
        print(f"diagram runs: {len(results)}")
        display(summary.head(12))
        display(summary.groupby(["cloud", "sample_kind", "dim"]).size().rename("runs").reset_index())
        """
    ),
    md(
        """
        ## 7. Observed Versus Controls

        The raw-patch signal is interesting only if it separates from the null
        controls and behaves sensibly relative to the synthetic positive
        control.
        """
    ),
    code(
        """
        topology_dims = [dim for dim in [1, 2] if dim <= MAXDIM]
        if not topology_dims:
            topology_dims = [0]

        display(
            summary[summary["dim"].isin(topology_dims)]
            .groupby(["cloud", "family", "sample_kind", "dim"])
            [["max_persistence_fraction", "top3_persistence", "n_finite", "pairwise_cv"]]
            .agg(["mean", "std"])
            .round(4)
        )

        plot_df = summary[summary["dim"].isin(topology_dims)].copy()
        g = sns.catplot(
            data=plot_df,
            x="sample_kind",
            y="max_persistence_fraction",
            hue="cloud",
            col="dim",
            kind="bar",
            height=4.2,
            aspect=1.15,
            sharey=False,
        )
        g.set_xticklabels(rotation=25, horizontalalignment="right")
        g.fig.suptitle("Normalized persistence: raw patches, controls, and optional FLUX", y=1.06)
        plt.show()

        means = (
            summary[summary["dim"].isin(topology_dims)]
            .groupby(["cloud", "family", "sample_kind", "dim"], as_index=False)["max_persistence_fraction"]
            .mean()
        )
        observed_means = means[means["sample_kind"] == "observed_dense"].rename(columns={"max_persistence_fraction": "observed_mean"})
        uniform_means = (
            means[means["sample_kind"] == "uniform_sphere"][["cloud", "dim", "max_persistence_fraction"]]
            .rename(columns={"max_persistence_fraction": "uniform_mean"})
        )
        random_means = (
            means[means["sample_kind"] == "random_tokens"][["cloud", "dim", "max_persistence_fraction"]]
            .rename(columns={"max_persistence_fraction": "random_token_mean"})
        )
        shuffle_means = (
            means[means["sample_kind"] == "channel_shuffle"][["cloud", "dim", "max_persistence_fraction"]]
            .rename(columns={"max_persistence_fraction": "channel_shuffle_mean"})
        )
        effect = (
            observed_means
            .merge(uniform_means, on=["cloud", "dim"], how="left")
            .merge(random_means, on=["cloud", "dim"], how="left")
            .merge(shuffle_means, on=["cloud", "dim"], how="left")
        )
        effect["delta_vs_uniform"] = effect["observed_mean"] - effect["uniform_mean"]
        effect["delta_vs_random_tokens"] = effect["observed_mean"] - effect["random_token_mean"]
        effect["delta_vs_channel_shuffle"] = effect["observed_mean"] - effect["channel_shuffle_mean"]
        effect = effect.sort_values(["dim", "cloud"])
        display(effect.round(4))
        """
    ),
    md(
        """
        ## 8. Representative Diagrams and Back-Mapped Patches

        The patch panels below show raw-patch landmarks, not certified cycle
        supports. They are still useful because a topology claim should be able
        to point back to visible image content.
        """
    ),
    code(
        """
        representative_seed = COMPARISON_SEEDS[0]
        diagram_keys = [
            ("raw_patches", representative_seed, "observed_dense"),
            ("raw_patches", representative_seed, "uniform_sphere"),
            ("raw_patches", representative_seed, "channel_shuffle"),
            ("flux_vae", representative_seed, "observed_dense"),
        ]
        representative_results = [results[key] for key in diagram_keys if key in results]
        if representative_results:
            plot_diagrams(representative_results)
            plot_betti_curves(representative_results)

        top_tables = []
        for key, result in results.items():
            cloud_name, seed, sample_kind = key
            if seed == representative_seed and (cloud_name == "raw_patches" or sample_kind == "noisy_circle"):
                top_tables.append(top_persistence_table(result, top_n=5).assign(cloud=cloud_name, sample_kind=sample_kind))
        if top_tables:
            display(pd.concat(top_tables, ignore_index=True)[["cloud", "sample_kind", "sample", "dim", "rank", "birth", "death", "persistence", "threshold"]].round(4))

        raw_observed_key = ("raw_patches", representative_seed, "observed_dense")
        raw_observed_sample = sample_registry[raw_observed_key]
        landmark_table = raw_observed_sample.metadata.copy()
        landmark_table["source_index"] = raw_observed_sample.source_indices
        landmark_xy = PCA(n_components=2, random_state=SEED).fit_transform(raw_observed_sample.tokens)
        landmark_table["landmark_pc1"] = landmark_xy[:, 0]
        landmark_table["landmark_pc2"] = landmark_xy[:, 1]
        display(landmark_table.head(12))

        plot_landmark_distance_matrix(raw_observed_sample)
        plot_landmark_patches(raw_observed_sample, raw_cloud, images, image_size=IMAGE_SIZE, n=min(18, len(raw_observed_sample.source_indices)))
        """
    ),
    md(
        """
        ## 9. Cautious Readout

        The goal is to decide whether raw patches are a meaningful baseline for
        later latent-token claims. A strong result would be stable across
        seeds, above null controls, visible in representative diagrams, and
        interpretable after patch back-mapping.
        """
    ),
    code(
        """
        print("Runtime recap")
        print(f"  encode seconds: {encode_seconds:.2f}")
        print(f"  TDA seconds:    {tda_seconds:.2f}")
        print(f"  diagram runs:   {len(results)}")
        print()

        if "effect" in globals() and not effect.empty:
            readout_cols = [
                "cloud",
                "family",
                "dim",
                "observed_mean",
                "uniform_mean",
                "random_token_mean",
                "channel_shuffle_mean",
                "delta_vs_uniform",
                "delta_vs_random_tokens",
                "delta_vs_channel_shuffle",
            ]
            available_cols = [col for col in readout_cols if col in effect.columns]
            display(effect[available_cols].round(4))

        raw_h1 = summary[(summary["cloud"] == "raw_patches") & (summary["sample_kind"] == "observed_dense") & (summary["dim"] == 1)]
        if len(raw_h1):
            best = raw_h1.sort_values("max_persistence_fraction", ascending=False).iloc[0]
            print(
                "Raw-patch H1 headline: "
                f"max normalized persistence {best['max_persistence_fraction']:.4f} "
                f"at threshold {best['threshold']:.4f} with {int(best['n_finite'])} finite H1 bars."
            )

        if "flux_vae" in token_clouds:
            print("FLUX comparison was available in this run.")
        else:
            print("FLUX comparison was not available in this run; read this as a raw-patch-only baseline.")

        print()
        print("Caveats")
        print("  1. Patch centering and L2 normalization remove absolute luminance and most contrast.")
        print("  2. These are landmarks, not recovered cycles; back-mapped patches are interpretive context.")
        print("  3. Twelve images and 95 landmarks are a compact baseline, not a final stability study.")
        print("  4. Treat any raw-patch feature inside the null-control range as preprocessing or finite-sample geometry.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
