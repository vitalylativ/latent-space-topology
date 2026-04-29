"""Build the FLUX persistent-feature interpretation notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "09_interpreting_persistent_features.ipynb"


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
        # 09 - Interpreting Persistent Features

        The previous notebooks asked whether the FLUX VAE latent-token cloud
        produces persistent homology that is stronger than simple controls. This
        notebook asks a more visual question:

        > When a long `H1` or `H2` bar appears, which image patches sit near the
        > landmarks associated with that feature?

        The mapping is approximate. `ripser` can return representative
        co-cycles, not canonical geometric cycles. We use their supporting
        landmark indices as a practical handle for inspection, then map those
        landmarks back to the local image patches that produced the FLUX latent
        tokens.
        """
    ),
    md(
        """
        ## Interpretation Contract

        The code below is designed as a triage loop, not as a proof that a
        feature is semantically meaningful.

        ```text
        beans images
          -> FLUX VAE posterior-mean tokens
          -> unit-sphere token directions
          -> dense token subset
          -> farthest-point landmarks
          -> ripser diagrams with co-cycles
          -> top H1/H2 bars
          -> co-cycle support landmarks
          -> source patches and nearest-neighbor patches
        ```

        The cautious readout at the end matters: a co-cycle support can move
        under small perturbations, and several different supports can represent
        the same persistent class. Treat the patch panels as clues for follow-up,
        not as a literal picture of the topological feature.
        """
    ),
    code(
        """
        from __future__ import annotations

        from dataclasses import replace
        from pathlib import Path
        import os
        import sys
        import time
        import warnings

        for candidate in [Path.cwd(), Path.cwd().parent]:
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from scipy.spatial.distance import pdist, squareform
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="The input point cloud has more columns than rows.*")
        plt.rcParams["figure.max_open_warning"] = 120
        sns.set_theme(style="whitegrid", context="notebook")

        from notebook_utils.encoder_explorer import (
            DEFAULT_IMAGE_DIR,
            approximate_patch,
            choose_device,
            extract_token_clouds,
            l2_normalize,
            load_project_images,
            seed_everything,
            shape_summary,
            show_image_grid,
        )
        from notebook_utils.flux_tda import (
            TDASample,
            build_observed_tda_sample,
            build_random_sphere_sample,
            build_uniform_sphere_sample,
            diagram_summary,
            plot_barcode,
            plot_betti_curves,
            plot_dense_maps,
            plot_diagrams,
            plot_landmark_distance_matrix,
            plot_landmark_patches,
            plot_pairwise_distance_hist,
            plot_pipeline_projection,
            ripser_diagrams,
            top_persistence_table,
        )
        """
    ),
    md(
        """
        ## 1. Runtime Knobs

        The default run keeps the interpretation loop modest: 12 images, about
        600 dense candidate tokens, 95 landmarks, and persistent homology through
        `H2`. Set `TOKENIZER_SMOKE=1` for a quick test.
        """
    ),
    code(
        """
        SEED = int(os.environ.get("TOKENIZER_NOTEBOOK_SEED", "72"))
        SMOKE = os.environ.get("TOKENIZER_SMOKE", "0") == "1"

        N_IMAGES = int(os.environ.get("TOKENIZER_N_IMAGES", "4" if SMOKE else "12"))
        BATCH_SIZE = int(os.environ.get("TOKENIZER_BATCH_SIZE", "2" if SMOKE else "4"))
        AUTOENCODER_SIZE = int(os.environ.get("TOKENIZER_AUTOENCODER_SIZE", "256"))
        IMAGE_DIR = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))

        N_DENSE = int(os.environ.get("FLUX_INTERPRET_N_DENSE", "250" if SMOKE else "600"))
        N_LANDMARKS = int(os.environ.get("FLUX_INTERPRET_N_LANDMARKS", "45" if SMOKE else "95"))
        K_DENSITY = int(os.environ.get("FLUX_INTERPRET_K_DENSITY", "10" if SMOKE else "16"))
        MAXDIM = int(os.environ.get("FLUX_INTERPRET_MAXDIM", "2"))
        DISTANCE_QUANTILE = float(os.environ.get("FLUX_INTERPRET_DISTANCE_QUANTILE", "0.82"))

        TOP_FEATURES_PER_DIM = int(os.environ.get("FLUX_INTERPRET_TOP_FEATURES_PER_DIM", "2" if SMOKE else "3"))
        MAX_SUPPORT_PATCHES = int(os.environ.get("FLUX_INTERPRET_MAX_SUPPORT_PATCHES", "8" if SMOKE else "12"))
        NEIGHBORS_PER_ANCHOR = int(os.environ.get("FLUX_INTERPRET_NEIGHBORS_PER_ANCHOR", "4" if SMOKE else "6"))
        ANCHORS_PER_FEATURE = int(os.environ.get("FLUX_INTERPRET_ANCHORS_PER_FEATURE", "2" if SMOKE else "3"))

        seed_everything(SEED)
        DEVICE = choose_device(force_cpu=os.environ.get("TOKENIZER_FORCE_CPU", "0") == "1")

        config = pd.DataFrame(
            [
                {"knob": "device", "value": DEVICE},
                {"knob": "smoke", "value": SMOKE},
                {"knob": "n_images", "value": N_IMAGES},
                {"knob": "batch_size", "value": BATCH_SIZE},
                {"knob": "autoencoder_size", "value": AUTOENCODER_SIZE},
                {"knob": "n_dense", "value": N_DENSE},
                {"knob": "n_landmarks", "value": N_LANDMARKS},
                {"knob": "k_density", "value": K_DENSITY},
                {"knob": "maxdim", "value": MAXDIM},
                {"knob": "distance_quantile", "value": DISTANCE_QUANTILE},
                {"knob": "top_features_per_dim", "value": TOP_FEATURES_PER_DIM},
                {"knob": "max_support_patches", "value": MAX_SUPPORT_PATCHES},
                {"knob": "neighbors_per_anchor", "value": NEIGHBORS_PER_ANCHOR},
                {"knob": "anchors_per_feature", "value": ANCHORS_PER_FEATURE},
                {"knob": "image_dir", "value": IMAGE_DIR},
            ]
        )
        display(config)
        """
    ),
    md(
        """
        ## 2. Load the Beans Images

        Labels are only a visual aid here. The object we topologize is the local
        distribution of FLUX VAE latent tokens.
        """
    ),
    code(
        """
        images, image_metadata = load_project_images(N_IMAGES, IMAGE_DIR)
        display(image_metadata.head(12))
        show_image_grid(images, image_metadata, n=min(12, len(images)), title="Images encoded for feature interpretation")
        """
    ),
    md(
        """
        ## 3. Encode with the FLUX VAE

        We run only the VAE encoder. Each 256x256 image gives a 32x32 grid of
        16-dimensional posterior-mean tokens.
        """
    ),
    code(
        """
        t0 = time.perf_counter()
        token_clouds, failures = extract_token_clouds(
            images,
            image_metadata,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            autoencoder_size=AUTOENCODER_SIZE,
            selected=["flux_vae"],
        )
        encode_seconds = time.perf_counter() - t0

        if not failures.empty:
            display(failures)
        if "flux_vae" not in token_clouds:
            raise RuntimeError("FLUX VAE token cloud was not produced; inspect the failures table above.")

        flux = token_clouds["flux_vae"]
        print(f"encoding seconds: {encode_seconds:.2f}")
        display(shape_summary(token_clouds))
        display(pd.DataFrame([flux.notes]))
        """
    ),
    md(
        """
        ## 4. Build the Observed Dense-Sphere Landmark Sample

        This follows the same pipeline as notebooks 06-08: normalize token
        directions to the unit sphere, keep locally dense candidates by
        kth-nearest-neighbor distance, then choose farthest-point landmarks
        inside that dense set.
        """
    ),
    code(
        """
        observed, dense_table = build_observed_tda_sample(
            flux,
            n_dense=N_DENSE,
            n_landmarks=N_LANDMARKS,
            k_density=K_DENSITY,
            seed=SEED,
        )

        print(observed.notes)
        display(dense_table.head(12))
        display(
            dense_table["kth_distance"]
            .describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
            .to_frame("dense kth-neighbor distance")
        )
        display(
            observed.metadata["label"]
            .astype(str)
            .value_counts()
            .rename_axis("label")
            .reset_index(name="landmark_count")
        )
        """
    ),
    code(
        """
        plot_pipeline_projection(flux, dense_table, observed, max_points=min(3000, len(flux.tokens)), seed=SEED)
        plot_dense_maps(flux, dense_table, image_ids=list(range(min(4, len(images)))))
        plot_landmark_distance_matrix(observed)
        plot_landmark_patches(observed, flux, images, image_size=AUTOENCODER_SIZE, n=min(18, len(observed.source_indices)))
        """
    ),
    md(
        """
        ## 5. Co-Cycle-Aware Persistent Homology Helpers

        `notebook_utils.flux_tda.ripser_diagrams` is enough for diagrams. Here
        we add a local wrapper that requests co-cycles from `ripser` so that the
        longest bars can be connected back to landmark indices.
        """
    ),
    code(
        """
        def ripser_with_cocycles(sample: TDASample, maxdim: int = 2, distance_quantile: float = 0.82) -> dict:
            import ripser

            distances = pdist(sample.tokens, metric="euclidean")
            threshold = float(np.quantile(distances, distance_quantile))
            raw = ripser.ripser(sample.tokens, maxdim=maxdim, thresh=threshold, do_cocycles=True)
            return {
                "sample": sample.name,
                "diagrams": raw["dgms"],
                "cocycles": raw.get("cocycles", []),
                "threshold": threshold,
                "distance_quantile": distance_quantile,
                "n_points": len(sample.tokens),
                "maxdim": maxdim,
            }


        def top_feature_table(result: dict, top_n: int = 3, dims: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
            rows = []
            for dim in dims:
                if dim >= len(result["diagrams"]):
                    continue
                diagram = result["diagrams"][dim]
                finite_rows = []
                for feature_id, (birth, death) in enumerate(diagram):
                    if np.isfinite(death):
                        finite_rows.append((feature_id, float(birth), float(death), float(death - birth)))
                if not finite_rows:
                    continue
                finite_rows = sorted(finite_rows, key=lambda row: row[3], reverse=True)[:top_n]
                for rank, (feature_id, birth, death, persistence) in enumerate(finite_rows, start=1):
                    rows.append(
                        {
                            "sample": result["sample"],
                            "dim": dim,
                            "rank": rank,
                            "feature_id": feature_id,
                            "birth": birth,
                            "death": death,
                            "persistence": persistence,
                            "persistence_fraction": persistence / result["threshold"] if result["threshold"] > 0 else np.nan,
                            "threshold": result["threshold"],
                        }
                    )
            return pd.DataFrame(rows)


        def cocycle_vertex_scores(result: dict, dim: int, feature_id: int) -> pd.DataFrame:
            cocycles = result.get("cocycles", [])
            if dim >= len(cocycles) or feature_id >= len(cocycles[dim]):
                return pd.DataFrame(columns=["landmark_local", "support_score"])

            cocycle = np.asarray(cocycles[dim][feature_id])
            if cocycle.size == 0:
                return pd.DataFrame(columns=["landmark_local", "support_score"])

            vertex_cols = min(dim + 1, cocycle.shape[1] - 1)
            vertices = cocycle[:, :vertex_cols].astype(int).reshape(-1)
            counts = pd.Series(vertices).value_counts().rename_axis("landmark_local").reset_index(name="support_score")
            return counts.sort_values(["support_score", "landmark_local"], ascending=[False, True]).reset_index(drop=True)


        def feature_support_table(result: dict, sample: TDASample, feature_rows: pd.DataFrame, cloud) -> pd.DataFrame:
            rows = []
            raw_norms = np.linalg.norm(cloud.tokens[sample.source_indices], axis=1)
            for feature in feature_rows.to_dict("records"):
                scores = cocycle_vertex_scores(result, int(feature["dim"]), int(feature["feature_id"]))
                for support_rank, score_row in enumerate(scores.to_dict("records"), start=1):
                    local_idx = int(score_row["landmark_local"])
                    if local_idx >= len(sample.source_indices):
                        continue
                    meta = sample.metadata.iloc[local_idx].to_dict()
                    rows.append(
                        {
                            "feature_label": f"H{int(feature['dim'])}.{int(feature['rank'])}",
                            "dim": int(feature["dim"]),
                            "feature_rank": int(feature["rank"]),
                            "feature_id": int(feature["feature_id"]),
                            "birth": float(feature["birth"]),
                            "death": float(feature["death"]),
                            "persistence": float(feature["persistence"]),
                            "persistence_fraction": float(feature["persistence_fraction"]),
                            "support_rank": support_rank,
                            "landmark_local": local_idx,
                            "source_index": int(sample.source_indices[local_idx]),
                            "support_score": int(score_row["support_score"]),
                            "raw_token_norm": float(raw_norms[local_idx]),
                            **meta,
                        }
                    )
            return pd.DataFrame(rows)


        def summarize_feature_support(support: pd.DataFrame) -> pd.DataFrame:
            if support.empty:
                return pd.DataFrame()
            rows = []
            for label, group in support.groupby("feature_label", sort=False):
                top_labels = (
                    group["label"]
                    .astype(str)
                    .value_counts()
                    .head(3)
                    .rename_axis("label")
                    .reset_index(name="count")
                )
                top_label_text = ", ".join(f"{row.label}:{row.count}" for row in top_labels.itertuples(index=False))
                rows.append(
                    {
                        "feature_label": label,
                        "dim": int(group["dim"].iloc[0]),
                        "feature_rank": int(group["feature_rank"].iloc[0]),
                        "support_landmarks": len(group),
                        "unique_images": int(group["image_id"].nunique()),
                        "top_labels": top_label_text,
                        "median_support_score": float(group["support_score"].median()),
                        "median_raw_token_norm": float(group["raw_token_norm"].median()),
                        "persistence_fraction": float(group["persistence_fraction"].iloc[0]),
                    }
                )
            return pd.DataFrame(rows).sort_values(["dim", "feature_rank"]).reset_index(drop=True)
        """
    ),
    md(
        """
        ## 6. Run Persistent Homology

        The observed run requests co-cycles for interpretation. Two small
        controls are included only to keep the observed bars in context.
        """
    ),
    code(
        """
        random_sample = build_random_sphere_sample(flux, n_landmarks=N_LANDMARKS, seed=SEED + 101)
        uniform_sample = build_uniform_sphere_sample(flux.channel_dim, n_landmarks=N_LANDMARKS, seed=SEED + 102)
        samples = [observed, random_sample, uniform_sample]

        t0 = time.perf_counter()
        observed_result = ripser_with_cocycles(observed, maxdim=MAXDIM, distance_quantile=DISTANCE_QUANTILE)
        control_results = [
            ripser_diagrams(sample, maxdim=MAXDIM, distance_quantile=DISTANCE_QUANTILE)
            for sample in samples[1:]
        ]
        results = [observed_result, *control_results]
        tda_seconds = time.perf_counter() - t0

        summary = pd.concat([diagram_summary(result) for result in results], ignore_index=True)
        summary["max_persistence_fraction"] = summary["max_persistence"] / summary["threshold"]

        print(f"TDA seconds: {tda_seconds:.2f}")
        display(summary.round(4))
        plot_pairwise_distance_hist(samples)
        plot_diagrams(results)
        plot_betti_curves(results)
        """
    ),
    md(
        """
        ## 7. Longest H1/H2 Bars

        The table below preserves `feature_id`, the row index needed to recover
        the corresponding co-cycle from `ripser`.
        """
    ),
    code(
        """
        observed_top_features = top_feature_table(
            observed_result,
            top_n=TOP_FEATURES_PER_DIM,
            dims=tuple(dim for dim in (1, 2) if dim <= MAXDIM),
        )
        display(observed_top_features.round(4))

        print("Observed barcode")
        plot_barcode(observed_result, max_bars_per_dim=40)

        print("Control longest bars")
        control_top_tables = [top_persistence_table(result, top_n=5) for result in control_results]
        display(pd.concat(control_top_tables, ignore_index=True).round(4))
        """
    ),
    md(
        """
        ## 8. Co-Cycle Support Landmarks

        For each selected feature, we collect the landmark vertices that appear
        in the representative co-cycle. A high support score means that landmark
        appears in more co-cycle simplices; it does not mean the landmark is
        uniquely responsible for the topology.
        """
    ),
    code(
        """
        support = feature_support_table(observed_result, observed, observed_top_features, flux)
        support_summary = summarize_feature_support(support)

        display(support_summary.round(4))
        if support.empty:
            print("No co-cycle support landmarks were returned for the selected features.")
        else:
            display(
                support[
                    [
                        "feature_label",
                        "support_rank",
                        "support_score",
                        "image_id",
                        "label",
                        "h",
                        "w",
                        "source_index",
                        "landmark_local",
                        "raw_token_norm",
                    ]
                ]
                .head(40)
                .round(4)
            )
        """
    ),
    md(
        """
        ## 9. Patch Panels for Feature-Associated Landmarks

        These panels map the highest-scoring support landmarks back to
        approximate source patches. The crops include a little spatial context
        around each FLUX latent cell.
        """
    ),
    code(
        """
        def plot_feature_patch_panel(
            support: pd.DataFrame,
            feature_label: str,
            cloud,
            images,
            image_size: int = 256,
            max_patches: int = 12,
            context_cells: int = 2,
        ) -> None:
            feature_support = (
                support[support["feature_label"] == feature_label]
                .sort_values(["support_score", "support_rank"], ascending=[False, True])
                .head(max_patches)
            )
            if feature_support.empty:
                print(f"{feature_label}: no support patches to show")
                return

            n = len(feature_support)
            cols = min(4, n)
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(2.35 * cols, 2.55 * rows))
            axes = np.asarray([axes]).reshape(-1)

            for ax, row in zip(axes, feature_support.itertuples(index=False)):
                ax.imshow(
                    approximate_patch(
                        cloud,
                        images,
                        int(row.source_index),
                        image_size=image_size,
                        context_cells=context_cells,
                    )
                )
                title = f"{row.feature_label} s={int(row.support_score)}\\nimg {int(row.image_id)} ({int(row.h)},{int(row.w)})"
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            for ax in axes[n:]:
                ax.axis("off")

            fig.suptitle(f"Approximate source patches for {feature_label}", y=1.02)
            plt.tight_layout()
            plt.show()


        if support.empty:
            print("Skipping patch panels because no co-cycle support was available.")
        else:
            for feature_label in support_summary["feature_label"]:
                plot_feature_patch_panel(
                    support,
                    feature_label,
                    flux,
                    images,
                    image_size=AUTOENCODER_SIZE,
                    max_patches=MAX_SUPPORT_PATCHES,
                    context_cells=2,
                )
        """
    ),
    md(
        """
        ## 10. Nearest Neighbors Around Feature Landmarks

        A single support landmark can be a noisy representative. The next step
        asks what local FLUX-token neighborhood surrounds the strongest support
        points, using all sphere-projected tokens from the encoded images.
        """
    ),
    code(
        """
        def nearest_feature_neighbors(
            cloud,
            sample: TDASample,
            support: pd.DataFrame,
            neighbors_per_anchor: int = 6,
            anchors_per_feature: int = 3,
        ) -> pd.DataFrame:
            if support.empty:
                return pd.DataFrame()

            sphere = l2_normalize(cloud.tokens.astype(np.float32)).astype(np.float32)
            n_neighbors = min(neighbors_per_anchor + 1, len(sphere))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(sphere)

            anchors = (
                support.sort_values(["feature_label", "support_score", "support_rank"], ascending=[True, False, True])
                .groupby("feature_label", sort=False)
                .head(anchors_per_feature)
            )

            rows = []
            for anchor in anchors.itertuples(index=False):
                anchor_source = int(anchor.source_index)
                distances, indices = nbrs.kneighbors(sphere[anchor_source : anchor_source + 1])
                neighbor_rank = 0
                for distance, neighbor_idx in zip(distances[0], indices[0]):
                    neighbor_idx = int(neighbor_idx)
                    if neighbor_idx == anchor_source:
                        continue
                    neighbor_rank += 1
                    meta = cloud.token_metadata.iloc[neighbor_idx].to_dict()
                    rows.append(
                        {
                            "feature_label": anchor.feature_label,
                            "anchor_source_index": anchor_source,
                            "anchor_image_id": int(anchor.image_id),
                            "anchor_h": int(anchor.h),
                            "anchor_w": int(anchor.w),
                            "anchor_label": anchor.label,
                            "anchor_support_score": int(anchor.support_score),
                            "neighbor_rank": neighbor_rank,
                            "neighbor_source_index": neighbor_idx,
                            "distance": float(distance),
                            **{f"neighbor_{key}": value for key, value in meta.items()},
                        }
                    )
                    if neighbor_rank >= neighbors_per_anchor:
                        break
            return pd.DataFrame(rows)


        def plot_neighbor_patch_strips(
            neighbor_table: pd.DataFrame,
            cloud,
            images,
            image_size: int = 256,
            max_rows: int = 8,
            context_cells: int = 2,
        ) -> None:
            if neighbor_table.empty:
                print("No nearest-neighbor patches to show.")
                return

            anchor_rows = (
                neighbor_table[
                    [
                        "feature_label",
                        "anchor_source_index",
                        "anchor_image_id",
                        "anchor_h",
                        "anchor_w",
                        "anchor_label",
                        "anchor_support_score",
                    ]
                ]
                .drop_duplicates()
                .head(max_rows)
            )
            cols = min(NEIGHBORS_PER_ANCHOR + 1, 7)
            rows = len(anchor_rows)
            fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.25 * rows))
            axes = np.asarray([axes]).reshape(rows, cols)

            for row_i, anchor in enumerate(anchor_rows.itertuples(index=False)):
                anchor_source = int(anchor.anchor_source_index)
                axes[row_i, 0].imshow(
                    approximate_patch(
                        cloud,
                        images,
                        anchor_source,
                        image_size=image_size,
                        context_cells=context_cells,
                    )
                )
                axes[row_i, 0].set_title(
                    f"{anchor.feature_label}\\nanchor s={int(anchor.anchor_support_score)}",
                    fontsize=8,
                )
                axes[row_i, 0].axis("off")

                neighbors = (
                    neighbor_table[neighbor_table["anchor_source_index"] == anchor_source]
                    .sort_values("neighbor_rank")
                    .head(cols - 1)
                )
                for col_i, item in enumerate(neighbors.itertuples(index=False), start=1):
                    axes[row_i, col_i].imshow(
                        approximate_patch(
                            cloud,
                            images,
                            int(item.neighbor_source_index),
                            image_size=image_size,
                            context_cells=context_cells,
                        )
                    )
                    axes[row_i, col_i].set_title(f"nn {int(item.neighbor_rank)}\\nd={item.distance:.3f}", fontsize=8)
                    axes[row_i, col_i].axis("off")
                for col_i in range(1 + len(neighbors), cols):
                    axes[row_i, col_i].axis("off")

            fig.suptitle("Feature support anchors and nearest token neighbors", y=1.01)
            plt.tight_layout()
            plt.show()


        neighbor_table = nearest_feature_neighbors(
            flux,
            observed,
            support,
            neighbors_per_anchor=NEIGHBORS_PER_ANCHOR,
            anchors_per_feature=ANCHORS_PER_FEATURE,
        )
        if neighbor_table.empty:
            print("No nearest neighbors computed because no support anchors were available.")
        else:
            display(
                neighbor_table[
                    [
                        "feature_label",
                        "anchor_image_id",
                        "anchor_label",
                        "anchor_h",
                        "anchor_w",
                        "anchor_support_score",
                        "neighbor_rank",
                        "distance",
                        "neighbor_image_id",
                        "neighbor_label",
                        "neighbor_h",
                        "neighbor_w",
                        "neighbor_source_index",
                    ]
                ]
                .head(60)
                .round(4)
            )
            plot_neighbor_patch_strips(
                neighbor_table,
                flux,
                images,
                image_size=AUTOENCODER_SIZE,
                max_rows=min(8, ANCHORS_PER_FEATURE * max(1, len(support_summary))),
                context_cells=2,
            )
        """
    ),
    md(
        """
        ## 11. Landmark Geometry Around the Selected Features

        This is a small geometry sanity check: are the support landmarks tightly
        clustered, spread around the landmark cloud, or mostly coming from one
        image region?
        """
    ),
    code(
        """
        def support_distance_summary(sample: TDASample, support: pd.DataFrame) -> pd.DataFrame:
            if support.empty:
                return pd.DataFrame()
            dist = squareform(pdist(sample.tokens, metric="euclidean"))
            rows = []
            for label, group in support.groupby("feature_label", sort=False):
                locals_ = group["landmark_local"].to_numpy(dtype=int)
                if len(locals_) < 2:
                    pairwise = np.array([])
                else:
                    sub = dist[np.ix_(locals_, locals_)]
                    pairwise = sub[np.triu_indices_from(sub, k=1)]
                rows.append(
                    {
                        "feature_label": label,
                        "support_landmarks": len(locals_),
                        "mean_support_distance": float(pairwise.mean()) if len(pairwise) else np.nan,
                        "min_support_distance": float(pairwise.min()) if len(pairwise) else np.nan,
                        "max_support_distance": float(pairwise.max()) if len(pairwise) else np.nan,
                        "cloud_pairwise_mean": float(pdist(sample.tokens, metric="euclidean").mean()),
                        "unique_images": int(group["image_id"].nunique()),
                        "unique_grid_cells": int(group[["h", "w"]].drop_duplicates().shape[0]),
                    }
                )
            return pd.DataFrame(rows)


        distance_summary = support_distance_summary(observed, support)
        display(distance_summary.round(4))

        if not support.empty:
            xy = PCA(n_components=2, random_state=SEED).fit_transform(observed.tokens)
            plot_df = observed.metadata.copy()
            plot_df["pc1"] = xy[:, 0]
            plot_df["pc2"] = xy[:, 1]
            plot_df["landmark_local"] = np.arange(len(plot_df))
            plot_df["feature_label"] = "not selected"
            selected = support.drop_duplicates(["feature_label", "landmark_local"])
            for feature_label, rows in selected.groupby("feature_label"):
                plot_df.loc[plot_df["landmark_local"].isin(rows["landmark_local"]), "feature_label"] = feature_label

            fig, ax = plt.subplots(figsize=(7.2, 5.4))
            sns.scatterplot(
                data=plot_df,
                x="pc1",
                y="pc2",
                hue="feature_label",
                style="feature_label",
                s=42,
                alpha=0.85,
                ax=ax,
            )
            ax.set_title("Selected co-cycle support landmarks in a PCA view")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.show()
        """
    ),
    md(
        """
        ## 12. Cautious Readout

        The final cell prints a compact project-log readout. Use it to decide
        whether a feature deserves a stronger run, not to make a claim from this
        single notebook.
        """
    ),
    code(
        """
        print("Runtime recap")
        print(f"  encode seconds: {encode_seconds:.2f}")
        print(f"  TDA seconds:    {tda_seconds:.2f}")
        print(f"  total measured: {encode_seconds + tda_seconds:.2f}")
        print()

        if observed_top_features.empty:
            print("No finite H1/H2 bars were selected under the current settings.")
        else:
            print("Selected observed features")
            display(observed_top_features[["dim", "rank", "birth", "death", "persistence", "persistence_fraction"]].round(4))

        if not support_summary.empty:
            print("Co-cycle support summary")
            display(support_summary.round(4))

        control_higher = summary[(summary["sample"] != observed.name) & (summary["dim"].isin([1, 2]))]
        observed_higher = summary[(summary["sample"] == observed.name) & (summary["dim"].isin([1, 2]))]
        if not control_higher.empty and not observed_higher.empty:
            compare = observed_higher[["dim", "max_persistence_fraction"]].rename(columns={"max_persistence_fraction": "observed_max_fraction"})
            control = (
                control_higher
                .groupby("dim", as_index=False)["max_persistence_fraction"]
                .max()
                .rename(columns={"max_persistence_fraction": "best_control_max_fraction"})
            )
            compare = compare.merge(control, on="dim", how="left")
            compare["observed_minus_best_control"] = compare["observed_max_fraction"] - compare["best_control_max_fraction"]
            print("Observed-versus-control scale check")
            display(compare.round(4))

        print(
            "Interpretation prompt: patches attached to a co-cycle support are visual clues. "
            "A stronger claim would require repeated seeds, nearby parameter settings, and ideally "
            "a cycle-level reconstruction whose mapped patches vary coherently rather than merely sharing labels or positions."
        )
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
