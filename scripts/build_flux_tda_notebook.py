"""Build the fast FLUX latent-token TDA exploration notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "06_flux_tda_exploration.ipynb"


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
        # 06 - Fast TDA Probe of FLUX Latent Tokens

        This notebook is the first topology pass on the FLUX VAE latent-token
        cloud. It follows the pipeline sketched in `project_note.md`, but keeps
        the computation deliberately small:

        ```text
        images
          -> FLUX VAE encoder
          -> spatial latent tokens in R^16
          -> project token directions onto S^15
          -> estimate local density by k-nearest-neighbor radius
          -> keep a dense subset
          -> choose farthest-point landmarks
          -> run Vietoris-Rips persistent homology with maxdim=2
          -> compare against simple controls
          -> map landmarks back to image patches
        ```

        The goal is not to declare "the topology of FLUX." The goal is to learn
        what a concrete, reproducible TDA pipeline does to this encoder-induced
        empirical distribution.
        """
    ),
    md(
        """
        ## Reading Guide

        The main object here is a point cloud of local latent tokens. For a
        256x256 image, the FLUX VAE gives a 32x32 grid of 16-dimensional vectors.
        We flatten those grid locations across images into one point cloud.

        Why normalize to a sphere? This follows the project note's candidate
        object: token directions on `S^15`. It removes latent norm information,
        so it is a modeling choice. That is why this notebook also keeps plots of
        the raw norms and uses controls.

        Why dense points? Classical image-patch TDA often focuses on high-density
        regions after careful preprocessing. Sparse outliers can dominate Rips
        complexes. Here density means: a point is dense if its `k`th nearest
        neighbor is close after sphere projection.

        Why landmarks? Full Rips complexes get expensive quickly. We first keep
        dense candidates, then choose landmarks by farthest-point sampling inside
        that dense set. This preserves broad coverage while keeping runtime low.
        """
    ),
    code(
        """
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

        warnings.filterwarnings("ignore", category=FutureWarning)
        plt.rcParams["figure.max_open_warning"] = 120
        sns.set_theme(style="whitegrid", context="notebook")

        from notebook_utils.encoder_explorer import (
            DEFAULT_IMAGE_DIR,
            choose_device,
            extract_token_clouds,
            load_project_images,
            plot_norm_distributions,
            seed_everything,
            shape_summary,
            show_image_grid,
        )
        from notebook_utils.flux_deep_dive import (
            plot_latent_norm_and_rgb_pca_maps,
            plot_original_reconstruction,
            plot_projection_colorings,
            projection_dataframe,
        )
        from notebook_utils.flux_tda import (
            build_channel_shuffle_dense_sample,
            build_matched_gaussian_sphere_sample,
            build_observed_tda_sample,
            build_random_sphere_sample,
            build_uniform_sphere_sample,
            diagram_summary,
            plot_barcode,
            plot_betti_curves,
            plot_dense_maps,
            plot_diagrams,
            plot_filtration_graph_snapshots,
            plot_landmark_distance_matrix,
            plot_landmark_patches,
            plot_pairwise_distance_hist,
            plot_persistence_lifetimes,
            plot_persistence_summary,
            plot_pipeline_projection,
            ripser_diagrams,
            top_persistence_table,
        )
        """
    ),
    md(
        """
        ## 1. Runtime Knobs

        These defaults are chosen so the notebook stays in the 1-2 minute range
        once model weights are cached. If you want a stronger but slower run, the
        first knobs to raise are `N_IMAGES`, `N_DENSE`, and `N_LANDMARKS`.
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

        N_DENSE = int(os.environ.get("FLUX_TDA_N_DENSE", "300" if SMOKE else "600"))
        N_LANDMARKS = int(os.environ.get("FLUX_TDA_N_LANDMARKS", "60" if SMOKE else "95"))
        K_DENSITY = int(os.environ.get("FLUX_TDA_K_DENSITY", "10" if SMOKE else "16"))
        MAXDIM = int(os.environ.get("FLUX_TDA_MAXDIM", "2"))
        DISTANCE_QUANTILE = float(os.environ.get("FLUX_TDA_DISTANCE_QUANTILE", "0.82"))

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
                {"knob": "image_dir", "value": IMAGE_DIR},
            ]
        )
        display(config)
        """
    ),
    md(
        """
        ## 2. Load the Downloaded Images

        We use the local bean-leaf images downloaded earlier. The labels are
        useful as a loose visual reference, but the TDA object is the local
        latent-token distribution, not an image classifier.
        """
    ),
    code(
        """
        images, image_metadata = load_project_images(N_IMAGES, IMAGE_DIR)
        display(image_metadata.head(12))
        show_image_grid(images, image_metadata, n=min(12, len(images)), title="Images encoded in this run")
        """
    ),
    md(
        """
        ## 3. Encode with the FLUX VAE

        Only the autoencoder runs here. No diffusion denoising or image
        generation is involved. The token cloud below is the posterior mean of
        the FLUX AutoencoderKL encoder.
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
        flux = token_clouds["flux_vae"]
        print(f"encoding seconds: {encode_seconds:.2f}")
        display(shape_summary(token_clouds))
        display(pd.DataFrame([flux.notes]))
        """
    ),
    md(
        """
        ## 4. Before Topology: What Are We Topologizing?

        This sanity block keeps the geometric object honest. The norm maps show
        where latent scale is large or small. The PCA-color maps show coarse local
        variation in the 16-channel field. The reconstruction check reminds us
        that these vectors are decoder-facing latents, not generic embeddings.
        """
    ),
    code(
        """
        plot_original_reconstruction(images, flux, image_size=AUTOENCODER_SIZE, max_images=min(4, len(images)))
        for image_id in range(min(4, len(images))):
            plot_latent_norm_and_rgb_pca_maps(flux, image_id=image_id)
        plot_norm_distributions({"flux_vae": flux})
        """
    ),
    md(
        """
        ## 5. The Sphere Projection

        Persistent homology depends on the object and metric. Here we deliberately
        switch from raw `R^16` vectors to unit directions on `S^15`, then measure
        Euclidean chord distance. This is close to cosine geometry:

        ```text
        ||u - v||^2 = 2 - 2 cos(theta)
        ```

        The next plot is only a 2D PCA view of the sphere-projected tokens. It is
        not the TDA input itself, but it makes the sampling stages visible.
        """
    ),
    code(
        """
        projection = projection_dataframe(flux, method="pca", view="unit", max_points=min(2500, len(flux.tokens)), seed=SEED)
        display(projection.head())
        plot_projection_colorings(flux, method="pca", view="unit", max_points=min(2500, len(flux.tokens)), seed=SEED)
        """
    ),
    md(
        """
        ## 6. Dense Subset + Landmarks

        We now build the main TDA sample:

        1. Normalize every token to unit length.
        2. Estimate local density with the `k`th-nearest-neighbor distance.
        3. Keep the points with the smallest such distances.
        4. Pick farthest-point landmarks inside that dense set.

        This mirrors the high-density image-patch idea, but with FLUX latent
        tokens instead of raw pixel patches.
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
        assert observed.tokens.shape[0] <= N_LANDMARKS
        assert observed.tokens.shape[1] == flux.channel_dim
        """
    ),
    code(
        """
        plot_pipeline_projection(flux, dense_table, observed, max_points=min(3000, len(flux.tokens)), seed=SEED)
        plot_dense_maps(flux, dense_table, image_ids=list(range(min(4, len(images)))))
        plot_landmark_patches(observed, flux, images, image_size=AUTOENCODER_SIZE, n=min(18, len(observed.source_indices)))
        """
    ),
    md(
        """
        ## 7. Controls

        A persistence diagram becomes interesting only relative to something.
        These controls are intentionally simple:

        - random sphere tokens from the same FLUX cloud;
        - uniform random points on `S^15`;
        - Gaussian samples matched to raw token mean/covariance, then projected;
        - channel-shuffled tokens, then the same sphere+density+landmark pipeline.

        They do not prove anything by themselves, but they help separate effects
        of the sampling pipeline from effects specific to the observed encoder
        cloud.
        """
    ),
    code(
        """
        random_sample = build_random_sphere_sample(flux, n_landmarks=N_LANDMARKS, seed=SEED + 1)
        uniform_sample = build_uniform_sphere_sample(flux.channel_dim, n_landmarks=N_LANDMARKS, seed=SEED + 2)
        gaussian_sample = build_matched_gaussian_sphere_sample(flux, n_landmarks=N_LANDMARKS, seed=SEED + 3)
        shuffled_sample = build_channel_shuffle_dense_sample(
            flux,
            n_dense=N_DENSE,
            n_landmarks=N_LANDMARKS,
            k_density=K_DENSITY,
            seed=SEED + 4,
        )

        samples = [observed, random_sample, uniform_sample, gaussian_sample, shuffled_sample]
        sample_table = pd.DataFrame(
            [
                {
                    "sample": sample.name,
                    "n_points": len(sample.tokens),
                    "dim": sample.tokens.shape[1],
                    "notes": sample.notes,
                }
                for sample in samples
            ]
        )
        display(sample_table)
        plot_pairwise_distance_hist(samples)
        plot_landmark_distance_matrix(observed)
        """
    ),
    md(
        """
        ## 8. Run Persistent Homology

        We use `ripser` on landmark point clouds with `maxdim=2`. The filtration
        is capped at a high pairwise-distance quantile instead of running all the
        way to the largest possible simplex scale. This is a runtime compromise,
        and it is part of the experiment definition.

        Dimensions:

        - `H0`: connected components merging as scale grows;
        - `H1`: loops or cycle-like gaps;
        - `H2`: void-like cavities.
        """
    ),
    code(
        """
        t0 = time.perf_counter()
        results = []
        for sample in samples:
            print(f"ripser: {sample.name}  n={len(sample.tokens)}  dim={sample.tokens.shape[1]}")
            results.append(ripser_diagrams(sample, maxdim=MAXDIM, distance_quantile=DISTANCE_QUANTILE))
        tda_seconds = time.perf_counter() - t0
        print(f"TDA seconds: {tda_seconds:.2f}")

        summary = pd.concat([diagram_summary(result) for result in results], ignore_index=True)
        display(summary)
        assert set(summary["dim"]) == set(range(MAXDIM + 1))
        """
    ),
    md(
        """
        ## 9. Persistence Diagrams and Summary Plots

        These are the first "topology fingerprints." Look for differences between
        observed FLUX landmarks and controls, but keep the caveat in mind: a
        stronger claim needs repetitions across seeds, thresholds, data subsets,
        and encoders.
        """
    ),
    code(
        """
        plot_diagrams(results)
        plot_persistence_summary(summary)
        plot_persistence_lifetimes(results)
        plot_betti_curves(results)
        """
    ),
    md(
        """
        ## 10. Barcodes and Filtration Snapshots

        Persistence diagrams summarize many scale events at once. The barcode
        view shows the longest finite intervals directly. The graph snapshots
        show how the landmark cloud connects as the Rips scale increases.
        """
    ),
    code(
        """
        display(top_persistence_table(results[0], top_n=10))
        plot_barcode(results[0], max_bars_per_dim=40)
        plot_filtration_graph_snapshots(observed, results[0], seed=SEED)
        """
    ),
    md(
        """
        ## 11. Control Barcodes

        If a long bar appears in the observed sample but also appears in every
        control, it may be a generic artifact of small high-dimensional point
        clouds or the sphere projection. If it appears only in the observed dense
        FLUX sample, it becomes a better candidate for follow-up.
        """
    ),
    code(
        """
        for result in results[1:]:
            display(top_persistence_table(result, top_n=6))
            plot_barcode(result, max_bars_per_dim=30)
        """
    ),
    md(
        """
        ## 12. Back-Mapping: What Are the Landmarks?

        This is the interpretability loop from `project_note.md`: if a feature
        looks interesting, recover the image patches around the landmarks. This
        notebook does not yet extract representative cycles, but it does show the
        actual dense landmarks that generated the diagram.
        """
    ),
    code(
        """
        landmark_meta = observed.metadata.copy()
        landmark_meta["source_index"] = observed.source_indices
        landmark_meta["landmark_norm_raw"] = np.linalg.norm(flux.tokens[observed.source_indices], axis=1)
        display(landmark_meta.head(20))

        by_label = (
            landmark_meta["label"]
            .astype(str)
            .value_counts()
            .rename_axis("label")
            .reset_index(name="landmark_count")
        )
        display(by_label)
        plot_landmark_patches(observed, flux, images, image_size=AUTOENCODER_SIZE, n=min(24, len(observed.source_indices)))
        """
    ),
    md(
        """
        ## 13. What Would Make a Topological Feature Convincing?

        Treat this run as a fast probe. A meaningful follow-up should ask whether
        the same signature survives:

        - different image subsets;
        - different random seeds for density candidates and landmarks;
        - different `k`, dense-set size, and landmark count;
        - raw tokens versus sphere-projected tokens versus whitened tokens;
        - different datasets and different tokenizers;
        - a direct comparison to raw natural-image patches.

        The strongest loop is: find a stable feature, identify supporting
        landmarks or cycles, map them back to patches, and ask whether the patches
        vary along a coherent visual direction.
        """
    ),
    code(
        """
        print("Runtime recap")
        print(f"  encode seconds: {encode_seconds:.2f}")
        print(f"  TDA seconds:    {tda_seconds:.2f}")
        print(f"  total measured: {encode_seconds + tda_seconds:.2f}")
        print()
        print("Suggested quick reruns:")
        print("  FLUX_TDA_N_LANDMARKS=70 FLUX_TDA_N_DENSE=400")
        print("  FLUX_TDA_N_LANDMARKS=120 FLUX_TDA_N_DENSE=800")
        print("  FLUX_TDA_DISTANCE_QUANTILE=0.75")
        print("  TOKENIZER_N_IMAGES=24")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.12"},
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
