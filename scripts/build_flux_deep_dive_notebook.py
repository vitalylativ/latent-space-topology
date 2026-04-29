"""Build the FLUX encoder latent-space deep-dive notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "05_flux_encoder_deep_dive.ipynb"


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
        # 05 - FLUX Encoder Deep Dive

        This notebook zooms in on the FLUX VAE encoder.

        We are not doing topology yet. The goal is to build visual intuition for
        the empirical distribution:

        ```text
        bean leaf image -> FLUX VAE encoder -> latent tensor B x 16 x 32 x 32
        ```

        Each spatial location gives a 16-dimensional local latent token. We will
        visualize those tokens as images, distributions, point clouds, spatial
        fields, nearest-neighbor patches, and image-level summaries.
        """
    ),
    md(
        """
        ## What Is the FLUX VAE Encoder?

        FLUX uses a latent image representation. A VAE-style autoencoder compresses
        RGB images into a smaller spatial tensor. The diffusion model then works
        in this latent space rather than directly in pixel space.

        Important consequences:

        - FLUX VAE latents are **continuous**, not discrete token IDs.
        - The public Diffusers VAE has **16 latent channels**.
        - For 256x256 inputs, the latent grid is **32x32**, so each image produces
          1024 local token vectors.
        - Token norms may be meaningful because the decoder was trained to read
          both direction and scale.
        - Normalizing tokens onto a sphere is therefore an explicit modeling
          choice, not a neutral cleanup step.
        """
    ),
    code(
        """
        from pathlib import Path
        import os
        import sys

        for candidate in [Path.cwd(), Path.cwd().parent]:
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import seaborn as sns
        from sklearn.decomposition import PCA

        plt.rcParams["figure.max_open_warning"] = 100

        from notebook_utils.encoder_explorer import (
            DEFAULT_IMAGE_DIR,
            choose_device,
            extract_token_clouds,
            load_project_images,
            plot_norm_distributions,
            seed_everything,
            shape_summary,
            show_image_grid,
            show_representative_patches,
        )
        from notebook_utils.flux_deep_dive import (
            channel_summary,
            distance_effect_dataframe,
            image_level_summary,
            interesting_query_indices,
            latent_tensor,
            dense_selection_summary,
            make_3d_embedding_dataframe,
            plot_channel_correlation,
            plot_channel_distributions,
            plot_channel_summary,
            plot_3d_plotly,
            plot_3d_static_grid,
            plot_distance_effects,
            plot_image_level_summary,
            plot_latent_channel_maps,
            plot_latent_norm_and_rgb_pca_maps,
            plot_norm_maps_many,
            plot_original_reconstruction,
            plot_pca_spectrum_and_loadings,
            plot_projection_colorings,
            plot_projection_grid,
            plot_spatial_offset_profile,
            projection_dataframe,
            show_neighbor_patch_retrieval,
            spatial_offset_profile,
        )
        """
    ),
    code(
        """
        SEED = int(os.environ.get("TOKENIZER_NOTEBOOK_SEED", "72"))
        SMOKE = os.environ.get("TOKENIZER_SMOKE", "0") == "1"
        N_IMAGES = int(os.environ.get("TOKENIZER_N_IMAGES", "4" if SMOKE else "24"))
        BATCH_SIZE = int(os.environ.get("TOKENIZER_BATCH_SIZE", "2" if SMOKE else "4"))
        AUTOENCODER_SIZE = int(os.environ.get("TOKENIZER_AUTOENCODER_SIZE", "256"))
        MAX_POINTS = int(os.environ.get("TOKENIZER_MAX_METRIC_POINTS", "1200" if SMOKE else "5000"))
        PROJECTION_POINTS = int(os.environ.get("TOKENIZER_PROJECTION_POINTS", "900" if SMOKE else "2500"))
        IMAGE_DIR = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))

        seed_everything(SEED)
        DEVICE = choose_device(force_cpu=os.environ.get("TOKENIZER_FORCE_CPU", "0") == "1")

        print(f"device: {DEVICE}")
        print(f"images: {N_IMAGES}")
        print(f"image_dir: {IMAGE_DIR}")
        print(f"projection sample: {PROJECTION_POINTS}")
        """
    ),
    md(
        """
        ## 1. Load the Local Images

        We use the downloaded bean-leaf images. The labels are not the main point,
        but they let us ask whether FLUX local tokens separate disease classes or
        mostly encode local visual texture.
        """
    ),
    code(
        """
        images, image_metadata = load_project_images(N_IMAGES, IMAGE_DIR)
        display(image_metadata.head(12))
        show_image_grid(images, image_metadata, n=min(12, len(images)), title="Input images")
        """
    ),
    md(
        """
        ## 2. Encode Images with the FLUX VAE

        The helper runs only the VAE encoder/decoder. No diffusion generation is
        happening here. That keeps this notebook light enough to run locally.
        """
    ),
    code(
        """
        token_clouds, failures = extract_token_clouds(
            images,
            image_metadata,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            autoencoder_size=AUTOENCODER_SIZE,
            selected=["flux_vae"],
        )
        if not failures.empty:
            display(failures)
        flux = token_clouds["flux_vae"]
        display(shape_summary(token_clouds))
        display(pd.DataFrame([flux.notes]))
        """
    ),
    md(
        """
        ## 3. Reconstruction Sanity Check

        If the decoder can reconstruct the image, the latent tensor is not just an
        arbitrary feature vector. It is an image-compression representation that
        must preserve information useful for decoding.
        """
    ),
    code(
        """
        plot_original_reconstruction(images, flux, image_size=AUTOENCODER_SIZE, max_images=min(4, len(images)))
        """
    ),
    md(
        """
        ## 4. The Tensor as a Spatial Field

        The point-cloud view flattens all `(h, w)` locations into independent rows.
        But the latent tensor is still spatial: neighboring latent cells come from
        neighboring image regions and are strongly correlated.

        First, look at the raw 16 channels as images.
        """
    ),
    code(
        """
        image_id = 0
        show_image_grid([images[image_id]], image_metadata.iloc[[image_id]], n=1, title="Image used for channel maps")
        plot_latent_channel_maps(flux, image_id=image_id)
        """
    ),
    md(
        """
        ## 5. Norm Map and Latent-PCA RGB Map

        The norm map shows where the latent vector length is large. The RGB map
        compresses the 16-channel latent vector at each spatial location down to
        three local PCA coordinates, purely for visualization.
        """
    ),
    code(
        """
        plot_latent_norm_and_rgb_pca_maps(flux, image_id=image_id)
        """
    ),
    md(
        """
        ## 6. Norm Maps Across Several Images

        If norm is meaningful, it should have spatial structure. Look for whether
        high-norm regions align with leaf edges, disease spots, background, or
        other local patterns.
        """
    ),
    code(
        """
        image_ids = list(range(min(4, len(images))))
        plot_norm_maps_many(images, flux, image_ids=image_ids, image_size=AUTOENCODER_SIZE)
        """
    ),
    md(
        """
        ## 7. Token Norm Distribution

        This is the first warning against blindly using cosine geometry. If token
        lengths vary substantially, unit-normalization will rewrite distances.
        """
    ),
    code(
        """
        plot_norm_distributions({"flux_vae": flux})
        norms = np.linalg.norm(flux.tokens, axis=1)
        display(pd.Series(norms).describe().to_frame("token_norm"))
        """
    ),
    md(
        """
        ## 8. Per-Channel Statistics

        The 16 channels are not interchangeable white-noise coordinates. We inspect
        means, scales, skewness, and tails. Strongly non-Gaussian channels can have
        a large effect on density and nearest-neighbor structure.
        """
    ),
    code(
        """
        ch_summary = channel_summary(flux)
        display(ch_summary)
        plot_channel_summary(ch_summary)
        """
    ),
    md(
        """
        ## 9. Per-Channel Distributions

        Each subplot is one latent channel over all images and all spatial
        positions. These histograms help show whether channels are centered,
        heavy-tailed, skewed, or multi-modal.
        """
    ),
    code(
        """
        plot_channel_distributions(flux)
        """
    ),
    md(
        """
        ## 10. Channel Correlation and Covariance

        If channels are correlated, Euclidean distance in raw coordinates is
        anisotropic. Whitening is one way to remove that covariance structure, but
        it also changes the object being studied.
        """
    ),
    code(
        """
        plot_channel_correlation(flux)
        """
    ),
    md(
        """
        ## 11. PCA Spectrum and Loadings

        PCA tells us whether most token variance lies in a few directions. The
        loading heatmap shows which original latent channels contribute to the
        leading PCs.
        """
    ),
    code(
        """
        pca = plot_pca_spectrum_and_loadings(flux, max_points=MAX_POINTS, seed=SEED)
        print("cumulative variance after 4 PCs:", pca.explained_variance_ratio_[:4].sum())
        print("cumulative variance after 8 PCs:", pca.explained_variance_ratio_[:8].sum())
        """
    ),
    md(
        """
        ## 12. PCA Projection, Colored Several Ways

        The same 2D projection can tell different stories depending on color:

        - label color: disease class or image domain;
        - image id: per-image regimes;
        - norm: whether scale drives the projection;
        - row/column: spatial organization inside the latent grid;
        - density: crowded versus sparse regions.
        """
    ),
    code(
        """
        plot_projection_colorings(flux, method="pca", view="raw", max_points=PROJECTION_POINTS, seed=SEED)
        """
    ),
    md(
        """
        ## 13. Raw vs Unit-Normalized vs Whitened Views

        Here we visualize three different mathematical objects:

        - raw FLUX tokens;
        - unit-norm directions;
        - PCA-whitened tokens.

        If the plots change a lot, topology will also change a lot.
        """
    ),
    code(
        """
        methods = ["pca"] if SMOKE else ["pca", "tsne", "spectral"]
        plot_projection_grid(flux, methods=methods, views=["raw", "unit", "whitened"], max_points=PROJECTION_POINTS, seed=SEED)
        """
    ),
    md(
        """
        ## 14. Optional Nonlinear Views

        PCA is linear. t-SNE and spectral embedding are not topology either, but
        they can reveal whether local neighborhoods look clustered, filamentary, or
        mostly continuous. These views are qualitative only.
        """
    ),
    code(
        """
        if SMOKE:
            print("Smoke mode: using PCA only. Set TOKENIZER_SMOKE=0 for t-SNE and spectral views.")
        else:
            plot_projection_colorings(flux, method="tsne", view="raw", max_points=1800, seed=SEED)
            plot_projection_colorings(flux, method="spectral", view="raw", max_points=1800, seed=SEED)
        """
    ),
    md(
        """
        ## 15. 3D PCA Projection

        This is interactive in JupyterLab. It is useful when the first two PCs
        flatten a structure that is clearer in three dimensions.
        """
    ),
    code(
        """
        idx = np.arange(len(flux.tokens))
        if len(idx) > PROJECTION_POINTS:
            rng = np.random.default_rng(SEED)
            idx = np.sort(rng.choice(idx, size=PROJECTION_POINTS, replace=False))
        xyz = PCA(n_components=3, random_state=SEED).fit_transform(flux.tokens[idx])
        meta = flux.token_metadata.iloc[idx].reset_index(drop=True)
        fig3d = px.scatter_3d(
            pd.DataFrame({
                "pc1": xyz[:, 0],
                "pc2": xyz[:, 1],
                "pc3": xyz[:, 2],
                "norm": np.linalg.norm(flux.tokens[idx], axis=1),
                "label": meta["label"].astype(str),
                "image_id": meta["image_id"].astype(str),
            }),
            x="pc1",
            y="pc2",
            z="pc3",
            color="norm",
            hover_data=["label", "image_id"],
            title="FLUX token cloud: 3D PCA colored by norm",
        )
        fig3d.update_traces(marker=dict(size=2))
        fig3d.show()
        """
    ),
    md(
        """
        ## 16. Pairwise Distance Geometry

        This directly measures how much preprocessing changes distances. Low rank
        correlation between raw and unit-normalized distances means norm is a major
        geometric factor.
        """
    ),
    code(
        """
        plot_distance_effects(flux, max_points=350 if SMOKE else 800, seed=SEED)
        """
    ),
    md(
        """
        ## 17. 3D Reductions: Raw, Sphere, Dense, Dense-on-Sphere

        Now we explicitly compare the combinations you asked for:

        - **raw / all**: raw FLUX token vectors, sampled broadly;
        - **sphere / all**: every token projected to the unit sphere;
        - **raw / dense_raw**: densest tokens selected in raw Euclidean geometry;
        - **sphere / dense_view**: tokens first projected to the sphere, then
          density is computed on the sphere-projected cloud.

        These are not cosmetic variations. They are different empirical objects.
        """
    ),
    code(
        """
        base_3d_conditions = [
            ("raw", "all"),
            ("sphere", "all"),
            ("raw", "dense_raw"),
            ("sphere", "dense_view"),
        ]
        plot_3d_static_grid(
            flux,
            conditions=base_3d_conditions,
            method="pca",
            max_points=900 if SMOKE else 2200,
            n_dense=500 if SMOKE else 1200,
            seed=SEED,
        )
        """
    ),
    md(
        """
        ## 18. Dense Selection Depends on the Geometry

        "The densest points" is not a universal set. Densest in raw space can
        differ from densest after unit-sphere projection or whitening. This table
        measures that overlap.
        """
    ),
    code(
        """
        dense_summary = dense_selection_summary(
            flux,
            views=["raw", "sphere", "whitened"],
            n_dense=500 if SMOKE else 1200,
            k_dense=16,
            seed=SEED,
        )
        display(dense_summary)
        """
    ),
    md(
        """
        ## 19. More 3D PCA Combinations

        Here are the combinations in a compact grid:

        - only projection: `sphere / all`;
        - only dense: `raw / dense_raw`;
        - dense after projection: `sphere / dense_view`;
        - dense after whitening: `whitened / dense_view`;
        - all whitened tokens: `whitened / all`.
        """
    ),
    code(
        """
        more_3d_conditions = [
            ("raw", "all"),
            ("sphere", "all"),
            ("whitened", "all"),
            ("raw", "dense_raw"),
            ("sphere", "dense_view"),
            ("whitened", "dense_view"),
        ]
        plot_3d_static_grid(
            flux,
            conditions=more_3d_conditions,
            method="pca",
            max_points=800 if SMOKE else 1800,
            n_dense=450 if SMOKE else 1000,
            seed=SEED + 10,
        )
        """
    ),
    md(
        """
        ## 20. Interactive 3D Views

        These are Plotly figures. In JupyterLab you can rotate them. The first is
        all sphere-projected tokens; the second is densest sphere-projected tokens.
        """
    ),
    code(
        """
        fig_sphere_all = plot_3d_plotly(
            flux,
            view="sphere",
            selection="all",
            method="pca",
            max_points=900 if SMOKE else 2500,
            n_dense=500 if SMOKE else 1200,
            seed=SEED,
            color="raw_norm",
        )
        fig_sphere_all.show()

        fig_sphere_dense = plot_3d_plotly(
            flux,
            view="sphere",
            selection="dense_view",
            method="pca",
            max_points=900 if SMOKE else 2500,
            n_dense=500 if SMOKE else 1200,
            seed=SEED + 1,
            color="raw_norm",
        )
        fig_sphere_dense.show()
        """
    ),
    md(
        """
        ## 21. Optional Nonlinear 3D Reductions

        PCA is the fastest and most stable. t-SNE and spectral embeddings are more
        qualitative. They can help inspect local neighborhood structure, but they
        should not be treated as evidence of topology by themselves.
        """
    ),
    code(
        """
        if SMOKE:
            print("Smoke mode: skipping nonlinear 3D reductions. Set TOKENIZER_SMOKE=0 to run them.")
        else:
            nonlinear_conditions = [
                ("raw", "dense_raw"),
                ("sphere", "dense_view"),
            ]
            plot_3d_static_grid(
                flux,
                conditions=nonlinear_conditions,
                method="tsne",
                max_points=1200,
                n_dense=900,
                seed=SEED + 20,
            )
            plot_3d_static_grid(
                flux,
                conditions=nonlinear_conditions,
                method="spectral",
                max_points=1200,
                n_dense=900,
                seed=SEED + 30,
            )
        """
    ),
    md(
        """
        ## 22. Spatial Autocorrelation

        Adjacent latent tokens are not independent samples. This plot shows how
        cosine similarity decays as two latent cells move farther apart on the
        32x32 grid.
        """
    ),
    code(
        """
        plot_spatial_offset_profile(flux, max_offset=6 if SMOKE else 10)
        """
    ),
    md(
        """
        ## 23. Spatial Offset Table

        Sometimes the heatmap hides the numbers. This table shows the most local
        offsets explicitly.
        """
    ),
    code(
        """
        profile = spatial_offset_profile(flux, max_offset=4)
        display(profile.sort_values(["manhattan", "dy", "dx"]).head(20))
        """
    ),
    md(
        """
        ## 24. Representative Patches

        Back-mapping is critical for interpretation. These are approximate image
        patches corresponding to high-norm, low-norm, dense, and sparse latent
        tokens.
        """
    ),
    code(
        """
        show_representative_patches(flux, images, image_size=AUTOENCODER_SIZE)
        """
    ),
    md(
        """
        ## 25. Nearest-Neighbor Patch Retrieval

        Pick a token, find nearest FLUX tokens from other images, and inspect the
        corresponding source patches. This helps answer whether local latent
        neighborhoods correspond to coherent visual neighborhoods.
        """
    ),
    code(
        """
        queries = interesting_query_indices(flux, max_points=MAX_POINTS, seed=SEED)
        print("query token indices:", queries)
        show_neighbor_patch_retrieval(flux, images, queries, image_size=AUTOENCODER_SIZE, k=6 if not SMOKE else 4)
        """
    ),
    md(
        """
        ## 26. Image-Level Latent Summaries

        So far, each point has been a local token. We can also summarize each
        whole image by mean channel values and norm statistics. This is not the
        same object, but it tells us whether image labels or image identity show up
        at a coarser scale.
        """
    ),
    code(
        """
        img_summary = image_level_summary(flux)
        display(img_summary.head())
        plot_image_level_summary(img_summary, image_metadata)
        """
    ),
    md(
        """
        ## 27. What We Learned About FLUX Latents

        Use this notebook to decide what the next topology experiment should
        control for:

        - token norms are visible and spatially structured;
        - channels have different distributions and are correlated;
        - raw, unit-normalized, and whitened geometries are not identical;
        - spatial correlation reduces the effective sample size;
        - nearest-neighbor patches give an interpretability loop;
        - image-level summaries are a different object than local token clouds.

        A future persistence diagram should always name which of these choices it
        used.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (latent-space-topology)",
            "language": "python",
            "name": "latent-space-topology",
        },
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
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {NOTEBOOK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
