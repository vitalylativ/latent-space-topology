"""Build the narrative geometry-before-topology notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "04_geometric_intuition_before_topology.ipynb"


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
        # 04 - Build Geometric Intuition Before Topology

        This notebook implements point 4 from `project_note.md`.

        The goal is to understand the point clouds before asking persistent
        homology to summarize them. A persistence diagram can look meaningful even
        when it is mostly an artifact of normalization, density filtering, or
        correlated samples. So here we inspect simpler geometry first.
        """
    ),
    md(
        """
        ## The Mental Model

        The object is not "the latent space" in the abstract. The object is a
        sampled empirical distribution:

        ```text
        images -> encoder -> spatial latent tensor -> local token vectors
        ```

        This notebook asks what those local token vectors look like under several
        views:

        - raw Euclidean vectors;
        - unit-normalized directions;
        - PCA-whitened vectors;
        - spatial fields rather than unordered bags.
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
        import seaborn as sns
        from sklearn.decomposition import PCA

        from notebook_utils.encoder_explorer import (
            DEFAULT_IMAGE_DIR,
            choose_device,
            distance_preprocessing_effects,
            extract_token_clouds,
            geometry_metrics,
            load_project_images,
            main_effect_table,
            make_cloud_views,
            plot_norm_distributions,
            plot_norm_maps,
            plot_pca_by_label,
            seed_everything,
            shape_summary,
            show_image_grid,
            show_representative_patches,
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
        VIT_SIZE = int(os.environ.get("TOKENIZER_VIT_SIZE", "224"))
        MAX_POINTS = int(os.environ.get("TOKENIZER_MAX_METRIC_POINTS", "1200" if SMOKE else "5000"))
        IMAGE_DIR = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))
        SELECTED = os.environ.get("TOKENIZER_ENCODERS")
        SELECTED = [x.strip() for x in SELECTED.split(",")] if SELECTED else None

        seed_everything(SEED)
        DEVICE = choose_device(force_cpu=os.environ.get("TOKENIZER_FORCE_CPU", "0") == "1")

        print(f"device: {DEVICE}")
        print(f"images: {N_IMAGES}")
        print(f"max sampled tokens per metric: {MAX_POINTS}")
        """
    ),
    md(
        """
        ## 1. Literature Review -> Geometry Diagnostics

        The literature review gives us a checklist:

        - **Natural image patches**: preprocessing is part of the mathematical
          object, not a cosmetic choice.
        - **Neural representation topology**: conclusions depend on what counts as
          a point.
        - **Generative latent geometry**: decoder-trained latents need not be
          isotropic Euclidean spaces.
        - **Image tokenizers**: continuous VAEs and discrete VQ codebooks have
          different geometry.
        - **Statistical PH**: density, outliers, and sampling change topology.

        So before topology we inspect norms, covariance/PCA, intrinsic dimension,
        nearest-neighbor density, spatial autocorrelation, preprocessing effects,
        and representative source patches.
        """
    ),
    code(
        """
        diagnostic_map = pd.DataFrame(
            [
                ("norms", "Are token lengths meaningful, or should directions be studied separately?"),
                ("PCA / covariance", "Is the cloud anisotropic or dominated by a few directions?"),
                ("intrinsic dimension", "How many local degrees of freedom does a view appear to use?"),
                ("nearest-neighbor density", "Are there dense regimes, sparse boundaries, or outliers?"),
                ("spatial autocorrelation", "How independent are adjacent spatial tokens?"),
                ("raw vs unit vs whitened", "How much does preprocessing rewrite the geometry?"),
                ("patch back-mapping", "Do extreme or representative tokens correspond to visible image patterns?"),
            ],
            columns=["diagnostic", "question"],
        )
        display(diagnostic_map)
        """
    ),
    md(
        """
        ## 2. Data: Use the Downloaded Images

        The notebook uses `data/images/beans` by default. This matters because
        later patch back-mapping only makes sense if we know exactly which source
        image and grid location each token came from.
        """
    ),
    code(
        """
        images, image_metadata = load_project_images(N_IMAGES, IMAGE_DIR)
        display(image_metadata.head(12))
        show_image_grid(images, image_metadata, n=min(8, len(images)), title="Images used for geometry exploration")
        """
    ),
    md(
        """
        ## 3. Extract Token Clouds

        We reuse the encoder set from the previous notebook:

        - FLUX VAE;
        - Stable Diffusion VAE;
        - Kandinsky MoVQ;
        - ViT patch encoder;
        - CLIP vision encoder;
        - raw centered/normalized patches.

        This cell only creates the objects. Interpretation comes afterward.
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
            vit_size=VIT_SIZE,
            selected=SELECTED,
        )

        display(shape_summary(token_clouds))
        if not failures.empty:
            display(failures)
        """
    ),
    md(
        """
        ## 4. First Question: Are Norms Part of the Signal?

        If we normalize every token to unit length, we throw away all norm
        information. That may be right for some questions, but it is not neutral.

        Read these plots as a warning label for future topology:

        - wide norm distributions mean spherical normalization changes the data;
        - constant norms mean the preprocessing already imposed a sphere;
        - spatial norm maps can reveal local image regions driving token scale.
        """
    ),
    code(
        """
        plot_norm_distributions(token_clouds)
        """
    ),
    code(
        """
        image_id = 0
        show_image_grid([images[image_id]], image_metadata.iloc[[image_id]], n=1, title="Image used for norm maps")
        plot_norm_maps(token_clouds, image_id=image_id)
        """
    ),
    md(
        """
        ## 5. Candidate Views of the Same Token Clouds

        For each encoder, we build three views:

        - **raw**: original token vectors;
        - **unit**: L2-normalized token directions;
        - **whitened**: PCA-whitened coordinates.

        These are different mathematical objects. A topological feature that only
        appears in one view may be a preprocessing artifact or may be a real effect
        of the chosen geometry. Either way, the view must be named explicitly.
        """
    ),
    code(
        """
        cloud_views = make_cloud_views(token_clouds, max_whiten_dim=64, seed=SEED)
        view_table = pd.DataFrame(
            [
                {
                    "view": name,
                    "cloud": view.cloud_name,
                    "kind": view.view_kind,
                    "shape": tuple(view.tokens.shape),
                    "notes": view.notes,
                }
                for name, view in cloud_views.items()
            ]
        )
        display(view_table)
        """
    ),
    md(
        """
        ## 6. Broad Geometry Scan

        Now we compute a compact metric table. Each row is one cloud view.

        The most important columns:

        - `norm_cv`: norm variation;
        - `pc1`: variance captured by the first principal component;
        - `participation_ratio`: rough effective covariance dimension;
        - `twonn_id`: local intrinsic-dimension estimate;
        - `density_q90_q10`: spread in nearest-neighbor distances;
        - `near_duplicate_fraction`: expected for VQ-style codebook effects;
        - `spatial_cosine_mean`: similarity of adjacent spatial tokens.
        """
    ),
    code(
        """
        metrics = geometry_metrics(cloud_views, n_images=len(images), max_points=MAX_POINTS, seed=SEED)
        display(metrics)
        """
    ),
    md(
        """
        ## 7. Main Effects Table

        This is the high-level readout. It flags effects large enough to remember
        before doing topology.
        """
    ),
    code(
        """
        distance_effects = distance_preprocessing_effects(cloud_views, max_points=700 if not SMOKE else 350, seed=SEED)
        effects = main_effect_table(metrics, distance_effects)
        display(
            effects[
                [
                    "cloud",
                    "family",
                    "dim",
                    "norm_cv",
                    "pc1",
                    "participation_ratio",
                    "twonn_id",
                    "density_q90_q10",
                    "near_duplicate_fraction",
                    "spatial_cosine_mean",
                    "raw_unit_spearman",
                    "raw_whitened_spearman",
                    "takeaway",
                ]
            ]
        )
        """
    ),
    code(
        """
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        raw = metrics[metrics["view_kind"] == "raw"].copy()

        sns.barplot(data=raw, x="cloud", y="norm_cv", ax=axes[0, 0])
        axes[0, 0].set_title("Norm variation")
        axes[0, 0].tick_params(axis="x", rotation=35)

        sns.barplot(data=raw, x="cloud", y="pc1", ax=axes[0, 1])
        axes[0, 1].set_title("Dominance of first PC")
        axes[0, 1].tick_params(axis="x", rotation=35)

        sns.barplot(data=raw, x="cloud", y="density_q90_q10", ax=axes[1, 0])
        axes[1, 0].set_title("Density variation")
        axes[1, 0].tick_params(axis="x", rotation=35)

        sns.barplot(data=raw, x="cloud", y="spatial_cosine_mean", ax=axes[1, 1])
        axes[1, 1].set_title("Adjacent-token cosine")
        axes[1, 1].tick_params(axis="x", rotation=35)

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 8. How Much Does Preprocessing Rewrite Distances?

        This is the most direct check against accidental topology.

        If `raw_vs_unit` is low, token norms strongly affect pairwise distances.
        If `raw_vs_whitened` is low, covariance anisotropy strongly affects
        pairwise distances.
        """
    ),
    code(
        """
        display(distance_effects)

        fig, ax = plt.subplots(figsize=(11, 4))
        sns.barplot(data=distance_effects, x="cloud", y="spearman_distance_corr", hue="comparison", ax=ax)
        ax.axhline(0.85, color="black", linestyle="--", linewidth=1)
        ax.set_title("Distance-rank stability under preprocessing")
        ax.set_ylabel("Spearman correlation with raw distances")
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 9. PCA Spectra

        PCA spectra are a simple way to see whether a cloud is flat, anisotropic,
        or dominated by a small number of directions. This is not a final model of
        the data, but it tells us what kind of geometry PH would be seeing.
        """
    ),
    code(
        """
        spectrum_rows = []
        for name, cloud in token_clouds.items():
            x = cloud.tokens
            if len(x) > MAX_POINTS:
                rng = np.random.default_rng(SEED)
                x = x[np.sort(rng.choice(len(x), size=MAX_POINTS, replace=False))]
            n_components = min(20, x.shape[0] - 1, x.shape[1])
            pca = PCA(n_components=n_components, random_state=SEED).fit(x)
            for component, ratio in enumerate(pca.explained_variance_ratio_, start=1):
                spectrum_rows.append({"cloud": name, "component": component, "explained_variance_ratio": ratio})

        spectrum = pd.DataFrame(spectrum_rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=spectrum, x="component", y="explained_variance_ratio", hue="cloud", marker="o", ax=ax)
        ax.set_yscale("log")
        ax.set_title("Raw-token PCA spectra")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 10. PCA Scatter: Local Tokens Colored Two Ways

        Coloring by label asks whether disease class separates token geometry.
        Coloring by image id asks whether each image forms its own local regime.

        If image id dominates, then treating all spatial tokens as independent
        samples overstates the effective sample size.
        """
    ),
    code(
        """
        for name in ["flux_vae", "sd_vae_ft_mse", "kandinsky_movq", "vit_base_patch16"]:
            if name in token_clouds:
                plot_pca_by_label(token_clouds[name], max_points=2500 if not SMOKE else 1200, seed=SEED)
        """
    ),
    md(
        """
        ## 11. Back-Mapping: What Do Extreme Tokens Look Like?

        A stable feature is only convincing if we can connect it back to images.
        Here we inspect approximate source patches for:

        - high-norm tokens;
        - low-norm tokens;
        - locally dense tokens;
        - locally sparse tokens.

        For autoencoders this is approximate because a latent cell's receptive
        field is larger than one exact crop. It is still a useful sanity check.
        """
    ),
    code(
        """
        for name in ["flux_vae", "kandinsky_movq", "raw_patches"]:
            if name in token_clouds:
                show_representative_patches(token_clouds[name], images, image_size=AUTOENCODER_SIZE)
        """
    ),
    md(
        """
        ## 12. Geometry-First Interpretation

        Before computing persistent homology, we now know what questions to carry
        forward:

        - Does a feature survive raw, unit-normalized, and whitened views?
        - Is it explained by norm variation or covariance anisotropy?
        - Is it explained by spatial correlation between neighboring tokens?
        - For VQ models, is it a codebook discreteness effect?
        - Can supporting landmarks be mapped back to coherent image patches?

        Only after these checks should a persistence diagram be treated as
        evidence rather than decoration.
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
