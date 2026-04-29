"""Build the narrative tokenizer / encoder exploration notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "03_understand_tokenizers_encoders.ipynb"


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
        # 03 - Understanding Image Tokenizers and Encoders

        This notebook is a field guide, not just an experiment runner.

        Goal: build enough context to understand what image encoders/tokenizers
        produce, why their token clouds may have different geometry, and which
        outputs are meaningful to compare before doing topology.

        We use the downloaded bean-leaf images in `data/images/beans` by default.
        """
    ),
    md(
        """
        ## How to Read This Notebook

        Each encoder family gets three layers of attention:

        1. **What it is**: the training objective and the kind of token it emits.
        2. **What it produces here**: output shape, grid size, reconstruction if available.
        3. **First visual intuition**: norms, spatial maps, code maps, and PCA projections.

        The code cells are intentionally small. Most implementation details live
        in `notebook_utils/encoder_explorer.py` so this notebook can stay readable.
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
        import torch

        from notebook_utils.encoder_explorer import (
            DEFAULT_IMAGE_DIR,
            choose_device,
            code_usage_table,
            encoder_story_table,
            extract_token_clouds,
            load_project_images,
            plot_code_map,
            plot_norm_distributions,
            plot_norm_maps,
            plot_pca_by_label,
            seed_everything,
            shape_summary,
            show_image_grid,
            show_reconstruction_grid,
            token_norm_table,
        )
        """
    ),
    code(
        """
        SEED = int(os.environ.get("TOKENIZER_NOTEBOOK_SEED", "72"))
        SMOKE = os.environ.get("TOKENIZER_SMOKE", "0") == "1"
        N_IMAGES = int(os.environ.get("TOKENIZER_N_IMAGES", "4" if SMOKE else "12"))
        BATCH_SIZE = int(os.environ.get("TOKENIZER_BATCH_SIZE", "2" if SMOKE else "4"))
        AUTOENCODER_SIZE = int(os.environ.get("TOKENIZER_AUTOENCODER_SIZE", "256"))
        VIT_SIZE = int(os.environ.get("TOKENIZER_VIT_SIZE", "224"))
        IMAGE_DIR = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))
        SELECTED = os.environ.get("TOKENIZER_ENCODERS")
        SELECTED = [x.strip() for x in SELECTED.split(",")] if SELECTED else None

        seed_everything(SEED)
        DEVICE = choose_device(force_cpu=os.environ.get("TOKENIZER_FORCE_CPU", "0") == "1")

        print(f"device: {DEVICE}")
        print(f"images: {N_IMAGES}")
        print(f"image_dir: {IMAGE_DIR}")
        print(f"torch: {torch.__version__}")
        print(f"mps available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
        """
    ),
    md(
        """
        ## 1. The Images We Will Tokenize

        A tokenizer is only meaningful relative to data. We start with the local
        downloaded bean-leaf images so every plot in this notebook can be
        connected back to visible image patches.
        """
    ),
    code(
        """
        images, image_metadata = load_project_images(N_IMAGES, IMAGE_DIR)
        display(image_metadata.head(12))
        show_image_grid(images, image_metadata, n=min(8, len(images)), title="Local images used in this run")
        """
    ),
    md(
        """
        ## 2. What Counts as an Image Tokenizer?

        In this project, "tokenizer" does not always mean discrete token IDs.
        There are at least three different objects:

        - **Continuous autoencoder latents**: vectors on a spatial grid, used by
          latent diffusion models.
        - **Discrete VQ codes**: integer codebook IDs plus learned embeddings.
        - **Patch embeddings**: representation features from a ViT/CLIP encoder.

        Treating all of these as the same kind of point cloud would be a category
        error. The table below is the context map for the rest of the notebook.
        """
    ),
    code(
        """
        pd.set_option("display.max_colwidth", 120)
        display(encoder_story_table())
        """
    ),
    md(
        """
        ## 3. Encoder Family Notes

        ### FLUX / Stable Diffusion VAEs

        Latent diffusion models do not run the denoising model directly on RGB
        pixels. They first compress an image with an autoencoder. The diffusion
        model then works on that spatial latent tensor.

        A KL autoencoder produces continuous latents. There is no finite codebook.
        Norms may matter because the decoder was trained to interpret both
        direction and scale. This is why normalizing every token onto a unit
        sphere is a modeling choice, not a harmless preprocessing detail.

        ### VQ / MoVQ

        A VQ tokenizer first encodes an image to continuous vectors, then replaces
        each vector with the nearest learned codebook entry. The output has both:

        - a discrete code index;
        - a learned embedding vector for that code.

        This can create exact repeats and code-frequency effects that a continuous
        VAE does not have.

        ### ViT and CLIP Patch Encoders

        ViT/CLIP patch tokens are not reconstruction latents. A patch token after
        transformer layers has seen other patches through attention. CLIP is also
        trained to align images with text, so its geometry is shaped by semantic
        contrastive learning rather than by image decoding.

        ### Raw Patches

        Raw centered/L2-normalized image patches connect us to the classical
        natural-image-patch topology literature. They are a baseline for asking:
        did the encoder preserve, simplify, or reorganize patch-level geometry?
        """
    ),
    md(
        """
        ## 4. Run the Encoders

        This cell is intentionally small. It downloads model weights if needed,
        runs encoder-only inference, and returns a dictionary of `TokenCloud`
        objects. Each cloud stores:

        - `tokens`: a row per spatial token;
        - `token_metadata`: image id and grid position for each row;
        - `grid_shape`: the spatial layout;
        - optional reconstructions or VQ code IDs.
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

        if failures.empty:
            print("All selected encoders ran.")
        else:
            display(failures)
        """
    ),
    md(
        """
        ## 5. What Shapes Did We Get?

        Shape is already a conceptual clue:

        - FLUX VAE: many low-dimensional continuous spatial tokens.
        - SD VAE: similar grid, fewer channels.
        - VQ: same kind of spatial grid, but quantized.
        - ViT/CLIP: fewer patch tokens, much higher channel dimension.
        - Raw patches: no learned model, but high-dimensional patch vectors.
        """
    ),
    code(
        """
        shapes = shape_summary(token_clouds)
        display(shapes)
        """
    ),
    code(
        """
        if not shapes.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            shapes.plot.bar(x="name", y="channel_dim", ax=axes[0], legend=False)
            axes[0].set_title("Token vector dimension")
            axes[0].tick_params(axis="x", rotation=35)
            shapes.assign(tokens=shapes["tokens_shape"].map(lambda s: s[0])).plot.bar(x="name", y="tokens", ax=axes[1], legend=False)
            axes[1].set_title("Number of token rows")
            axes[1].tick_params(axis="x", rotation=35)
            plt.tight_layout()
            plt.show()
        """
    ),
    md(
        """
        ## 6. Decoder Sanity Check

        Autoencoder and VQ tokenizers are constrained by a decoder. Reconstructions
        are not the point of this notebook, but they tell us whether a latent is
        genuinely an image-compression object rather than just an abstract feature.
        """
    ),
    code(
        """
        show_reconstruction_grid(images, token_clouds, image_size=AUTOENCODER_SIZE, max_images=min(4, len(images)))
        """
    ),
    md(
        """
        ## 7. Are Token Norms Meaningful?

        A token vector has direction and length. For continuous autoencoder
        latents, the length can carry decoder-relevant information. For raw
        patches, we deliberately normalized length away. For transformer features,
        norms reflect the representation model and its normalization layers.

        Before using cosine distance or projecting everything onto a sphere, we
        should inspect norm distributions directly.
        """
    ),
    code(
        """
        norm_stats = token_norm_table(token_clouds)
        display(norm_stats)
        plot_norm_distributions(token_clouds)
        """
    ),
    md(
        """
        ## 8. Where Are Large-Norm Tokens in the Image?

        The same norm distribution can mean different things spatially. These maps
        show token norms laid back onto each encoder's spatial grid for one image.

        This is often more intuitive than a histogram: it shows whether the encoder
        reacts to local lesions, leaf edges, background, or global structure.
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
        ## 9. VQ Codes Are Discrete Objects

        For VQ/MoVQ, the code index is not a visualization detail. Code frequency
        tells us how much of the learned dictionary is used on this image sample,
        and repeated codes create repeated embedding vectors in the point cloud.
        """
    ),
    code(
        """
        vq = token_clouds.get("kandinsky_movq")
        if vq is None:
            print("VQ tokenizer did not run.")
        else:
            usage = code_usage_table(vq)
            if usage.empty:
                print("No code indices exposed by this VQ model.")
            else:
                probs = usage["frequency"].to_numpy()
                entropy = -(probs * np.log(probs + 1e-12)).sum()
                print(f"unique codes: {len(usage)}")
                print(f"code perplexity: {np.exp(entropy):.1f}")
                display(usage.head(20))
                plot_code_map(vq, image_id=0)
        """
    ),
    md(
        """
        ## 10. A First Projection: What Does a Token Cloud Look Like?

        PCA is not topology. It is a rough microscope. Here we project one token
        cloud at a time and color points by label and by source image.

        Things to notice:

        - Do tokens from the same image cluster?
        - Do labels separate?
        - Is the cloud dominated by a few directions?
        - Does a projection look continuous or codebook-like?
        """
    ),
    code(
        """
        for name in ["flux_vae", "kandinsky_movq", "vit_base_patch16", "raw_patches"]:
            if name in token_clouds:
                plot_pca_by_label(token_clouds[name], max_points=2000, seed=SEED)
        """
    ),
    md(
        """
        ## 11. What This Notebook Should Have Taught Us

        - FLUX/SD VAEs produce continuous spatial latents. Their norms are part of
          the representation unless we intentionally remove them.
        - VQ models produce both code IDs and embeddings. Duplicate tokens and
          code usage are expected, not bugs.
        - ViT/CLIP patch tokens are representation features, not decoder latents.
          They are useful comparisons but answer a different question.
        - Raw patches are the bridge to the natural-image-patch topology baseline.

        The next notebook should therefore not ask "what is the topology?" yet.
        It should first ask how norms, PCA structure, density, spatial correlation,
        whitening, and normalization reshape the candidate point cloud.
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
