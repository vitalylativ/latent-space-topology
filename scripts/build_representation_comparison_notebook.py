"""Build the cross-representation TDA comparison notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "08_representation_comparison.ipynb"


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
        # 08 - Cross-Representation TDA Comparison

        Notebook 07 treated the FLUX diagram as pipeline-sensitive unless it
        survives controls. This notebook asks a complementary question:

        > If we keep the TDA pipeline fixed, do different image representations
        > produce different topological fingerprints?

        The comparison includes continuous diffusion VAEs, a VQ-style tokenizer,
        transformer patch embeddings, CLIP vision patch embeddings, and raw
        image patches. This is still exploratory: the goal is to find promising
        contrasts, not to rank encoders.
        """
    ),
    md(
        """
        ## What Is Held Fixed?

        For every representation we use the same high-level pipeline:

        ```text
        images
          -> encoder or patch extractor
          -> spatial/local token vectors
          -> L2-normalize token vectors to a unit sphere
          -> select a dense subset by kth-nearest-neighbor distance
          -> choose farthest-point landmarks
          -> run Vietoris-Rips persistent homology
          -> compare observed landmarks to simple controls
        ```

        The token dimension and grid size differ across representations. That is
        part of the object being compared, so the plots report normalized
        persistence as `persistence / filtration_threshold` in addition to raw
        counts.
        """
    ),
    code(
        """
        from __future__ import annotations

        from dataclasses import dataclass
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
        from scipy.spatial.distance import pdist

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="The input point cloud has more columns than rows.*")
        plt.rcParams["figure.max_open_warning"] = 120
        sns.set_theme(style="whitegrid", context="notebook")

        from notebook_utils.encoder_explorer import (
            DEFAULT_IMAGE_DIR,
            choose_device,
            code_usage_table,
            extract_token_clouds,
            l2_normalize,
            load_project_images,
            seed_everything,
            shape_summary,
            show_image_grid,
            token_norm_table,
        )
        from notebook_utils.flux_tda import (
            TDASample,
            build_observed_tda_sample,
            build_random_sphere_sample,
            build_uniform_sphere_sample,
            plot_betti_curves,
            plot_diagrams,
            ripser_diagrams,
            top_persistence_table,
        )
        """
    ),
    md(
        """
        ## 1. Runtime Knobs

        The default run uses all encoder families that were introduced earlier.
        Set `TOKENIZER_ENCODERS` to a comma-separated subset when iterating.
        """
    ),
    code(
        """
        def parse_int_list(text: str) -> list[int]:
            return [int(part.strip()) for part in text.split(",") if part.strip()]


        def parse_str_list(text: str) -> list[str]:
            return [part.strip() for part in text.split(",") if part.strip()]


        SEED = int(os.environ.get("TOKENIZER_NOTEBOOK_SEED", "72"))
        SMOKE = os.environ.get("TOKENIZER_SMOKE", "0") == "1"

        default_encoders = "flux_vae,sd_vae_ft_mse,kandinsky_movq,vit_base_patch16,clip_vit_base_patch32,raw_patches"
        smoke_encoders = "flux_vae,sd_vae_ft_mse,raw_patches"
        SELECTED = parse_str_list(os.environ.get("TOKENIZER_ENCODERS", smoke_encoders if SMOKE else default_encoders))

        N_IMAGES = int(os.environ.get("TOKENIZER_N_IMAGES", "4" if SMOKE else "12"))
        BATCH_SIZE = int(os.environ.get("TOKENIZER_BATCH_SIZE", "2" if SMOKE else "4"))
        AUTOENCODER_SIZE = int(os.environ.get("TOKENIZER_AUTOENCODER_SIZE", "256"))
        VIT_SIZE = int(os.environ.get("TOKENIZER_VIT_SIZE", "224"))
        IMAGE_DIR = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))

        COMPARISON_SEEDS = parse_int_list(os.environ.get("TOKENIZER_COMPARISON_SEEDS", "72" if SMOKE else "72,73"))
        N_DENSE = int(os.environ.get("TOKENIZER_COMPARISON_N_DENSE", "250" if SMOKE else "500"))
        N_LANDMARKS = int(os.environ.get("TOKENIZER_COMPARISON_N_LANDMARKS", "45" if SMOKE else "75"))
        K_DENSITY = int(os.environ.get("TOKENIZER_COMPARISON_K_DENSITY", "10" if SMOKE else "16"))
        MAXDIM = int(os.environ.get("TOKENIZER_COMPARISON_MAXDIM", "2"))
        DISTANCE_QUANTILE = float(os.environ.get("TOKENIZER_COMPARISON_DISTANCE_QUANTILE", "0.82"))

        seed_everything(SEED)
        DEVICE = choose_device(force_cpu=os.environ.get("TOKENIZER_FORCE_CPU", "0") == "1")

        display(
            pd.DataFrame(
                [
                    {"knob": "device", "value": DEVICE},
                    {"knob": "smoke", "value": SMOKE},
                    {"knob": "selected", "value": SELECTED},
                    {"knob": "n_images", "value": N_IMAGES},
                    {"knob": "seeds", "value": COMPARISON_SEEDS},
                    {"knob": "n_dense", "value": N_DENSE},
                    {"knob": "n_landmarks", "value": N_LANDMARKS},
                    {"knob": "k_density", "value": K_DENSITY},
                    {"knob": "maxdim", "value": MAXDIM},
                    {"knob": "distance_quantile", "value": DISTANCE_QUANTILE},
                    {"knob": "image_dir", "value": IMAGE_DIR},
                ]
            )
        )
        """
    ),
    md(
        """
        ## 2. Load Images

        The same image subset is used for every representation. Labels are shown
        only to help interpret what the local patches came from; the TDA object
        is the token distribution.
        """
    ),
    code(
        """
        images, image_metadata = load_project_images(N_IMAGES, IMAGE_DIR)
        display(image_metadata.head(12))
        show_image_grid(images, image_metadata, n=min(12, len(images)), title="Images used for representation comparison")
        """
    ),
    md(
        """
        ## 3. Extract Token Clouds

        This cell may download model weights on a fresh machine. Once cached, the
        full default set usually runs quickly enough for exploratory work.
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
            vit_size=VIT_SIZE,
            selected=SELECTED,
        )
        encode_seconds = time.perf_counter() - t0

        if not failures.empty:
            display(failures)

        print(f"encoding seconds: {encode_seconds:.2f}")
        display(shape_summary(token_clouds))
        display(token_norm_table(token_clouds).round(4))
        """
    ),
    md(
        """
        ## 4. Representation Notes

        The same word "token" hides several different objects. The table below
        keeps those differences visible before comparing diagrams.
        """
    ),
    code(
        """
        representation_rows = []
        for name, cloud in token_clouds.items():
            norms = np.linalg.norm(cloud.tokens, axis=1)
            row = {
                "cloud": name,
                "family": cloud.family,
                "token_kind": cloud.token_kind,
                "n_tokens": len(cloud.tokens),
                "channel_dim": cloud.channel_dim,
                "grid_shape": cloud.grid_shape,
                "mean_norm": float(norms.mean()),
                "std_norm": float(norms.std()),
                "cv_norm": float(norms.std() / max(norms.mean(), 1e-12)),
                "model_id": cloud.model_id,
            }
            if cloud.code_indices is not None:
                usage = code_usage_table(cloud)
                row["unique_codes"] = len(usage)
                row["top_code_frequency"] = float(usage["frequency"].iloc[0])
            representation_rows.append(row)

        representation_table = pd.DataFrame(representation_rows)
        display(representation_table)

        if "kandinsky_movq" in token_clouds and token_clouds["kandinsky_movq"].code_indices is not None:
            print("Top Kandinsky / MoVQ code usage")
            display(code_usage_table(token_clouds["kandinsky_movq"]).head(12))
        """
    ),
    md(
        """
        ## 5. Comparison Helpers

        Controls are intentionally cheap:

        - `observed_dense`: dense sphere landmarks from the representation;
        - `random_tokens`: random sphere-normalized tokens from the same cloud;
        - `uniform_sphere`: random directions in the same ambient dimension;
        - `diag_gaussian`: independent per-channel Gaussian matched to the raw
          token mean and standard deviation, then sphere-normalized.

        The diagonal Gaussian is not a full covariance control. It is a fast
        baseline for marginal scale effects.
        """
    ),
    code(
        """
        def build_diag_gaussian_sphere_sample(cloud, n_landmarks: int, seed: int) -> TDASample:
            rng = np.random.default_rng(seed)
            raw = cloud.tokens.astype(np.float32)
            mean = raw.mean(axis=0)
            std = raw.std(axis=0)
            std = np.maximum(std, 1e-6)
            x = rng.normal(loc=mean, scale=std, size=(n_landmarks, raw.shape[1])).astype(np.float32)
            x = l2_normalize(x).astype(np.float32)
            return TDASample(
                name="diag_gaussian_sphere",
                tokens=x,
                source_indices=np.arange(n_landmarks),
                metadata=pd.DataFrame({"image_id": ["control"] * n_landmarks, "label": ["diag_gaussian"] * n_landmarks, "h": 0, "w": 0}),
                notes={"control": "independent Gaussian matched to per-channel raw means/stds, then sphere-projected"},
            )


        def pairwise_stats(sample: TDASample) -> dict[str, float]:
            distances = pdist(sample.tokens, metric="euclidean")
            mean = float(distances.mean()) if len(distances) else 0.0
            std = float(distances.std()) if len(distances) else 0.0
            return {
                "pairwise_mean": mean,
                "pairwise_std": std,
                "pairwise_cv": std / mean if mean > 1e-12 else 0.0,
            }


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


        def build_samples_for_cloud(cloud, seed: int) -> list[tuple[str, TDASample]]:
            observed, _ = build_observed_tda_sample(
                cloud,
                n_dense=N_DENSE,
                n_landmarks=N_LANDMARKS,
                k_density=K_DENSITY,
                seed=seed,
            )
            observed.name = "observed_dense"
            return [
                ("observed_dense", observed),
                ("random_tokens", build_random_sphere_sample(cloud, n_landmarks=N_LANDMARKS, seed=seed + 101)),
                ("uniform_sphere", build_uniform_sphere_sample(cloud.channel_dim, n_landmarks=N_LANDMARKS, seed=seed + 102)),
                ("diag_gaussian", build_diag_gaussian_sphere_sample(cloud, n_landmarks=N_LANDMARKS, seed=seed + 103)),
            ]
        """
    ),
    md(
        """
        ## 6. Run the Same TDA Pipeline for Every Representation

        Runtime is mostly controlled by the number of landmarks and diagram
        runs. With the default settings this is meant to be a compact
        comparison, not a final benchmark.
        """
    ),
    code(
        """
        results: dict[tuple, dict] = {}
        summaries = []

        t0 = time.perf_counter()
        for cloud_name, cloud in token_clouds.items():
            for seed in COMPARISON_SEEDS:
                for sample_kind, sample in build_samples_for_cloud(cloud, seed):
                    print(f"ripser: {cloud_name:24s} {sample_kind:16s} seed={seed} n={len(sample.tokens)} dim={sample.tokens.shape[1]}")
                    result = ripser_diagrams(sample, maxdim=MAXDIM, distance_quantile=DISTANCE_QUANTILE)
                    result["sample"] = sample_kind
                    key = (cloud_name, seed, sample_kind)
                    results[key] = result
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

        tda_seconds = time.perf_counter() - t0
        summary = pd.concat(summaries, ignore_index=True)

        print(f"TDA seconds: {tda_seconds:.2f}")
        print(f"diagram runs: {len(results)}")
        display(summary.head())
        display(summary.groupby(["cloud", "sample_kind", "dim"]).size().rename("runs").reset_index())
        """
    ),
    md(
        """
        ## 7. Observed Fingerprints Across Representations

        This is the main comparison: observed dense landmarks only, summarized
        across seeds. If a representation has a distinctive topology-like
        signature, it should start to appear here.
        """
    ),
    code(
        """
        observed = summary[(summary["sample_kind"] == "observed_dense") & (summary["dim"].isin([1, 2]))].copy()

        display(
            observed
            .groupby(["cloud", "family", "dim"])
            [["max_persistence_fraction", "n_finite", "pairwise_cv"]]
            .agg(["mean", "std"])
            .round(4)
        )

        g = sns.catplot(
            data=observed,
            x="cloud",
            y="max_persistence_fraction",
            hue="dim",
            kind="bar",
            height=4.8,
            aspect=1.7,
            sharey=False,
        )
        g.set_xticklabels(rotation=25, horizontalalignment="right")
        g.fig.suptitle("Observed dense landmarks: normalized persistence", y=1.03)
        plt.show()
        """
    ),
    md(
        """
        ## 8. Observed Versus Controls

        A representation-level fingerprint is more interesting if it differs
        from its own controls. This plot compares `H1` persistence against the
        random-token, uniform-sphere, and diagonal-Gaussian baselines.
        """
    ),
    code(
        """
        h1 = summary[summary["dim"] == 1].copy()

        display(
            h1
            .groupby(["cloud", "sample_kind"])
            [["max_persistence_fraction", "top3_persistence", "pairwise_cv"]]
            .agg(["mean", "std"])
            .round(4)
        )

        g = sns.catplot(
            data=h1,
            x="sample_kind",
            y="max_persistence_fraction",
            hue="cloud",
            kind="bar",
            height=5.0,
            aspect=1.65,
            sharey=False,
        )
        g.set_xticklabels(rotation=20, horizontalalignment="right")
        g.fig.suptitle("H1 persistence: observed versus controls", y=1.03)
        plt.show()
        """
    ),
    md(
        """
        ## 9. Effect Sizes Against Uniform Sphere

        The table below subtracts each representation's uniform-sphere control
        from its observed dense-landmark summary. Positive values mean the
        observed representation had stronger normalized persistence than a
        random cloud in the same ambient dimension.
        """
    ),
    code(
        """
        means = (
            summary[summary["dim"].isin([1, 2])]
            .groupby(["cloud", "family", "sample_kind", "dim"], as_index=False)
            ["max_persistence_fraction"]
            .mean()
        )
        observed_means = means[means["sample_kind"] == "observed_dense"].rename(columns={"max_persistence_fraction": "observed_mean"})
        uniform_means = means[means["sample_kind"] == "uniform_sphere"][["cloud", "dim", "max_persistence_fraction"]].rename(columns={"max_persistence_fraction": "uniform_mean"})
        random_means = means[means["sample_kind"] == "random_tokens"][["cloud", "dim", "max_persistence_fraction"]].rename(columns={"max_persistence_fraction": "random_token_mean"})

        effect = (
            observed_means
            .merge(uniform_means, on=["cloud", "dim"], how="left")
            .merge(random_means, on=["cloud", "dim"], how="left")
        )
        effect["delta_vs_uniform"] = effect["observed_mean"] - effect["uniform_mean"]
        effect["delta_vs_random_tokens"] = effect["observed_mean"] - effect["random_token_mean"]
        effect = effect.sort_values(["dim", "delta_vs_uniform"], ascending=[True, False])
        display(effect[["cloud", "family", "dim", "observed_mean", "uniform_mean", "random_token_mean", "delta_vs_uniform", "delta_vs_random_tokens"]].round(4))
        """
    ),
    md(
        """
        ## 10. Distance Concentration Check

        High-dimensional clouds often have concentrated pairwise distances. If
        distances are too concentrated, persistent features can be dominated by
        finite-sample geometry rather than representation structure.
        """
    ),
    code(
        """
        distance_view = summary[(summary["sample_kind"] == "observed_dense") & (summary["dim"] == 1)].copy()
        display(distance_view[["cloud", "family", "channel_dim", "seed", "pairwise_mean", "pairwise_std", "pairwise_cv", "threshold"]].round(4))

        fig, ax = plt.subplots(figsize=(9, 4.4))
        sns.barplot(data=distance_view, x="cloud", y="pairwise_cv", hue="family", dodge=False, ax=ax)
        ax.set_title("Observed landmark pairwise-distance concentration")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 11. Representative Diagrams

        These are one-seed diagrams for the observed dense landmarks. They are
        not the main evidence, but they make the summary table easier to inspect.
        """
    ),
    code(
        """
        representative_seed = COMPARISON_SEEDS[0]
        representative_results = [
            results[(cloud_name, representative_seed, "observed_dense")]
            for cloud_name in token_clouds
            if (cloud_name, representative_seed, "observed_dense") in results
        ]
        plot_diagrams(representative_results)
        plot_betti_curves(representative_results)
        """
    ),
    md(
        """
        ## 12. Longest Bars

        The longest finite bars are useful for spotting outliers. They are not
        enough for a claim, but they tell us which representation deserves
        follow-up with cycle or landmark interpretation.
        """
    ),
    code(
        """
        top_tables = []
        for cloud_name in token_clouds:
            key = (cloud_name, representative_seed, "observed_dense")
            if key in results:
                top_tables.append(top_persistence_table(results[key], top_n=5).assign(cloud=cloud_name))
        display(pd.concat(top_tables, ignore_index=True)[["cloud", "sample", "dim", "rank", "birth", "death", "persistence", "threshold"]].round(4))
        """
    ),
    md(
        """
        ## 13. Readout

        This notebook should be read as a prioritization tool. If one
        representation is consistently above its controls, the next notebook
        should map the relevant landmarks or cycles back to patches.
        """
    ),
    code(
        """
        h1_effect = effect[effect["dim"] == 1].sort_values("delta_vs_uniform", ascending=False)
        h2_effect = effect[effect["dim"] == 2].sort_values("delta_vs_uniform", ascending=False) if MAXDIM >= 2 else pd.DataFrame()

        print("Runtime recap")
        print(f"  encode seconds: {encode_seconds:.2f}")
        print(f"  TDA seconds:    {tda_seconds:.2f}")
        print(f"  diagram runs:   {len(results)}")
        print()
        print("H1 observed-minus-uniform ranking")
        display(h1_effect[["cloud", "family", "observed_mean", "uniform_mean", "delta_vs_uniform", "delta_vs_random_tokens"]].round(4))
        if not h2_effect.empty:
            print("H2 observed-minus-uniform ranking")
            display(h2_effect[["cloud", "family", "observed_mean", "uniform_mean", "delta_vs_uniform", "delta_vs_random_tokens"]].round(4))
        print(
            "Interpretation prompt: prioritize representations whose observed dense landmarks beat both "
            "uniform and random-token controls across seeds; deprioritize apparent features that sit inside "
            "the control range."
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
