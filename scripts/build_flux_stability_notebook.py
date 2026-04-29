"""Build the FLUX TDA stability and artifact-control notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "07_stability_and_artifact_controls.ipynb"


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
        # 07 - Stability and Artifact Controls

        Notebook 06 showed that one concrete FLUX latent-token TDA pipeline can
        produce persistence diagrams. This notebook asks the more important
        follow-up question:

        > Which parts of the diagram survive perturbations, and which parts look
        > like artifacts of the sampling pipeline?

        The experiment keeps the computation modest but repeats the pipeline
        across random seeds, dense-set sizes, landmark counts, and token views.
        It also compares the observed FLUX cloud with simple controls and one
        synthetic positive control.
        """
    ),
    md(
        """
        ## What Counts as Evidence Here?

        A visible bar in one diagram is not yet evidence for latent-space
        structure. A more credible signal should:

        - appear across several random seeds;
        - be stronger for observed FLUX tokens than for null controls;
        - not depend on exactly one dense-set or landmark setting;
        - be interpretable after mapping landmarks back to images;
        - be described numerically, not only visually.

        This notebook focuses on the first four checks. Cycle-level
        interpretability is left for a later notebook.
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

        warnings.filterwarnings("ignore", category=FutureWarning)
        plt.rcParams["figure.max_open_warning"] = 120
        sns.set_theme(style="whitegrid", context="notebook")

        from notebook_utils.encoder_explorer import (
            DEFAULT_IMAGE_DIR,
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
            build_channel_shuffle_dense_sample,
            build_matched_gaussian_sphere_sample,
            build_observed_tda_sample,
            build_random_sphere_sample,
            build_uniform_sphere_sample,
            farthest_point_indices,
            plot_betti_curves,
            plot_diagrams,
            plot_persistence_summary,
            ripser_diagrams,
            select_dense_indices,
            top_persistence_table,
        )
        """
    ),
    md(
        """
        ## 1. Runtime Knobs

        The defaults are deliberately small enough to rerun. Increase
        `FLUX_STABILITY_SEEDS`, `TOKENIZER_N_IMAGES`, or the landmark counts
        when moving from this exploratory pass to a stronger experiment.
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

        N_IMAGES = int(os.environ.get("TOKENIZER_N_IMAGES", "4" if SMOKE else "12"))
        BATCH_SIZE = int(os.environ.get("TOKENIZER_BATCH_SIZE", "2" if SMOKE else "4"))
        AUTOENCODER_SIZE = int(os.environ.get("TOKENIZER_AUTOENCODER_SIZE", "256"))
        IMAGE_DIR = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))

        STABILITY_SEEDS = parse_int_list(os.environ.get("FLUX_STABILITY_SEEDS", "72,73" if SMOKE else "72,73,74,75"))
        VIEWS = parse_str_list(os.environ.get("FLUX_STABILITY_VIEWS", "sphere,raw,whitened"))
        MAXDIM = int(os.environ.get("FLUX_STABILITY_MAXDIM", "2"))
        DISTANCE_QUANTILE = float(os.environ.get("FLUX_STABILITY_DISTANCE_QUANTILE", "0.82"))

        SWEEP_CONFIGS = pd.DataFrame(
            [
                {"config": "baseline", "n_dense": 600, "n_landmarks": 95, "k_density": 16},
                {"config": "fewer_landmarks", "n_dense": 600, "n_landmarks": 70, "k_density": 16},
                {"config": "smaller_dense", "n_dense": 400, "n_landmarks": 95, "k_density": 16},
            ]
        )
        if SMOKE:
            SWEEP_CONFIGS = pd.DataFrame(
                [
                    {"config": "baseline", "n_dense": 250, "n_landmarks": 45, "k_density": 10},
                    {"config": "fewer_landmarks", "n_dense": 250, "n_landmarks": 35, "k_density": 10},
                ]
            )

        seed_everything(SEED)
        DEVICE = choose_device(force_cpu=os.environ.get("TOKENIZER_FORCE_CPU", "0") == "1")

        display(
            pd.DataFrame(
                [
                    {"knob": "device", "value": DEVICE},
                    {"knob": "smoke", "value": SMOKE},
                    {"knob": "n_images", "value": N_IMAGES},
                    {"knob": "seeds", "value": STABILITY_SEEDS},
                    {"knob": "views", "value": VIEWS},
                    {"knob": "maxdim", "value": MAXDIM},
                    {"knob": "distance_quantile", "value": DISTANCE_QUANTILE},
                    {"knob": "image_dir", "value": IMAGE_DIR},
                ]
            )
        )
        display(SWEEP_CONFIGS)
        """
    ),
    md(
        """
        ## 2. Load Data and Encode FLUX Once

        We reuse the same FLUX token cloud for every perturbation. The repeated
        part of this notebook is the TDA sampling pipeline, not the encoder
        forward pass.
        """
    ),
    code(
        """
        images, image_metadata = load_project_images(N_IMAGES, IMAGE_DIR)
        display(image_metadata.head(12))
        show_image_grid(images, image_metadata, n=min(12, len(images)), title="Images used for stability sweep")
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
        ## 3. Views and Summary Helpers

        The first artifact check is whether the result depends on the chosen
        geometric object:

        - `sphere`: L2-normalized token directions on `S^15`;
        - `raw`: original FLUX posterior-mean token vectors;
        - `whitened`: covariance-whitened token vectors with Euclidean distance.

        The persistence scale is different for each view, so we track both raw
        persistence and persistence divided by the run's filtration threshold.
        """
    ),
    code(
        """
        def whiten_tokens(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
            x = x.astype(np.float32)
            centered = x - x.mean(axis=0, keepdims=True)
            cov = np.cov(centered, rowvar=False)
            values, vectors = np.linalg.eigh(cov + np.eye(cov.shape[0]) * eps)
            values = np.maximum(values, eps)
            return (centered @ vectors @ np.diag(1.0 / np.sqrt(values))).astype(np.float32)


        def token_view(cloud, view: str) -> np.ndarray:
            raw = cloud.tokens.astype(np.float32)
            if view == "raw":
                return raw
            if view == "sphere":
                return l2_normalize(raw).astype(np.float32)
            if view == "whitened":
                return whiten_tokens(raw)
            raise ValueError(f"unknown view: {view}")


        def build_observed_view_sample(cloud, view: str, n_dense: int, n_landmarks: int, k_density: int, seed: int) -> TDASample:
            x = token_view(cloud, view)
            dense_global_idx, dense_kth = select_dense_indices(x, n_dense=n_dense, k=k_density, seed=seed)
            dense_tokens = x[dense_global_idx]
            landmark_local_idx = farthest_point_indices(dense_tokens, n_landmarks=n_landmarks, seed=seed)
            landmark_global_idx = dense_global_idx[landmark_local_idx]
            return TDASample(
                name=f"observed_{view}",
                tokens=dense_tokens[landmark_local_idx],
                source_indices=landmark_global_idx,
                metadata=cloud.token_metadata.iloc[landmark_global_idx].reset_index(drop=True),
                notes={
                    "view": view,
                    "density": "small kth-neighbor distance",
                    "n_dense": len(dense_global_idx),
                    "n_landmarks": len(landmark_global_idx),
                    "k_density": k_density,
                    "dense_kth_median": float(np.median(dense_kth)),
                },
            )


        def build_synthetic_circle_sample(dim: int, n_landmarks: int, seed: int, noise: float = 0.025) -> TDASample:
            rng = np.random.default_rng(seed)
            theta = np.linspace(0, 2 * np.pi, n_landmarks, endpoint=False)
            x = np.zeros((n_landmarks, dim), dtype=np.float32)
            x[:, 0] = np.cos(theta)
            x[:, 1] = np.sin(theta)
            if dim > 2:
                x[:, 2:] = rng.normal(scale=noise, size=(n_landmarks, dim - 2))
            x = l2_normalize(x).astype(np.float32)
            return TDASample(
                name="synthetic_circle_positive",
                tokens=x,
                source_indices=np.arange(n_landmarks),
                metadata=pd.DataFrame({"image_id": ["control"] * n_landmarks, "label": ["circle"] * n_landmarks, "h": 0, "w": 0}),
                notes={"control": "noisy circle embedded in the same ambient dimension"},
            )


        def summarize_result(result: dict, *, seed: int, config: str, view: str, group: str, n_dense: int, n_landmarks: int, k_density: int) -> pd.DataFrame:
            rows = []
            for dim, diagram in enumerate(result["diagrams"]):
                finite = diagram[np.isfinite(diagram[:, 1])]
                persistence = finite[:, 1] - finite[:, 0] if len(finite) else np.array([])
                rows.append(
                    {
                        "seed": seed,
                        "config": config,
                        "view": view,
                        "group": group,
                        "sample": result["sample"],
                        "dim": dim,
                        "n_dense": n_dense,
                        "n_landmarks": n_landmarks,
                        "k_density": k_density,
                        "n_features": len(diagram),
                        "n_finite": len(finite),
                        "max_persistence": float(persistence.max()) if len(persistence) else 0.0,
                        "mean_persistence": float(persistence.mean()) if len(persistence) else 0.0,
                        "total_persistence": float(persistence.sum()) if len(persistence) else 0.0,
                        "top3_persistence": float(np.sort(persistence)[-3:].sum()) if len(persistence) else 0.0,
                        "threshold": result["threshold"],
                        "max_persistence_fraction": float(persistence.max() / result["threshold"]) if len(persistence) and result["threshold"] > 0 else 0.0,
                    }
                )
            return pd.DataFrame(rows)


        def run_one(sample: TDASample, *, seed: int, config: str, view: str, group: str, n_dense: int, n_landmarks: int, k_density: int) -> tuple[dict, pd.DataFrame]:
            result = ripser_diagrams(sample, maxdim=MAXDIM, distance_quantile=DISTANCE_QUANTILE)
            result["sample"] = sample.name
            result["notes"] = sample.notes
            summary = summarize_result(
                result,
                seed=seed,
                config=config,
                view=view,
                group=group,
                n_dense=n_dense,
                n_landmarks=n_landmarks,
                k_density=k_density,
            )
            return result, summary
        """
    ),
    md(
        """
        ## 4. Run the Stability Sweep

        This sweep has two parts:

        1. observed FLUX tokens under several views and sampling knobs;
        2. sphere-view controls under the baseline sampling knob.

        The synthetic circle is a positive control: if `H1` does not stand out
        there, this whole pipeline is too weak to trust.
        """
    ),
    code(
        """
        results: dict[tuple, dict] = {}
        summaries = []

        t0 = time.perf_counter()

        for cfg in SWEEP_CONFIGS.to_dict("records"):
            for seed in STABILITY_SEEDS:
                for view in VIEWS:
                    sample = build_observed_view_sample(flux, view=view, seed=seed, **{k: cfg[k] for k in ["n_dense", "n_landmarks", "k_density"]})
                    result, summary = run_one(sample, seed=seed, config=cfg["config"], view=view, group="observed", **{k: cfg[k] for k in ["n_dense", "n_landmarks", "k_density"]})
                    key = (cfg["config"], seed, view, sample.name)
                    results[key] = result
                    summaries.append(summary)

        baseline = SWEEP_CONFIGS.iloc[0].to_dict()
        for seed in STABILITY_SEEDS:
            observed, _ = build_observed_tda_sample(
                flux,
                n_dense=int(baseline["n_dense"]),
                n_landmarks=int(baseline["n_landmarks"]),
                k_density=int(baseline["k_density"]),
                seed=seed,
            )
            control_samples = [
                replace(observed, name="observed_sphere_baseline"),
                build_random_sphere_sample(flux, n_landmarks=int(baseline["n_landmarks"]), seed=seed + 101),
                build_uniform_sphere_sample(flux.channel_dim, n_landmarks=int(baseline["n_landmarks"]), seed=seed + 102),
                build_matched_gaussian_sphere_sample(flux, n_landmarks=int(baseline["n_landmarks"]), seed=seed + 103),
                build_channel_shuffle_dense_sample(
                    flux,
                    n_dense=int(baseline["n_dense"]),
                    n_landmarks=int(baseline["n_landmarks"]),
                    k_density=int(baseline["k_density"]),
                    seed=seed + 104,
                ),
                build_synthetic_circle_sample(flux.channel_dim, n_landmarks=int(baseline["n_landmarks"]), seed=seed + 105),
            ]
            for sample in control_samples:
                group = "positive_control" if "positive" in sample.name else "control"
                if sample.name == "observed_sphere_baseline":
                    group = "observed"
                result, summary = run_one(
                    sample,
                    seed=seed,
                    config="control_baseline",
                    view="sphere",
                    group=group,
                    n_dense=int(baseline["n_dense"]),
                    n_landmarks=int(baseline["n_landmarks"]),
                    k_density=int(baseline["k_density"]),
                )
                key = ("control_baseline", seed, "sphere", sample.name)
                results[key] = result
                summaries.append(summary)

        sweep_seconds = time.perf_counter() - t0
        summary = pd.concat(summaries, ignore_index=True)

        print(f"sweep seconds: {sweep_seconds:.2f}")
        print(f"diagram runs: {len(results)}")
        display(summary.head())
        display(summary.groupby(["group", "sample", "dim"]).size().rename("runs").reset_index())
        """
    ),
    md(
        """
        ## 5. View Sensitivity

        This plot asks whether the observed FLUX signal changes when the same
        dense-landmark procedure is applied to raw, sphere-normalized, or
        whitened tokens.
        """
    ),
    code(
        """
        observed = summary[(summary["group"] == "observed") & (summary["config"].isin(SWEEP_CONFIGS["config"]))]
        observed_higher = observed[observed["dim"].isin([1, 2])].copy()

        display(
            observed_higher
            .groupby(["view", "config", "dim"])
            [["max_persistence_fraction", "n_finite", "total_persistence"]]
            .agg(["mean", "std"])
            .round(4)
        )

        g = sns.catplot(
            data=observed_higher,
            x="view",
            y="max_persistence_fraction",
            hue="config",
            col="dim",
            kind="box",
            height=4,
            aspect=1.1,
            sharey=False,
        )
        g.fig.suptitle("Observed FLUX: view and sampling sensitivity", y=1.05)
        plt.show()
        """
    ),
    md(
        """
        ## 6. Observed Versus Controls

        The cleanest comparison is the baseline sphere pipeline. We compare the
        observed dense landmarks to simple null clouds and to the synthetic
        circle positive control.
        """
    ),
    code(
        """
        baseline_summary = summary[(summary["config"] == "control_baseline") & (summary["dim"].isin([1, 2]))].copy()

        display(
            baseline_summary
            .groupby(["sample", "dim"])
            [["max_persistence_fraction", "n_finite", "top3_persistence"]]
            .agg(["mean", "std"])
            .round(4)
        )

        g = sns.catplot(
            data=baseline_summary,
            x="sample",
            y="max_persistence_fraction",
            hue="dim",
            kind="box",
            height=5,
            aspect=1.7,
            sharey=False,
        )
        g.set_xticklabels(rotation=25, horizontalalignment="right")
        g.fig.suptitle("Baseline sphere pipeline: observed versus controls", y=1.03)
        plt.show()
        """
    ),
    md(
        """
        ## 7. Effect-Size Table

        This small table compares each sample's mean persistence fraction against
        the uniform-sphere control. It is not a statistical test; it is a compact
        triage view for deciding what deserves more runs.
        """
    ),
    code(
        """
        effect = (
            baseline_summary
            .groupby(["sample", "dim"], as_index=False)["max_persistence_fraction"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        uniform = effect[effect["sample"] == "uniform_sphere_control"][["dim", "mean"]].rename(columns={"mean": "uniform_mean"})
        effect = effect.merge(uniform, on="dim", how="left")
        effect["delta_vs_uniform"] = effect["mean"] - effect["uniform_mean"]
        effect = effect.sort_values(["dim", "delta_vs_uniform"], ascending=[True, False])
        display(effect.round(4))
        """
    ),
    md(
        """
        ## 8. Representative Diagrams

        Summary plots are the main evidence in this notebook, but a few
        representative diagrams help sanity-check the numbers. These diagrams
        all use the first stability seed and the baseline sphere setup.
        """
    ),
    code(
        """
        representative_seed = STABILITY_SEEDS[0]
        representative_keys = [
            ("control_baseline", representative_seed, "sphere", "observed_sphere_baseline"),
            ("control_baseline", representative_seed, "sphere", "uniform_sphere_control"),
            ("control_baseline", representative_seed, "sphere", "matched_gaussian_then_sphere"),
            ("control_baseline", representative_seed, "sphere", "channel_shuffle_dense_sphere"),
            ("control_baseline", representative_seed, "sphere", "synthetic_circle_positive"),
        ]
        representative_results = [results[key] for key in representative_keys if key in results]
        plot_diagrams(representative_results)
        plot_persistence_summary(pd.concat([
            summarize_result(
                result,
                seed=representative_seed,
                config="representative",
                view="sphere",
                group="representative",
                n_dense=int(baseline["n_dense"]),
                n_landmarks=int(baseline["n_landmarks"]),
                k_density=int(baseline["k_density"]),
            )
            for result in representative_results
        ]))
        plot_betti_curves(representative_results)
        """
    ),
    md(
        """
        ## 9. Longest Bars

        The next table lists the longest finite bars for the representative
        observed run and for the positive control. This is a quick check that the
        circle control really concentrates its strongest signal in `H1`.
        """
    ),
    code(
        """
        top_tables = []
        for key in [
            ("control_baseline", representative_seed, "sphere", "observed_sphere_baseline"),
            ("control_baseline", representative_seed, "sphere", "synthetic_circle_positive"),
        ]:
            if key in results:
                top_tables.append(top_persistence_table(results[key], top_n=8).assign(run_key=str(key)))
        display(pd.concat(top_tables, ignore_index=True))
        """
    ),
    md(
        """
        ## 10. Readout

        This final cell prints a compact readout for the project log. The most
        useful next step is to take any feature that survives here and map the
        supporting landmarks or cycles back to image patches.
        """
    ),
    code(
        """
        best_observed = (
            observed_higher
            .groupby(["view", "config", "dim"], as_index=False)["max_persistence_fraction"]
            .mean()
            .sort_values("max_persistence_fraction", ascending=False)
            .head(8)
        )
        best_controls = (
            baseline_summary
            .groupby(["sample", "dim"], as_index=False)["max_persistence_fraction"]
            .mean()
            .sort_values("max_persistence_fraction", ascending=False)
            .head(8)
        )

        print("Runtime recap")
        print(f"  encode seconds: {encode_seconds:.2f}")
        print(f"  sweep seconds:  {sweep_seconds:.2f}")
        print(f"  diagram runs:   {len(results)}")
        print()
        print("Strongest observed FLUX settings by normalized persistence:")
        display(best_observed.round(4))
        print("Strongest baseline/control settings by normalized persistence:")
        display(best_controls.round(4))

        readout_rows = []
        for sample_name in ["observed_sphere_baseline", "uniform_sphere_control", "matched_gaussian_then_sphere", "channel_shuffle_dense_sphere", "synthetic_circle_positive"]:
            row = effect[(effect["sample"] == sample_name) & (effect["dim"] == 1)]
            if not row.empty:
                readout_rows.append(
                    {
                        "sample": sample_name,
                        "h1_mean_persistence_fraction": float(row["mean"].iloc[0]),
                        "h1_delta_vs_uniform": float(row["delta_vs_uniform"].iloc[0]),
                    }
                )
        readout = pd.DataFrame(readout_rows).sort_values("h1_mean_persistence_fraction", ascending=False)
        print("H1 baseline readout")
        display(readout.round(4))
        print(
            "Interpretation prompt: if observed_sphere_baseline is not clearly above the null controls, "
            "treat the single-run diagram from notebook 06 as pipeline-sensitive rather than as evidence "
            "for a stable FLUX-specific loop."
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
