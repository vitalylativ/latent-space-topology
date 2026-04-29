"""Build the long-lived H1 cycle hunt report notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "12_cycle_hunt_long_lived_h1.ipynb"


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
        # 12 - Cycle Hunt for the Claimed Long-Lived H1 Feature

        We had not yet reproduced the long-lived cycle that motivated the
        discussion. This notebook summarizes a deliberately broad search over
        data, preprocessing, sampling, landmark counts, and filtration choices.

        Updated short version:

        - **Raw / sphere / whitened FLUX tokens:** no robust long-lived
          FLUX-specific `H1` cycle above controls in the earlier Rips-style
          sweeps.
        - **Broader data:** CIFAR-10 and Fashion-MNIST did not make the cycle
          appear.
        - **PCA8 + sphere + dense farthest-point landmarks:** a strong and
          repeatable `H1` candidate appears.
        - **Friend-style witness complex on S^15 tokens:** the local adapted
          run now shows a candidate at 5000 witnesses and 35 landmarks.

        The witness-complex result is the closest match to the collaborator's
        description, so it changes the status from "probably no cycle" to
        **promising local candidate, not yet scaled or validated**.
        """
    ),
    md(
        """
        ## Sweep Sources

        The detailed CSVs and worker scripts were written under
        `outputs/cycle_hunt/`, which is intentionally ignored by Git. This
        notebook embeds the relevant summaries so collaborators can read the
        conclusion without rerunning the sweeps.

        Search branches:

        - beans parameter sweep: raw/sphere/whitened/PCA-whitened FLUX views;
        - broader data sweep: beans, CIFAR-10, Fashion-MNIST;
        - alternate algorithm sweep: PCA projections, whitening, random/FPS/dense
          samplers;
        - local targeted sweep: PCA8 + sphere normalization + dense FPS,
          full-filtration and threshold-sensitivity checks;
        - local witness-complex replication: S^15-normalized FLUX spatial
          tokens, dense witnesses, 30/35 landmarks, weak witness complex.
        """
    ),
    code(
        """
        from __future__ import annotations

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook")
        """
    ),
    md(
        """
        ## 1. Beans FLUX Parameter Sweep

        This was the most direct check: current beans images, FLUX latents,
        several common views and density/landmark settings. The top candidates
        did **not** beat their control envelope.
        """
    ),
    code(
        """
        beans_summary = pd.DataFrame(
            [
                {
                    "n_images": 48,
                    "view": "pca_whitened",
                    "config": "light",
                    "distance_quantile": 0.75,
                    "observed_h1_fraction_mean": 0.127,
                    "control_h1_fraction_mean": 0.139,
                    "delta_vs_control": -0.012,
                    "ratio_vs_control": 0.914,
                },
                {
                    "n_images": 48,
                    "view": "pca_whitened",
                    "config": "baseline",
                    "distance_quantile": 0.75,
                    "observed_h1_fraction_mean": 0.119,
                    "control_h1_fraction_mean": 0.140,
                    "delta_vs_control": -0.021,
                    "ratio_vs_control": 0.849,
                },
                {
                    "n_images": 24,
                    "view": "raw",
                    "config": "wide_k",
                    "distance_quantile": 0.75,
                    "observed_h1_fraction_mean": 0.113,
                    "control_h1_fraction_mean": 0.140,
                    "delta_vs_control": -0.026,
                    "ratio_vs_control": 0.812,
                },
            ]
        )
        display(beans_summary)

        fig, ax = plt.subplots(figsize=(7, 3.8))
        plot_df = beans_summary.melt(
            id_vars=["view", "config"],
            value_vars=["observed_h1_fraction_mean", "control_h1_fraction_mean"],
            var_name="kind",
            value_name="normalized_h1",
        )
        sns.barplot(data=plot_df, x="view", y="normalized_h1", hue="kind", ax=ax)
        ax.set_title("Beans sweep: best observed settings still inside controls")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 2. Broader Image Data

        We then tested whether bean leaves were too specific. The broader data
        sweep used FLUX VAE tokens on local beans, CIFAR-10, and Fashion-MNIST.
        None produced an observed dense-landmark `H1` candidate above controls.
        """
    ),
    code(
        """
        data_summary = pd.DataFrame(
            [
                {
                    "dataset": "beans_local",
                    "n_images": 48,
                    "top_observed_h1_fraction": 0.0771,
                    "top_control_h1_fraction": 0.1489,
                    "paired_wins": "0/6",
                    "candidate_beats_controls": False,
                },
                {
                    "dataset": "cifar10",
                    "n_images": 48,
                    "top_observed_h1_fraction": 0.0524,
                    "top_control_h1_fraction": 0.1807,
                    "paired_wins": "0/6",
                    "candidate_beats_controls": False,
                },
                {
                    "dataset": "fashion_mnist",
                    "n_images": 48,
                    "top_observed_h1_fraction": 0.0284,
                    "top_control_h1_fraction": 0.1489,
                    "paired_wins": "0/6",
                    "candidate_beats_controls": False,
                },
            ]
        )
        display(data_summary)

        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        plot_df = data_summary.melt(
            id_vars=["dataset"],
            value_vars=["top_observed_h1_fraction", "top_control_h1_fraction"],
            var_name="kind",
            value_name="normalized_h1",
        )
        sns.barplot(data=plot_df, x="dataset", y="normalized_h1", hue="kind", ax=ax)
        ax.set_title("Broader data sweep: observed FLUX H1 loses to controls")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 3. Alternate Sampling and Projection Sweep

        A wider algorithm sweep tried PCA projections, whitening, random
        landmarks, FPS, and density-based samplers. Some settings had small
        positive deltas, but none cleared the stricter long-lived/control-
        separated rule used by that worker.
        """
    ),
    code(
        """
        algorithm_effects = pd.DataFrame(
            [
                {
                    "view": "pca_whiten2",
                    "sampler": "random95",
                    "observed_mean_h1_fraction": 0.1297,
                    "hardest_control": "channel_shuffle",
                    "hardest_control_mean": 0.1113,
                    "delta": 0.0184,
                },
                {
                    "view": "pca_sphere8",
                    "sampler": "random95",
                    "observed_mean_h1_fraction": 0.1519,
                    "hardest_control": "channel_shuffle",
                    "hardest_control_mean": 0.1368,
                    "delta": 0.0151,
                },
                {
                    "view": "pca_whiten8",
                    "sampler": "fps_all95",
                    "observed_mean_h1_fraction": 0.1228,
                    "hardest_control": "view_null",
                    "hardest_control_mean": 0.1094,
                    "delta": 0.0134,
                },
                {
                    "view": "whiten16",
                    "sampler": "dense200_fps95",
                    "observed_mean_h1_fraction": 0.1395,
                    "hardest_control": "view_null",
                    "hardest_control_mean": 0.1266,
                    "delta": 0.0129,
                },
                {
                    "view": "raw16",
                    "sampler": "random95",
                    "observed_mean_h1_fraction": 0.1149,
                    "hardest_control": "channel_shuffle",
                    "hardest_control_mean": 0.1079,
                    "delta": 0.0069,
                },
            ]
        )
        display(algorithm_effects)

        fig, ax = plt.subplots(figsize=(8, 3.8))
        sns.barplot(data=algorithm_effects, x="view", y="delta", hue="sampler", ax=ax)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Algorithm sweep: positive deltas were small")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 4. Targeted PCA8 + Sphere Candidate

        The local integration sweep found a much stronger candidate when we did
        all of the following:

        1. project FLUX tokens to 8 PCA coordinates;
        2. L2-normalize those PCA coordinates to a sphere;
        3. select dense candidates with `k=32`;
        4. choose farthest-point landmarks from the dense subset;
        5. run full `H1` filtrations.

        This candidate beat random-token, uniform-sphere, and channel-shuffled
        controls across 20 seeds and several landmark counts.
        """
    ),
    code(
        """
        pca8_robustness = pd.DataFrame(
            [
                {"view": "pca8_sphere", "n_landmarks": 50, "observed_mean": 0.3747, "control_mean": 0.1702, "delta_mean": 0.2045, "win_rate": 1.00},
                {"view": "pca8_sphere", "n_landmarks": 70, "observed_mean": 0.3868, "control_mean": 0.1783, "delta_mean": 0.2085, "win_rate": 1.00},
                {"view": "pca8_sphere", "n_landmarks": 95, "observed_mean": 0.3951, "control_mean": 0.1713, "delta_mean": 0.2238, "win_rate": 1.00},
                {"view": "pca8_sphere", "n_landmarks": 140, "observed_mean": 0.3932, "control_mean": 0.1818, "delta_mean": 0.2114, "win_rate": 1.00},
                {"view": "pca8_sphere", "n_landmarks": 200, "observed_mean": 0.3984, "control_mean": 0.1829, "delta_mean": 0.2155, "win_rate": 1.00},
                {"view": "pca8_sphere", "n_landmarks": 260, "observed_mean": 0.4126, "control_mean": 0.1859, "delta_mean": 0.2267, "win_rate": 1.00},
                {"view": "sphere", "n_landmarks": 95, "observed_mean": 0.0446, "control_mean": 0.1191, "delta_mean": -0.0745, "win_rate": 0.00},
                {"view": "whitened", "n_landmarks": 95, "observed_mean": 0.0805, "control_mean": 0.1158, "delta_mean": -0.0353, "win_rate": 0.00},
            ]
        )
        display(pca8_robustness)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.lineplot(
            data=pca8_robustness[pca8_robustness["view"] == "pca8_sphere"],
            x="n_landmarks",
            y="observed_mean",
            marker="o",
            label="observed",
            ax=axes[0],
        )
        sns.lineplot(
            data=pca8_robustness[pca8_robustness["view"] == "pca8_sphere"],
            x="n_landmarks",
            y="control_mean",
            marker="o",
            label="best control",
            ax=axes[0],
        )
        axes[0].set_title("PCA8+sphere dense FPS: stable H1 separation")
        axes[0].set_ylabel("max H1 persistence / max pairwise distance")

        sns.barplot(data=pca8_robustness, x="view", y="delta_mean", hue="n_landmarks", ax=axes[1])
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_title("Original sphere/whitened views do not show it")
        axes[1].tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 5. Filtration Threshold Sensitivity

        The PCA8+sphere candidate is not only a full-filtration artifact. In a
        targeted threshold check, the candidate stayed above controls for
        quantile thresholds from `0.75` through `0.95`, as well as the full
        filtration.
        """
    ),
    code(
        """
        threshold_sensitivity = pd.DataFrame(
            [
                {"threshold_mode": "0.75", "observed_mean": 0.4676, "control_mean": 0.2099, "delta_mean": 0.2577},
                {"threshold_mode": "0.82", "observed_mean": 0.4431, "control_mean": 0.2027, "delta_mean": 0.2404},
                {"threshold_mode": "0.90", "observed_mean": 0.4214, "control_mean": 0.1943, "delta_mean": 0.2271},
                {"threshold_mode": "0.95", "observed_mean": 0.4100, "control_mean": 0.1877, "delta_mean": 0.2223},
                {"threshold_mode": "full", "observed_mean": 0.3888, "control_mean": 0.1729, "delta_mean": 0.2159},
            ]
        )
        display(threshold_sensitivity)

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        plot_df = threshold_sensitivity.melt(
            id_vars=["threshold_mode"],
            value_vars=["observed_mean", "control_mean"],
            var_name="kind",
            value_name="normalized_h1",
        )
        sns.lineplot(data=plot_df, x="threshold_mode", y="normalized_h1", hue="kind", marker="o", ax=ax)
        ax.set_title("PCA8+sphere candidate survives threshold changes")
        ax.set_ylabel("capped/finite max H1 normalized by threshold")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 6. Collaborator-Style Witness Complex

        The collaborator's pipeline was different from our earlier Rips-style
        checks. Instead of one landmark cloud per image sample, it treats each
        FLUX latent tensor `(16, 32, 32)` as `1024` vectors in `R^16`, projects
        those vectors to `S^15`, keeps the densest tokens as witnesses, chooses
        only `30-35` landmarks, and builds a weak witness complex.

        A full reproduction with 100-200 images and 10k-20k witnesses is heavy
        locally. The adapted run below uses the cached 48-image beans cloud
        (`49152` spatial tokens), approximate anchor-density for the local
        sweep, and exact-density spot checks on the strongest setting.
        """
    ),
    code(
        """
        witness_local = pd.DataFrame(
            [
                {"dataset": "beans_local", "n_witnesses": 1000, "n_landmarks": 30, "observed_mean": 0.0375, "control_mean": 0.3190, "delta_mean": -0.2815, "win_rate": 0.00, "density": "anchors"},
                {"dataset": "beans_local", "n_witnesses": 1000, "n_landmarks": 35, "observed_mean": 0.1404, "control_mean": 0.3374, "delta_mean": -0.1970, "win_rate": 0.00, "density": "anchors"},
                {"dataset": "beans_local", "n_witnesses": 2000, "n_landmarks": 30, "observed_mean": 0.0771, "control_mean": 0.2530, "delta_mean": -0.1760, "win_rate": 0.00, "density": "anchors"},
                {"dataset": "beans_local", "n_witnesses": 2000, "n_landmarks": 35, "observed_mean": 0.0771, "control_mean": 0.2899, "delta_mean": -0.2128, "win_rate": 0.00, "density": "anchors"},
                {"dataset": "beans_local", "n_witnesses": 5000, "n_landmarks": 30, "observed_mean": 0.2333, "control_mean": 0.2492, "delta_mean": -0.0159, "win_rate": 0.50, "density": "anchors"},
                {"dataset": "beans_local", "n_witnesses": 5000, "n_landmarks": 35, "observed_mean": 0.3281, "control_mean": 0.1443, "delta_mean": 0.1838, "win_rate": 1.00, "density": "anchors"},
            ]
        )
        witness_local["setting"] = (
            witness_local["n_witnesses"].astype(str)
            + "/"
            + witness_local["n_landmarks"].astype(str)
        )
        display(witness_local)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        plot_df = witness_local.melt(
            id_vars=["setting"],
            value_vars=["observed_mean", "control_mean"],
            var_name="kind",
            value_name="normalized_h1",
        )
        sns.barplot(data=plot_df, x="setting", y="normalized_h1", hue="kind", ax=axes[0])
        axes[0].set_title("Weak witness H1: signal appears at 5000/35")
        axes[0].set_xlabel("witnesses/landmarks")
        axes[0].set_ylabel("max H1 persistence / max filtration")

        sns.lineplot(
            data=witness_local,
            x="n_witnesses",
            y="delta_mean",
            hue="n_landmarks",
            marker="o",
            ax=axes[1],
        )
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_title("Observed minus strongest control")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        witness_exact_checks = pd.DataFrame(
            [
                {"seed": 72, "n_witnesses": 5000, "n_landmarks": 35, "observed": 0.3296, "best_control": 0.1090, "delta": 0.2206, "h1_finite_count": 3},
                {"seed": 73, "n_witnesses": 5000, "n_landmarks": 35, "observed": 0.3416, "best_control": 0.1796, "delta": 0.1620, "h1_finite_count": 1},
            ]
        )
        display(witness_exact_checks)

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        plot_df = witness_exact_checks.melt(
            id_vars=["seed"],
            value_vars=["observed", "best_control"],
            var_name="kind",
            value_name="normalized_h1",
        )
        sns.barplot(data=plot_df, x="seed", y="normalized_h1", hue="kind", ax=ax)
        ax.set_title("Exact-density spot checks preserve the witness signal")
        ax.set_ylabel("max H1 persistence / max filtration")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 7. Decision Readout

        The friend's claim can be interpreted in two different ways:

        1. **Claim about Rips-style topology of raw/sphere/whitened FLUX
           token clouds.**  
           Our sweeps do **not** support this. Those views stayed below
           random-token, uniform-sphere, or channel-shuffled controls.

        2. **Claim about a projected/preprocessed FLUX token cloud.**  
           This remains plausible. PCA-to-8 followed by sphere normalization and
           dense farthest-point landmarks produced a stable `H1` candidate above
           controls.

        3. **Claim about the collaborator's witness-complex pipeline on
           S^15-normalized spatial tokens.**  
           This now has local support. The 5000-witness / 35-landmark setting
           beats local controls, and exact-density spot checks preserve the
           separation.

        So the next scientific question is:

        > Does the witness-complex cycle survive scale, image diversity,
        > density-selection variants, and the exact 10k-20k witness regime?
        """
    ),
    code(
        """
        verdict = pd.DataFrame(
            [
                {
                    "question": "Raw/sphere/whitened FLUX has long-lived H1?",
                    "answer": "not supported by Rips sweeps",
                    "evidence": "observed H1 stayed below random/uniform/shuffle controls in direct token-cloud checks",
                },
                {
                    "question": "Beans are too specific?",
                    "answer": "not supported",
                    "evidence": "CIFAR-10 and Fashion-MNIST also lost to controls",
                },
                {
                    "question": "Can we induce/recover a long H1 after preprocessing?",
                    "answer": "yes, PCA8+sphere+dense FPS",
                    "evidence": "20-seed targeted check beat all controls at all landmark sizes",
                },
                {
                    "question": "Does the friend-style witness complex recover H1 on S15 tokens?",
                    "answer": "locally yes",
                    "evidence": "5000 witnesses / 35 landmarks beat controls; exact-density spot checks agree",
                },
                {
                    "question": "Should we call this intrinsic FLUX topology?",
                    "answer": "not yet",
                    "evidence": "current witness evidence is local, small-cache, and not yet tested at 10k-20k witnesses or broad image diversity",
                },
            ]
        )
        display(verdict)

        print("Recommended next checks")
        print("  1. Run the witness pipeline at 10k and 20k witnesses when compute allows.")
        print("  2. Add random natural-image caches, not only beans leaves.")
        print("  3. Compare anchor density, exact kNN density, random landmarks, and density-top landmarks.")
        print("  4. Bootstrap images and seeds around the 5000/35 witness setting.")
        print("  5. Only call it a FLUX latent cycle if the witness signal survives those perturbations.")
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
