"""Build the visual Mapper-node stability report notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "17_mapper_node_stability_report.ipynb"


def md(source: str) -> dict:
    text = dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": "md-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(source: str, hidden: bool = False) -> dict:
    text = dedent(source).strip()
    metadata = {"jupyter": {"source_hidden": True}, "collapsed": True} if hidden else {}
    return {
        "cell_type": "code",
        "id": "code-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10],
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """
        # 17 - Mapper Node Stability Report

        This notebook explains the frozen Mapper-node stability result with
        pictures first. The source of truth is still the CSV/JSON output from
        `scripts/run_mapper_node_stability.py`; the plots below are views of
        those frozen outputs, plus saved patch-atlas records for Beans.
        """
    ),
    code(
        """
        from __future__ import annotations

        import json
        import os
        from pathlib import Path
        import sys
        import warnings

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import display, Markdown

        for candidate in [Path.cwd(), Path.cwd().parent]:
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        ROOT = Path.cwd()
        if not (ROOT / "notebooks").exists() and (ROOT.parent / "notebooks").exists():
            ROOT = ROOT.parent

        warnings.filterwarnings("ignore", category=FutureWarning)
        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams["figure.dpi"] = 120
        plt.rcParams["savefig.dpi"] = 160

        from notebook_utils.diagnostic_workbench import load_cached_flux_clouds, maybe_load_raw_patch_cloud
        from notebook_utils.encoder_explorer import DEFAULT_IMAGE_DIR, approximate_patch, load_project_images
        """,
        hidden=True,
    ),
    md(
        """
        ## 1. Load The Frozen Result
        """
    ),
    code(
        """
        OUTPUT_DIR = Path(os.environ.get("MAPPER_NODE_STABILITY_OUTPUT", ROOT / "outputs" / "mapper_node_stability_v1"))
        if not OUTPUT_DIR.is_absolute():
            OUTPUT_DIR = ROOT / OUTPUT_DIR
        if not OUTPUT_DIR.exists():
            raise FileNotFoundError(f"Run scripts/run_mapper_node_stability.py first. Missing: {OUTPUT_DIR}")

        config = json.loads((OUTPUT_DIR / "config_snapshot.json").read_text())
        verdict = json.loads((OUTPUT_DIR / "verdict.json").read_text())
        counts = pd.read_csv(OUTPUT_DIR / "counts.csv")
        paired = pd.read_csv(OUTPUT_DIR / "paired_primary.csv")
        tracks = pd.read_csv(OUTPUT_DIR / "tracks.csv")
        matches = pd.read_csv(OUTPUT_DIR / "matches.csv")
        run_stats = pd.read_csv(OUTPUT_DIR / "run_stats.csv")
        shuffles = pd.read_csv(OUTPUT_DIR / "metadata_shuffle_counts.csv") if (OUTPUT_DIR / "metadata_shuffle_counts.csv").exists() else pd.DataFrame()
        atlas_records = pd.read_csv(OUTPUT_DIR / "patch_atlas_records.csv") if (OUTPUT_DIR / "patch_atlas_records.csv").exists() else pd.DataFrame()

        thresholds = config["matching"]["stable_track_thresholds"] | config["interpretability"]["primary_metadata_threshold"]
        palette = {
            "observed": "#26734d",
            "channel_shuffle": "#8c6bb1",
            "norm_random_directions": "#d95f02",
            "raw_patches": "#566573",
        }
        sample_order = ["observed", "channel_shuffle", "norm_random_directions", "raw_patches"]

        def short_sample(name: str) -> str:
            return {
                "observed": "observed",
                "channel_shuffle": "channel shuffle",
                "norm_random_directions": "norm-random",
                "raw_patches": "raw patches",
            }.get(str(name), str(name))

        display(
            Markdown(
                f'''
                **Output:** `{OUTPUT_DIR.relative_to(ROOT)}`  
                **Verdict:** `{verdict["status"]}`  
                **Mean delta:** `{verdict["mean_delta"]}`  
                **Bootstrap 95% CI:** `{verdict["bootstrap_ci_95"]}`  
                **Positive datasets:** `{verdict["positive_datasets"]} / {verdict["n_datasets"]}`  
                **Metadata-shuffle wins:** `{verdict["metadata_shuffle_wins"]} / {verdict["n_datasets"]}`
                '''
            )
        )
        """,
    ),
    md(
        """
        ## 2. Process And Result, Step By Step

        This section is the plain-English contract for the whole notebook.

        ### What is one data point?

        One point is **not an image**. One point is one spatial FLUX VAE latent
        token.

        ```text
        one image -> 32 x 32 latent grid -> 1024 token vectors
        one token vector -> one point in R^16
        ```

        Each token keeps metadata: source image id, image label, and grid
        location. That lets us later map a Mapper node back to approximate image
        patches.

        ### What is a Mapper node?

        Mapper does this:

        ```text
        high-dimensional tokens
          -> compute a 2D lens, here norm + local density
          -> cover the lens with overlapping bins
          -> inside each bin, cluster the original R^16 tokens
          -> each cluster becomes a Mapper node
        ```

        So a Mapper node is a **local group of latent tokens**. The graph edges
        come from overlap between bins, but the current stability test mostly
        cares about node membership.

        ### What does rerun mean?

        In this experiment, rerun means:

        ```text
        take one fixed dataset token pool
        sample about 75% of tokens for seed 72
        sample about 75% of tokens for seed 73
        sample about 75% of tokens for seed 74
        sample about 75% of tokens for seed 75
        rebuild Mapper each time
        ```

        These are **overlapping token subsamples**. They are not different
        held-out pictures. This makes the current claim weaker but easier to
        audit, because the same source token can appear in multiple runs.

        ### How do we compare nodes across Mapper graphs?

        Node ids are local to each graph:

        ```text
        seed 72 node 10 does not automatically equal seed 73 node 10
        ```

        Instead, each node is treated as a set of source-token ids.

        ```text
        node A = {5, 19, 44, 80, 91}
        node B = {19, 44, 80, 102}

        Jaccard(A, B) = |A intersection B| / |A union B| = 3 / 6 = 0.50
        ```

        The script computes all pairwise node overlaps from a reference run to
        another run and uses one-to-one Hungarian matching. A reference node is
        considered recurring if it has good matches in enough reruns.

        ### What gets counted?

        A node track is counted only if it passes all gates:

        ```text
        reference node size >= 20
        recurrence >= 0.67
        median best-match Jaccard >= 0.15
        label purity excess >= 0.15
        ```

        `label purity excess` means:

        ```text
        dominant label fraction inside the node
        minus
        dataset majority-label baseline
        ```

        The final measured quantity is:

        ```text
        number of stable + label-enriched Mapper node tracks
        ```

        The primary statistic compares real FLUX tokens to the strongest
        representation control:

        ```text
        observed count - max(control counts)
        ```
        """
    ),
    code(
        """
        fig, axes = plt.subplots(3, 1, figsize=(13, 8.2))
        for ax in axes:
            ax.axis("off")

        # Row 1: data object.
        ax = axes[0]
        boxes = [
            ("image", 0.08),
            ("32 x 32\\nlatent grid", 0.28),
            ("1024 tokens\\nper image", 0.50),
            ("each token is\\na point in R^16", 0.75),
        ]
        for text, x in boxes:
            ax.text(x, 0.56, text, ha="center", va="center", fontsize=11.5, bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef6f2", edgecolor="#555555"), transform=ax.transAxes)
        for x0, x1 in [(0.14, 0.22), (0.35, 0.43), (0.58, 0.68)]:
            ax.annotate("", xy=(x1, 0.56), xytext=(x0, 0.56), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"), xycoords=ax.transAxes, textcoords=ax.transAxes)
        ax.text(0.5, 0.12, "Data object: spatial latent tokens, not whole images.", ha="center", fontsize=10.5, color="#333333", transform=ax.transAxes)

        # Row 2: Mapper construction.
        ax = axes[1]
        boxes = [
            ("R^16 tokens", 0.08),
            ("2D lens\\nnorm + density", 0.28),
            ("overlapping\\nlens bins", 0.49),
            ("cluster original\\ntokens in each bin", 0.70),
            ("Mapper nodes", 0.90),
        ]
        for text, x in boxes:
            ax.text(x, 0.56, text, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f2", edgecolor="#555555"), transform=ax.transAxes)
        for x0, x1 in [(0.15, 0.21), (0.35, 0.42), (0.56, 0.63), (0.78, 0.84)]:
            ax.annotate("", xy=(x1, 0.56), xytext=(x0, 0.56), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"), xycoords=ax.transAxes, textcoords=ax.transAxes)
        ax.text(0.5, 0.12, "Mapper creates local token groups; edges are secondary for this stability test.", ha="center", fontsize=10.5, color="#333333", transform=ax.transAxes)

        # Row 3: matching and count.
        ax = axes[2]
        boxes = [
            ("seed 72\\nMapper nodes", 0.11),
            ("seed 73/74/75\\nMapper nodes", 0.35),
            ("compare node\\nsource-token sets", 0.58),
            ("count stable +\\nlabel-enriched tracks", 0.84),
        ]
        for text, x in boxes:
            ax.text(x, 0.56, text, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef2f8", edgecolor="#555555"), transform=ax.transAxes)
        for x0, x1 in [(0.20, 0.27), (0.45, 0.51), (0.67, 0.76)]:
            ax.annotate("", xy=(x1, 0.56), xytext=(x0, 0.56), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"), xycoords=ax.transAxes, textcoords=ax.transAxes)
        ax.text(0.5, 0.12, "Comparison rule: Jaccard overlap of source-token ids, then one-to-one matching.", ha="center", fontsize=10.5, color="#333333", transform=ax.transAxes)

        plt.tight_layout()
        plt.show()
        """,
    ),
    code(
        """
        result_lines = []
        for _, row in paired.iterrows():
            result_lines.append(
                f"- **{row['dataset']}**: observed `{row['observed_count']}`, strongest control `{row['max_control_count']}` "
                f"(`{row['hardest_control']}`), delta `+{row['stable_enriched_track_count_delta']}`."
            )
        display(
            Markdown(
                "### Result Summary Before The Plots\\n\\n"
                + "\\n".join(result_lines)
                + f"\\n\\nOverall mean delta: `{verdict['mean_delta']}`. "
                + f"Bootstrap 95% CI: `{verdict['bootstrap_ci_95']}`. "
                + f"Positive datasets: `{verdict['positive_datasets']} / {verdict['n_datasets']}`.\\n\\n"
                + "**Interpretation.** Under this exact token-resampling definition, real FLUX geometry produces more stable, "
                + "label-enriched Mapper node tracks than the strongest matched control. The caveat is that Beans raw patches "
                + "also produce such tracks, so the result is not yet FLUX-specific."
            )
        )
        """,
    ),
    md(
        """
        ## 3. Compact Pipeline Picture

        Mapper is rerun on overlapping token subsamples. A node is useful only
        if a similar node reappears in the other runs by exact source-token
        overlap. Then we ask whether that stable node is unusually concentrated
        in one label.

        The controls keep parts of the data distribution while breaking
        representation structure:

        - `channel_shuffle`: preserves each channel's marginal distribution but
          breaks cross-channel relationships.
        - `norm_random_directions`: preserves token norms but randomizes token
          directions.
        - label permutation: keeps the graph fixed and asks how much label
          enrichment appears by chance.
        """
    ),
    code(
        """
        def draw_pipeline() -> None:
            fig, ax = plt.subplots(figsize=(12, 2.4))
            ax.axis("off")
            boxes = [
                ("FLUX tokens", 0.06),
                ("overlapping\\nsubsamples", 0.25),
                ("Mapper graph\\nper seed", 0.44),
                ("match nodes by\\nsource-token Jaccard", 0.64),
                ("count stable +\\nlabel-enriched tracks", 0.86),
            ]
            for text, x in boxes:
                ax.text(
                    x,
                    0.55,
                    text,
                    ha="center",
                    va="center",
                    fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f2", edgecolor="#6b6b6b"),
                    transform=ax.transAxes,
                )
            for x0, x1 in zip([0.13, 0.33, 0.53, 0.74], [0.19, 0.38, 0.58, 0.80]):
                ax.annotate(
                    "",
                    xy=(x1, 0.55),
                    xytext=(x0, 0.55),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#444444"),
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                )
            ax.text(
                0.5,
                0.12,
                "Stable here means recurring under resampling; enriched means class concentration above the declared threshold.",
                ha="center",
                va="center",
                fontsize=10,
                color="#444444",
                transform=ax.transAxes,
            )
            plt.show()


        draw_pipeline()
        """,
    ),
    md(
        """
        ## 4. Primary Result

        The registered statistic is:

        `stable_enriched_track_count_delta = observed_count - max(control_counts)`

        Visually: green bars should sit above the purple/orange controls, and
        the delta should be above zero for the datasets.
        """
    ),
    code(
        """
        flux_counts = counts[counts["representation"] == "flux_vae"].copy()
        flux_counts["sample_label"] = flux_counts["sample_kind"].map(short_sample)
        flux_counts["sample_order"] = flux_counts["sample_kind"].map({name: i for i, name in enumerate(sample_order)})
        flux_counts = flux_counts.sort_values(["dataset", "sample_order"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.45, 1.0]})
        sns.barplot(
            data=flux_counts,
            x="dataset",
            y="stable_enriched_track_count",
            hue="sample_kind",
            hue_order=sample_order[:3],
            palette=palette,
            ax=axes[0],
        )
        axes[0].set_title("Stable label-enriched Mapper tracks")
        axes[0].set_ylabel("count")
        axes[0].set_xlabel("")
        axes[0].legend(title="", labels=[short_sample(t.get_text()) for t in axes[0].legend_.texts])

        for _, row in flux_counts[flux_counts["sample_kind"] == "observed"].iterrows():
            if pd.notna(row.get("metadata_shuffle_stable_enriched_count_q95")):
                x_pos = sorted(flux_counts["dataset"].unique()).index(row["dataset"])
                axes[0].scatter(
                    [x_pos],
                    [row["metadata_shuffle_stable_enriched_count_q95"]],
                    marker="_",
                    s=260,
                    color="black",
                    linewidth=2.2,
                    zorder=4,
                    label="metadata null q95",
                )
        handles, labels = axes[0].get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        axes[0].legend(dedup.values(), dedup.keys(), title="", fontsize=9)

        sns.barplot(
            data=paired,
            x="dataset",
            y="stable_enriched_track_count_delta",
            color="#4c78a8",
            ax=axes[1],
        )
        axes[1].axhline(0, color="#333333", linewidth=1.1)
        axes[1].set_title("Observed minus strongest control")
        axes[1].set_ylabel("delta")
        axes[1].set_xlabel("")
        for i, row in paired.reset_index(drop=True).iterrows():
            axes[1].text(i, row["stable_enriched_track_count_delta"] + 0.35, f"+{int(row['stable_enriched_track_count_delta'])}", ha="center", va="bottom", fontsize=11)
        plt.tight_layout()
        plt.show()

        display(
            Markdown(
                f'''
                **Interpretation.** Observed FLUX wins on all three datasets under the frozen rule. The strongest control is
                `{paired["hardest_control"].mode().iloc[0]}` in this run, which means preserving token norms while randomizing
                directions explains some stable nodes but not the full observed label-enriched count.
                '''
            )
        )
        """,
    ),
    md(
        """
        ## 5. Stability And Enrichment Landscape

        Each point below is a reference Mapper node. The vertical line is the
        stability threshold; the horizontal line is the label-enrichment
        threshold. Points in the upper-right are the nodes counted by the
        primary metric, provided they also pass the size and recurrence rules.
        """
    ),
    code(
        """
        plot_tracks = tracks[tracks["representation"] == "flux_vae"].copy()
        plot_tracks["sample_label"] = plot_tracks["sample_kind"].map(short_sample)
        g = sns.relplot(
            data=plot_tracks,
            x="median_best_jaccard",
            y="label_purity_excess",
            hue="sample_kind",
            size="reference_size",
            sizes=(18, 180),
            col="dataset",
            col_order=sorted(plot_tracks["dataset"].unique()),
            palette=palette,
            alpha=0.70,
            height=4.2,
            aspect=1.0,
        )
        for ax in g.axes.flat:
            ax.axvline(thresholds["median_best_jaccard_min"], color="#333333", linewidth=1, linestyle=":")
            ax.axhline(thresholds["label_purity_excess_min"], color="#333333", linewidth=1, linestyle=":")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=min(-0.08, plot_tracks["label_purity_excess"].min() - 0.02))
        g.set_axis_labels("median best-match Jaccard", "label purity excess")
        g.fig.suptitle("Node stability versus label enrichment", y=1.05)
        plt.show()

        display(
            Markdown(
                '''
                **Interpretation.** The positive result is not simply "Mapper made many nodes." The counted nodes are those
                that land past the stability and enrichment thresholds. Controls can create stable nodes, but far fewer of
                them are also label-enriched.
                '''
            )
        )
        """,
    ),
    md(
        """
        ## 6. Threshold Funnel

        This plot shows where nodes are lost. A robust claim should not rely on
        a single lucky final node after a huge funnel. Here we still keep the
        final count modest, but observed FLUX retains more enriched tracks than
        the controls.
        """
    ),
    code(
        """
        funnel_rows = []
        for keys, group in tracks.groupby(["dataset", "representation", "sample_kind", "baseline_group"]):
            dataset, representation, sample_kind, baseline_group = keys
            size = group["passes_size"]
            recurrent = size & group["passes_recurrence"]
            stable = group["is_stable_track"]
            enriched = group["is_stable_enriched_track"]
            funnel_rows.extend(
                [
                    {"dataset": dataset, "representation": representation, "sample_kind": sample_kind, "stage": "all nodes", "count": len(group)},
                    {"dataset": dataset, "representation": representation, "sample_kind": sample_kind, "stage": "large enough", "count": int(size.sum())},
                    {"dataset": dataset, "representation": representation, "sample_kind": sample_kind, "stage": "recurrent", "count": int(recurrent.sum())},
                    {"dataset": dataset, "representation": representation, "sample_kind": sample_kind, "stage": "stable", "count": int(stable.sum())},
                    {"dataset": dataset, "representation": representation, "sample_kind": sample_kind, "stage": "stable + enriched", "count": int(enriched.sum())},
                ]
            )
        funnel = pd.DataFrame(funnel_rows)
        funnel_flux = funnel[funnel["representation"] == "flux_vae"].copy()
        funnel_flux["stage"] = pd.Categorical(
            funnel_flux["stage"],
            ["all nodes", "large enough", "recurrent", "stable", "stable + enriched"],
            ordered=True,
        )
        g = sns.catplot(
            data=funnel_flux,
            x="count",
            y="stage",
            hue="sample_kind",
            col="dataset",
            kind="bar",
            palette=palette,
            height=4.0,
            aspect=1.05,
            sharex=False,
        )
        g.set_axis_labels("node count", "")
        g.fig.suptitle("How many nodes survive each threshold?", y=1.05)
        plt.show()
        """,
    ),
    md(
        """
        ## 7. Metadata-Shuffle Null

        The graph and node stability are held fixed. Only labels are permuted.
        If the observed count is far to the right of this distribution, the
        label enrichment is unlikely to be a random labeling accident.
        """
    ),
    code(
        """
        observed_counts = counts[(counts["representation"] == "flux_vae") & (counts["sample_kind"] == "observed")].set_index("dataset")
        observed_shuffles = shuffles[(shuffles["representation"] == "flux_vae") & (shuffles["baseline_group"] == "observed")].copy()

        if observed_shuffles.empty:
            display(Markdown("No metadata-shuffle table was saved."))
        else:
            datasets = sorted(observed_shuffles["dataset"].unique())
            fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.6), sharey=True)
            axes = np.asarray(axes).reshape(-1)
            for ax, dataset in zip(axes, datasets):
                values = observed_shuffles[observed_shuffles["dataset"] == dataset]["stable_enriched_track_count"]
                sns.histplot(values, bins=np.arange(values.min(), values.max() + 2) - 0.5, color="#a6bddb", edgecolor="white", ax=ax)
                observed = float(observed_counts.loc[dataset, "stable_enriched_track_count"])
                q95 = float(observed_counts.loc[dataset, "metadata_shuffle_stable_enriched_count_q95"])
                ax.axvline(observed, color="#26734d", linewidth=2.4, label="observed")
                ax.axvline(q95, color="#222222", linewidth=1.5, linestyle=":", label="shuffle q95")
                ax.set_title(dataset)
                ax.set_xlabel("stable enriched tracks")
                ax.legend(fontsize=8)
            axes[0].set_ylabel("metadata shuffles")
            plt.tight_layout()
            plt.show()

            display(
                Markdown(
                    '''
                    **Interpretation.** The observed counts beat the 95th percentile of label permutations in all three datasets.
                    This supports the phrase "label-enriched" as something attached to the geometry, not only to random labels.
                    '''
                )
            )
        """,
    ),
    md(
        """
        ## 8. Mapper Run Sanity Checks

        We do not want the primary result to be a disguised coverage artifact.
        The plots below show run-level Mapper graph summaries across seeds.
        """
    ),
    code(
        """
        stats_flux = run_stats[run_stats["representation"] == "flux_vae"].copy()
        metrics = [
            ("nodes", "nodes"),
            ("coverage_fraction", "coverage"),
            ("weighted_label_purity", "weighted label purity"),
            ("graph_h1_rank", "graph cycle rank"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes = axes.reshape(-1)
        for ax, (metric, label) in zip(axes, metrics):
            sns.stripplot(
                data=stats_flux,
                x="dataset",
                y=metric,
                hue="sample_kind",
                hue_order=sample_order[:3],
                palette=palette,
                dodge=True,
                size=5,
                alpha=0.78,
                ax=ax,
            )
            ax.set_title(label)
            ax.set_xlabel("")
            ax.legend_.remove()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, [short_sample(label) for label in labels], loc="upper center", ncol=3, frameon=False)
        fig.suptitle("Run-level Mapper summaries across seeds", y=1.02)
        plt.tight_layout()
        plt.show()

        display(
            Markdown(
                '''
                **Interpretation.** Controls often have comparable node counts and coverage. That is good: the primary result
                is not just "the observed graph exists." The differentiating part is the number of stable nodes that also carry
                label enrichment.
                '''
            )
        )
        """,
    ),
    md(
        """
        ## 9. Where Stable Nodes Sit In The Lens

        These are reference-node centers in the Mapper lens. The plot is not the
        graph itself, but it shows whether enriched stable tracks are localized
        in particular lens regions.
        """
    ),
    code(
        """
        observed_tracks = tracks[(tracks["representation"] == "flux_vae") & (tracks["sample_kind"] == "observed")].copy()
        observed_tracks["track_kind"] = np.where(observed_tracks["is_stable_enriched_track"], "stable + enriched", np.where(observed_tracks["is_stable_track"], "stable only", "other"))
        style_order = ["stable + enriched", "stable only", "other"]
        g = sns.relplot(
            data=observed_tracks,
            x="lens_0",
            y="lens_1",
            hue="label_purity_excess",
            size="reference_size",
            style="track_kind",
            style_order=style_order,
            col="dataset",
            palette="viridis",
            sizes=(20, 180),
            alpha=0.82,
            height=4.1,
            aspect=1.0,
        )
        g.fig.suptitle("Observed reference nodes in the norm-density lens", y=1.05)
        g.set_axis_labels("lens 0", "lens 1")
        plt.show()
        """,
    ),
    md(
        """
        ## 10. Match Strength

        A node track is only meaningful if it can be recovered across other
        bootstrap samples. Jaccard is conservative here because each run sees
        only part of the fixed token pool.
        """
    ),
    code(
        """
        observed_matches = matches[(matches["representation"] == "flux_vae") & (matches["baseline_group"] == "observed")].copy()
        if observed_matches.empty:
            display(Markdown("No observed matches were saved."))
        else:
            g = sns.displot(
                data=observed_matches,
                x="jaccard",
                col="dataset",
                col_wrap=3,
                bins=18,
                color="#74a9cf",
                height=3.2,
                aspect=1.15,
            )
            for ax in g.axes.flat:
                ax.axvline(thresholds["median_best_jaccard_min"], color="#333333", linestyle=":", linewidth=1.2)
            g.fig.suptitle("Accepted node-match Jaccard values", y=1.05)
            plt.show()
        """,
    ),
    md(
        """
        ## 11. Beans Patch Atlas: FLUX Observed Nodes

        These are real approximate source patches for saved stable candidate
        nodes. They are not generated images and not feature inversion. Each row
        is one Mapper node; columns are representative patches from that node.
        """
    ),
    code(
        """
        def _load_bean_images_for_cloud(cloud):
            image_dir = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))
            n_images = int(cloud.token_metadata["image_id"].max()) + 1
            return load_project_images(n_images=n_images, image_dir=image_dir)[0]


        def _atlas_cloud(representation: str):
            if representation == "flux_vae":
                clouds = load_cached_flux_clouds()
                return clouds.get("beans_local")
            if representation == "raw_patches":
                return maybe_load_raw_patch_cloud(n_images=48, image_size=256, image_dir=os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR)))
            return None


        def plot_saved_patch_atlas(
            records: pd.DataFrame,
            representation: str,
            sample_kind: str,
            title: str,
            max_nodes: int = 10,
            patches_per_node: int = 6,
        ) -> None:
            subset = records[
                (records["dataset"] == "beans_local")
                & (records["representation"] == representation)
                & (records["sample_kind"] == sample_kind)
            ].copy()
            if subset.empty:
                display(Markdown(f"No saved atlas records for `{representation}` / `{sample_kind}`."))
                return
            sort_cols = [col for col in ["is_stable_enriched_track", "recurrence", "median_best_jaccard", "label_purity_excess", "reference_size"] if col in subset.columns]
            subset = subset.sort_values(sort_cols, ascending=False)
            node_ids = subset["reference_node_id"].drop_duplicates().head(max_nodes).astype(int).tolist()
            cloud = _atlas_cloud(representation)
            if cloud is None:
                display(Markdown(f"Could not load cloud for `{representation}`."))
                return
            images = _load_bean_images_for_cloud(cloud)
            context = 2 if cloud.grid_shape[0] >= 16 else 1

            fig, axes = plt.subplots(len(node_ids), patches_per_node, figsize=(1.72 * patches_per_node, 1.95 * len(node_ids)))
            axes = np.asarray(axes).reshape(len(node_ids), patches_per_node)
            for row_i, node_id in enumerate(node_ids):
                node_records = subset[subset["reference_node_id"] == node_id].head(patches_per_node)
                first = node_records.iloc[0]
                label = [
                    f"node {node_id}",
                    f"rec={first['recurrence']:.2f}",
                    f"J={first['median_best_jaccard']:.2f}",
                    f"excess={first['label_purity_excess']:.2f}",
                ]
                for col_i in range(patches_per_node):
                    ax = axes[row_i, col_i]
                    if col_i < len(node_records):
                        source_index = int(node_records.iloc[col_i]["source_index"])
                        ax.imshow(approximate_patch(cloud, images, source_index, image_size=256, context_cells=context))
                    ax.axis("off")
                    if col_i == 0:
                        ax.set_ylabel("\\n".join(label), rotation=0, ha="right", va="center", fontsize=8)
            fig.suptitle(title, y=1.01)
            plt.tight_layout()
            plt.show()


        plot_saved_patch_atlas(
            atlas_records,
            representation="flux_vae",
            sample_kind="observed",
            title="Beans FLUX: saved stable-node patch atlas",
            max_nodes=10,
            patches_per_node=6,
        )
        """,
    ),
    md(
        """
        **Interpretation.** This gallery is the first sanity check against
        abstract metrics. If a row contains visually related patches, the node is
        more likely to represent a local visual regime. If a row is visually
        mixed or dominated by one image/spatial region, the label enrichment may
        be a dataset artifact rather than a representation concept.
        """
    ),
    md(
        """
        ## 12. Beans Patch Atlas: Raw-Patch Specificity Baseline

        The raw-patch baseline is a critical warning. It also produces stable
        enriched tracks, so the current positive Mapper result is not yet
        FLUX-specific. Some of the signal may come from natural image patch
        statistics that FLUX preserves.
        """
    ),
    code(
        """
        plot_saved_patch_atlas(
            atlas_records,
            representation="raw_patches",
            sample_kind="raw_patches",
            title="Beans raw patches: specificity-baseline patch atlas",
            max_nodes=10,
            patches_per_node=6,
        )
        """,
    ),
    md(
        """
        ## 13. Bottom Line

        The result is positive but narrow:

        - The old global `H1` loop story is still not supported.
        - Mapper finds recurring local regions of the token cloud.
        - More of those regions are label-enriched in observed FLUX tokens than
          in the matched representation controls.
        - The Beans raw-patch baseline means we should not yet call this a
          FLUX-specific mechanism.

        The next serious question is therefore:

        > Are stable Mapper nodes capturing representation-specific structure,
        > or mostly natural-image patch structure that many encoders preserve?
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
print(f"wrote {NOTEBOOK_PATH.relative_to(ROOT)}")
