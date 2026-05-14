"""Build the thin Mapper-node stability report notebook."""

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

        This notebook is deliberately thin. It reads the frozen outputs from
        `scripts/run_mapper_node_stability.py` and reports the primary decision,
        dataset-level effects, stable tracks, metadata-shuffle checks, and saved
        Beans patch-atlas records.
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

        from notebook_utils.diagnostic_workbench import load_cached_flux_clouds
        from notebook_utils.encoder_explorer import DEFAULT_IMAGE_DIR, approximate_patch, load_project_images
        """,
        hidden=True,
    ),
    md(
        """
        ## 1. Load Frozen Outputs
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

        display(Markdown(f"Output directory: `{OUTPUT_DIR.relative_to(ROOT)}`"))
        display(
            pd.DataFrame(
                [
                    {"field": "status", "value": verdict.get("status")},
                    {"field": "mean_delta", "value": verdict.get("mean_delta")},
                    {"field": "bootstrap_ci_95", "value": verdict.get("bootstrap_ci_95")},
                    {"field": "positive_datasets", "value": f"{verdict.get('positive_datasets')} / {verdict.get('n_datasets')}"},
                    {"field": "metadata_shuffle_wins", "value": f"{verdict.get('metadata_shuffle_wins')} / {verdict.get('n_datasets')}"},
                ]
            )
        )
        """,
    ),
    md(
        """
        ## 2. Primary Decision

        The registered statistic is:

        `stable_enriched_track_count_delta = observed_count - max(control_counts)`
        """
    ),
    code(
        """
        display(paired)

        if not paired.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.0))
            sns.barplot(data=paired, x="dataset", y="stable_enriched_track_count_delta", color="#4c78a8", ax=ax)
            ax.axhline(0, color="0.25", linewidth=1)
            ax.set_title("Observed minus strongest representation control")
            ax.set_ylabel("stable enriched track count delta")
            ax.set_xlabel("")
            plt.tight_layout()
            plt.show()
        """,
    ),
    md(
        """
        ## 3. Counts By Condition
        """
    ),
    code(
        """
        cols = [
            "dataset",
            "representation",
            "sample_kind",
            "baseline_group",
            "stable_track_count",
            "stable_enriched_track_count",
            "median_best_jaccard_stable",
            "mean_label_purity_excess_stable",
            "metadata_shuffle_stable_enriched_count_q95",
            "metadata_shuffle_win",
        ]
        display(counts[[col for col in cols if col in counts.columns]].sort_values(["dataset", "representation", "baseline_group", "sample_kind"]))
        """,
    ),
    md(
        """
        ## 4. Mapper Run Statistics
        """
    ),
    code(
        """
        stat_cols = [
            "dataset",
            "representation",
            "sample_kind",
            "seed",
            "nodes",
            "edges",
            "graph_h1_rank",
            "coverage_fraction",
            "weighted_label_purity",
        ]
        display(run_stats[[col for col in stat_cols if col in run_stats.columns]].round(3).head(80))
        """,
    ),
    md(
        """
        ## 5. Stable Tracks
        """
    ),
    code(
        """
        stable_cols = [
            "dataset",
            "representation",
            "sample_kind",
            "reference_node_id",
            "reference_size",
            "recurrence",
            "median_best_jaccard",
            "label_purity",
            "label_purity_excess",
            "dominant_label",
            "dominant_image_fraction",
            "spatial_radius",
            "is_stable_track",
            "is_stable_enriched_track",
        ]
        stable = tracks[tracks["is_stable_track"]].copy()
        sort_cols = [col for col in ["is_stable_enriched_track", "label_purity_excess", "median_best_jaccard", "reference_size"] if col in stable.columns]
        if sort_cols:
            stable = stable.sort_values(sort_cols, ascending=False)
        display(stable[[col for col in stable_cols if col in stable.columns]].head(40).round(3))
        """,
    ),
    md(
        """
        ## 6. Metadata-Shuffle Null
        """
    ),
    code(
        """
        if shuffles.empty:
            display(Markdown("No metadata-shuffle table was saved."))
        else:
            observed_shuffles = shuffles[shuffles["baseline_group"] == "observed"].copy()
            display(observed_shuffles.groupby(["dataset", "sample_kind"])["stable_enriched_track_count"].describe().round(3))
            if not observed_shuffles.empty:
                grid = sns.displot(
                    data=observed_shuffles,
                    x="stable_enriched_track_count",
                    col="dataset",
                    col_wrap=3,
                    bins=10,
                    height=3.0,
                    aspect=1.15,
                )
                grid.fig.suptitle("Metadata-shuffle stable enriched track counts", y=1.04)
                plt.show()
        """,
    ),
    md(
        """
        ## 7. Beans Patch Atlas Records

        These cells are not recomputing Mapper. They render the saved
        `source_index` records from the experiment outputs.
        """
    ),
    code(
        """
        def plot_saved_atlas(records: pd.DataFrame, max_nodes: int = 8, patches_per_node: int = 6) -> None:
            if records.empty:
                display(Markdown("No patch-atlas records were saved."))
                return
            flux_clouds = load_cached_flux_clouds()
            if "beans_local" not in flux_clouds:
                display(Markdown("Beans FLUX cache is unavailable, so patches cannot be rendered."))
                return
            cloud = flux_clouds["beans_local"]
            image_dir = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))
            n_images = int(cloud.token_metadata["image_id"].max()) + 1
            images, _ = load_project_images(n_images=n_images, image_dir=image_dir)

            records = records[
                (records["dataset"] == "beans_local")
                & (records["representation"] == "flux_vae")
                & (records["sample_kind"] == "observed")
            ].copy()
            if records.empty:
                display(Markdown("No observed Beans FLUX atlas records were saved."))
                return
            node_ids = records["reference_node_id"].drop_duplicates().head(max_nodes).astype(int).tolist()
            fig, axes = plt.subplots(len(node_ids), patches_per_node, figsize=(1.75 * patches_per_node, 1.9 * len(node_ids)))
            axes = np.asarray(axes).reshape(len(node_ids), patches_per_node)
            context = 2 if cloud.grid_shape[0] >= 16 else 1
            for row_i, node_id in enumerate(node_ids):
                node_records = records[records["reference_node_id"] == node_id].head(patches_per_node)
                label_parts = [f"node {node_id}"]
                first = node_records.iloc[0]
                if "recurrence" in first:
                    label_parts.append(f"rec={first['recurrence']:.2f}")
                if "median_best_jaccard" in first:
                    label_parts.append(f"J={first['median_best_jaccard']:.2f}")
                if "label_purity_excess" in first:
                    label_parts.append(f"excess={first['label_purity_excess']:.2f}")
                for col_i in range(patches_per_node):
                    ax = axes[row_i, col_i]
                    if col_i < len(node_records):
                        source_index = int(node_records.iloc[col_i]["source_index"])
                        ax.imshow(approximate_patch(cloud, images, source_index, image_size=256, context_cells=context))
                    ax.axis("off")
                    if col_i == 0:
                        ax.set_ylabel("\\n".join(label_parts), rotation=0, ha="right", va="center", fontsize=8)
            fig.suptitle("Saved observed Beans FLUX patch atlas", y=1.01)
            plt.tight_layout()
            plt.show()


        display(atlas_records.head(30))
        plot_saved_atlas(atlas_records)
        """,
    ),
    md(
        """
        ## 8. Readout

        This report should be read as a pass/fail for the current Mapper-node
        claim, not for the whole project. A fail means the current node-stability
        criterion does not beat the strongest matched control under the frozen
        rules. A pass means Mapper has earned a more serious role as a local
        representation diagnostic.
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
