"""Build the latent diagnostic workbench notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "16_latent_diagnostic_workbench.ipynb"


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
        # 16 - Latent Diagnostic Workbench

        This notebook is the practical next step after the geometry, TDA, and
        Mapper explorations. It makes one mixed feature table, checks whether
        Mapper nodes can be matched across seeds, and builds a small
        activation-atlas-style patch panel for the stable Beans nodes.

        The point is not to prove a grand topology claim. The point is to find
        which summaries are stable enough to become useful features.
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
        plt.rcParams["figure.max_open_warning"] = 120

        from notebook_utils.encoder_explorer import DEFAULT_IMAGE_DIR, load_project_images
        from notebook_utils.diagnostic_workbench import (
            annotate_stability_with_node_summary,
            build_mapper_stability_runs,
            compute_feature_table,
            load_cached_flux_clouds,
            maybe_load_raw_patch_cloud,
            patch_atlas_records,
            plot_feature_heatmap,
            plot_patch_atlas,
            plot_stability_overview,
            reference_node_stability,
            save_feature_table,
            select_stable_atlas_nodes,
            summarize_stability_runs,
        )
        """,
        hidden=True,
    ),
    md(
        """
        ## 1. Setup

        The notebook is cache-first. In smoke mode it uses fewer clouds and
        fewer points so the whole notebook can be rerun quickly.
        """
    ),
    code(
        """
        def parse_int_list(text: str) -> list[int]:
            return [int(part.strip()) for part in text.split(",") if part.strip()]


        SMOKE = os.environ.get("TOKENIZER_SMOKE", "0") == "1"
        SEED = int(os.environ.get("TOKENIZER_NOTEBOOK_SEED", "72"))
        IMAGE_DIR = os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))
        IMAGE_SIZE = int(os.environ.get("TOKENIZER_AUTOENCODER_SIZE", "256"))

        MAX_CLOUDS = int(os.environ.get("DIAGNOSTIC_MAX_CLOUDS", "2" if SMOKE else "0"))
        MAX_CLOUDS = None if MAX_CLOUDS <= 0 else MAX_CLOUDS
        MAX_GEOMETRY_POINTS = int(os.environ.get("DIAGNOSTIC_MAX_GEOMETRY_POINTS", "650" if SMOKE else "1600"))
        MAX_TDA_POINTS = int(os.environ.get("DIAGNOSTIC_MAX_TDA_POINTS", "90" if SMOKE else "220"))
        MAX_MAPPER_POINTS = int(os.environ.get("DIAGNOSTIC_MAX_MAPPER_POINTS", "650" if SMOKE else "1400"))
        STABILITY_POOL_POINTS = int(os.environ.get("DIAGNOSTIC_STABILITY_POOL_POINTS", "850" if SMOKE else "2200"))
        STABILITY_SAMPLE_FRACTION = float(os.environ.get("DIAGNOSTIC_STABILITY_SAMPLE_FRACTION", "0.75"))
        STABILITY_SEEDS = parse_int_list(os.environ.get("DIAGNOSTIC_STABILITY_SEEDS", "72,73" if SMOKE else "72,73,74"))

        mapper_config = {
            "config": "baseline",
            "n_intervals": int(os.environ.get("DIAGNOSTIC_MAPPER_INTERVALS", "6" if SMOKE else "8")),
            "overlap": float(os.environ.get("DIAGNOSTIC_MAPPER_OVERLAP", "0.35")),
            "min_bin_points": int(os.environ.get("DIAGNOSTIC_MAPPER_MIN_BIN_POINTS", "8" if SMOKE else "12")),
            "min_cluster_size": int(os.environ.get("DIAGNOSTIC_MAPPER_MIN_CLUSTER_SIZE", "4" if SMOKE else "5")),
            "eps_quantile": float(os.environ.get("DIAGNOSTIC_MAPPER_EPS_QUANTILE", "0.35")),
        }

        display(
            pd.DataFrame(
                [
                    {"knob": "smoke", "value": SMOKE},
                    {"knob": "seed", "value": SEED},
                    {"knob": "max_clouds", "value": MAX_CLOUDS},
                    {"knob": "max_geometry_points", "value": MAX_GEOMETRY_POINTS},
                    {"knob": "max_tda_points", "value": MAX_TDA_POINTS},
                    {"knob": "max_mapper_points", "value": MAX_MAPPER_POINTS},
                    {"knob": "stability_pool_points", "value": STABILITY_POOL_POINTS},
                    {"knob": "stability_seeds", "value": STABILITY_SEEDS},
                ]
            )
        )
        display(pd.DataFrame([mapper_config]))
        """,
        hidden=True,
    ),
    md(
        """
        ## 2. Load Available Representations

        Right now the offline cache is strongest for FLUX VAE tokens. If local
        Beans images are available, we also add a raw-patch baseline.
        """
    ),
    code(
        """
        flux_clouds = load_cached_flux_clouds(max_clouds=MAX_CLOUDS)
        if not flux_clouds:
            raise FileNotFoundError("No local FLUX token caches found under outputs/cycle_hunt/data_sweep/cache")

        table_clouds = {f"{dataset}:flux_vae": cloud for dataset, cloud in flux_clouds.items()}

        raw_patch_cloud = maybe_load_raw_patch_cloud(
            n_images=8 if SMOKE else 48,
            image_size=IMAGE_SIZE,
            image_dir=IMAGE_DIR,
        )
        if raw_patch_cloud is not None:
            table_clouds["beans_local:raw_patches"] = raw_patch_cloud

        inventory = []
        for key, cloud in table_clouds.items():
            inventory.append(
                {
                    "key": key,
                    "dataset": cloud.notes.get("dataset"),
                    "name": cloud.name,
                    "tokens": len(cloud.tokens),
                    "dim": cloud.tokens.shape[1],
                    "grid": cloud.grid_shape,
                }
            )
        display(pd.DataFrame(inventory))
        """,
    ),
    md(
        """
        ## 3. One Feature Table

        This table intentionally mixes geometry, TDA, and Mapper rows. The
        `feature_family` column says which columns are meaningful for a row.
        """
    ),
    code(
        """
        feature_table = compute_feature_table(
            table_clouds,
            include_tda=True,
            include_mapper=True,
            max_geometry_points=MAX_GEOMETRY_POINTS,
            max_tda_points=MAX_TDA_POINTS,
            max_mapper_points=MAX_MAPPER_POINTS,
            seed=SEED,
        )
        paths = save_feature_table(feature_table)

        display(Markdown("Saved: " + ", ".join(f"`{path.relative_to(ROOT)}`" for path in paths.values())))
        important_cols = [
            "dataset",
            "representation",
            "feature_family",
            "view",
            "n_tokens",
            "ambient_dim",
            "norm_cv",
            "pc1",
            "participation_ratio",
            "twonn_id",
            "density_q90_q10",
            "spatial_cosine_mean",
            "same_image_nn_rate",
            "label_silhouette",
            "h1_max_persistence_norm",
            "nodes",
            "edges",
            "graph_h1_rank",
            "coverage_fraction",
            "weighted_label_purity",
            "error",
        ]
        existing_cols = [col for col in important_cols if col in feature_table.columns]
        display(feature_table[existing_cols].round(3))
        """,
    ),
    code(
        """
        plot_feature_heatmap(feature_table)
        """,
    ),
    md(
        """
        ## 4. Stable Mapper-Node Matching

        Mapper node IDs are local to one run. The matching below asks a simpler
        question: if we bootstrap overlapping samples from the same token pool,
        does a reference node recover the same source tokens in other runs?
        """
    ),
    code(
        """
        target_dataset = "beans_local" if "beans_local" in flux_clouds else sorted(flux_clouds)[0]
        target_cloud = flux_clouds[target_dataset]

        runs = build_mapper_stability_runs(
            target_cloud,
            seeds=STABILITY_SEEDS,
            lens_name="norm_density",
            config=mapper_config,
            pool_points=STABILITY_POOL_POINTS,
            sample_fraction=STABILITY_SAMPLE_FRACTION,
            pool_seed=SEED,
        )
        run_stats = summarize_stability_runs(runs)
        display(Markdown(f"Target cloud: **{target_dataset}**"))
        display(run_stats.round(3))

        stability, matches = reference_node_stability(
            runs,
            reference_run=sorted(runs)[0],
            min_jaccard=0.05 if SMOKE else 0.08,
            matching="hungarian",
        )
        reference_run = sorted(runs)[0]
        stability_summary = annotate_stability_with_node_summary(stability, runs[reference_run])

        summary_cols = [
            "reference_node_id",
            "reference_size",
            "recurrence",
            "median_best_jaccard",
            "max_best_jaccard",
            "dominant_label",
            "label_purity",
            "label_purity_excess",
            "dominant_image_fraction",
            "spatial_radius",
        ]
        summary_cols = [col for col in summary_cols if col in stability_summary.columns]
        display(stability_summary[summary_cols].head(20).round(3))
        display(matches.head(20).round(3) if not matches.empty else pd.DataFrame({"message": ["No node matches above threshold"]}))
        plot_stability_overview(stability_summary)
        """,
    ),
    md(
        """
        ## 5. Patch Atlas For Stable Nodes

        This is the Activation-Atlas-inspired part, but kept honest: every cell
        is a real approximate source patch attached to a stable Mapper node.
        """
    ),
    code(
        """
        try:
            n_images = int(target_cloud.token_metadata["image_id"].max()) + 1
            images, image_metadata = load_project_images(n_images=n_images, image_dir=IMAGE_DIR)
            atlas_nodes = select_stable_atlas_nodes(
                stability_summary,
                max_nodes=4 if SMOKE else 8,
                min_recurrence=0.5,
                min_size=8 if SMOKE else 16,
            )
            display(Markdown("Selected nodes: " + ", ".join(str(node) for node in atlas_nodes)))
            atlas_records = patch_atlas_records(
                runs[reference_run].graph,
                target_cloud,
                atlas_nodes,
                patches_per_node=4 if SMOKE else 6,
            )
            display(atlas_records.head(40))
            plot_patch_atlas(
                runs[reference_run].graph,
                target_cloud,
                images,
                atlas_nodes,
                node_summary=stability_summary,
                patches_per_node=4 if SMOKE else 6,
                image_size=IMAGE_SIZE,
                title=f"{target_dataset}: stable Mapper-node patch atlas",
            )
        except Exception as exc:
            display(Markdown(f"Patch atlas unavailable: `{type(exc).__name__}: {exc}`"))
        """,
    ),
    md(
        """
        ## 6. Frozen Confirmatory Contract

        The workbench above is exploratory. The config below is the draft
        confirmatory target: stable, label-enriched Mapper node tracks versus
        the strongest matched control.
        """
    ),
    code(
        """
        config_path = ROOT / "experiment_configs" / "mapper_node_stability_v1.json"
        prereg = json.loads(config_path.read_text())
        display(
            pd.DataFrame(
                [
                    {"field": "primary_question", "value": prereg["primary_question"]},
                    {"field": "primary_statistic", "value": prereg["primary_statistic"]},
                    {"field": "pass_rule", "value": "; ".join(prereg["decision_rule"]["pass"])},
                    {"field": "fail_interpretation", "value": prereg["decision_rule"]["fail_interpretation"]},
                ]
            )
        )
        """,
    ),
    md(
        """
        ## Current Interpretation

        Treat any stable node as a candidate regime, not as proof. The next
        useful question is whether the stable nodes beat controls in count,
        enrichment, and recurrence. If they do, Mapper gives us a compact local
        diagnostic. If they do not, the feature table and patch atlas still tell
        us where the representation varies and where Mapper is mostly drawing
        the cover geometry.
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
