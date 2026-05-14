"""Run the frozen Mapper-node stability experiment.

The experiment is intentionally table-first. It builds observed and matched
control Mapper graphs, matches reference nodes across overlapping bootstrap
samples by exact source-token Jaccard, counts stable label-enriched tracks, and
writes compact CSV/JSON/markdown outputs for a thin report notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebook_utils.diagnostic_workbench import (
    MapperRun,
    annotate_stability_with_node_summary,
    build_mapper_stability_runs,
    load_cached_flux_clouds,
    maybe_load_raw_patch_cloud,
    patch_atlas_records,
    reference_node_stability,
    summarize_stability_runs,
)
from notebook_utils.encoder_explorer import DEFAULT_IMAGE_DIR, TokenCloud


DEFAULT_CONFIG = ROOT / "experiment_configs" / "mapper_node_stability_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "mapper_node_stability_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", default=os.environ.get("TOKENIZER_SMOKE") == "1")
    parser.add_argument("--metadata-shuffles", type=int, default=int(os.environ.get("MAPPER_METADATA_SHUFFLES", "64")))
    parser.add_argument("--bootstrap-samples", type=int, default=int(os.environ.get("MAPPER_BOOTSTRAP_SAMPLES", "10000")))
    parser.add_argument("--image-dir", type=Path, default=Path(os.environ.get("TOKENIZER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))))
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("TOKENIZER_AUTOENCODER_SIZE", "256")))
    return parser.parse_args()


def _mapper_config(config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    mapper = config["mapper"]
    return {
        "config": "baseline",
        "n_intervals": int(mapper["n_intervals"] if not smoke else min(6, mapper["n_intervals"])),
        "overlap": float(mapper["overlap"]),
        "clusterer": str(mapper.get("clusterer", "dbscan")),
        "min_bin_points": int(mapper["min_bin_points"] if not smoke else min(8, mapper["min_bin_points"])),
        "min_cluster_size": int(mapper["min_cluster_size"] if not smoke else min(4, mapper["min_cluster_size"])),
        "eps_quantile": float(mapper["eps_quantile"]),
    }


def _seeds(config: dict[str, Any], smoke: bool) -> list[int]:
    seeds = [int(seed) for seed in config["mapper"]["seeds"]]
    return seeds[:2] if smoke else seeds


def _pool_points(config: dict[str, Any], smoke: bool) -> int:
    points = int(config["mapper"]["pool_points"])
    return min(points, 850) if smoke else points


def _thresholds(config: dict[str, Any]) -> dict[str, float]:
    stable = config["matching"]["stable_track_thresholds"]
    interp = config["interpretability"]["primary_metadata_threshold"]
    return {
        "reference_node_size_min": float(stable["reference_node_size_min"]),
        "recurrence_min": float(stable["recurrence_min"]),
        "median_best_jaccard_min": float(stable["median_best_jaccard_min"]),
        "label_purity_excess_min": float(interp["label_purity_excess_min"]),
    }


def _control_cloud(cloud: TokenCloud, control_kind: str, seed: int) -> TokenCloud:
    rng = np.random.default_rng(seed)
    x = cloud.tokens.astype(np.float32)
    if control_kind == "channel_shuffle":
        tokens = x.copy()
        for dim in range(tokens.shape[1]):
            rng.shuffle(tokens[:, dim])
    elif control_kind == "norm_random_directions":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        directions = rng.normal(size=x.shape).astype(np.float32)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
        tokens = directions * norms
    else:
        raise ValueError(f"Unsupported representation control: {control_kind}")
    return replace(cloud, name=f"{cloud.name}:{control_kind}", tokens=tokens.astype(np.float32))


def _apply_track_flags(table: pd.DataFrame, thresholds: dict[str, float], label_col: str = "label") -> pd.DataFrame:
    out = table.copy()
    purity_col = f"{label_col}_purity_excess"
    if purity_col not in out.columns:
        out[purity_col] = np.nan
    out["passes_size"] = out["reference_size"] >= thresholds["reference_node_size_min"]
    out["passes_recurrence"] = out["recurrence"] >= thresholds["recurrence_min"]
    out["passes_jaccard"] = out["median_best_jaccard"] >= thresholds["median_best_jaccard_min"]
    out["passes_label_enrichment"] = out[purity_col] >= thresholds["label_purity_excess_min"]
    out["is_stable_track"] = out["passes_size"] & out["passes_recurrence"] & out["passes_jaccard"]
    out["is_stable_enriched_track"] = out["is_stable_track"] & out["passes_label_enrichment"]
    return out


def _metadata_shuffle_counts(
    stability: pd.DataFrame,
    reference_run: MapperRun,
    thresholds: dict[str, float],
    n_shuffles: int,
    seed: int,
    label_col: str = "label",
) -> pd.DataFrame:
    if n_shuffles <= 0 or label_col not in reference_run.metadata.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    labels = reference_run.metadata[label_col].to_numpy(copy=True)
    for shuffle_id in range(n_shuffles):
        shuffled = reference_run.metadata.copy()
        shuffled[label_col] = rng.permutation(labels)
        shuffled_run = MapperRun(
            run_id=reference_run.run_id,
            seed=reference_run.seed,
            graph=reference_run.graph,
            x=reference_run.x,
            metadata=shuffled,
            source_indices=reference_run.source_indices,
            lens_name=reference_run.lens_name,
            config=reference_run.config,
        )
        annotated = annotate_stability_with_node_summary(stability, shuffled_run, label_col=label_col)
        flagged = _apply_track_flags(annotated, thresholds, label_col=label_col)
        stable = flagged[flagged["is_stable_track"]]
        rows.append(
            {
                "shuffle_id": shuffle_id,
                "stable_track_count": int(flagged["is_stable_track"].sum()),
                "stable_enriched_track_count": int(flagged["is_stable_enriched_track"].sum()),
                "max_label_purity_excess": float(stable[f"{label_col}_purity_excess"].max()) if not stable.empty else 0.0,
                "mean_label_purity_excess": float(stable[f"{label_col}_purity_excess"].mean()) if not stable.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _count_summary(tracks: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    stable = tracks[tracks["is_stable_track"]]
    enriched = tracks[tracks["is_stable_enriched_track"]]
    row: dict[str, Any] = {
        "reference_nodes": int(len(tracks)),
        "stable_track_count": int(len(stable)),
        "stable_enriched_track_count": int(len(enriched)),
        "median_best_jaccard_all": float(tracks["median_best_jaccard"].median()) if len(tracks) else 0.0,
        "median_best_jaccard_stable": float(stable["median_best_jaccard"].median()) if len(stable) else 0.0,
        "mean_reference_size_stable": float(stable["reference_size"].mean()) if len(stable) else 0.0,
    }
    purity_col = f"{label_col}_purity_excess"
    if purity_col in stable.columns:
        row["mean_label_purity_excess_stable"] = float(stable[purity_col].mean()) if len(stable) else 0.0
    if "dominant_image_fraction" in stable.columns:
        row["mean_dominant_image_fraction_stable"] = float(stable["dominant_image_fraction"].mean()) if len(stable) else 0.0
    if "spatial_radius" in stable.columns:
        row["mean_spatial_radius_stable"] = float(stable["spatial_radius"].mean()) if len(stable) else 0.0
    return row


def _safe_write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _table_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No rows._"
    text = table.fillna("").astype(str)
    headers = list(text.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in headers) + " |")
    return "\n".join(lines)


def _condition_run(
    dataset: str,
    representation: str,
    sample_kind: str,
    baseline_group: str,
    cloud: TokenCloud,
    config: dict[str, Any],
    mapper_config: dict[str, Any],
    thresholds: dict[str, float],
    smoke: bool,
    metadata_shuffles: int,
    output_dir: Path,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seeds = _seeds(config, smoke)
    mapper = config["mapper"]
    t0 = time.perf_counter()
    runs = build_mapper_stability_runs(
        cloud,
        seeds=seeds,
        lens_name=str(mapper["lens"]),
        config=mapper_config,
        pool_points=_pool_points(config, smoke),
        sample_fraction=float(mapper["sample_fraction"]),
        pool_seed=seeds[0],
    )
    run_stats = summarize_stability_runs(runs)
    run_stats.insert(0, "baseline_group", baseline_group)
    run_stats.insert(0, "sample_kind", sample_kind)
    run_stats.insert(0, "representation", representation)
    run_stats.insert(0, "dataset", dataset)

    reference_run = sorted(runs)[0]
    stability, matches = reference_node_stability(
        runs,
        reference_run=reference_run,
        min_jaccard=0.05,
        matching="hungarian",
    )
    tracks = annotate_stability_with_node_summary(stability, runs[reference_run], label_col=label_col)
    tracks = _apply_track_flags(tracks, thresholds, label_col=label_col)
    tracks.insert(0, "baseline_group", baseline_group)
    tracks.insert(0, "sample_kind", sample_kind)
    tracks.insert(0, "representation", representation)
    tracks.insert(0, "dataset", dataset)
    tracks["reference_run"] = reference_run

    if not matches.empty:
        matches.insert(0, "baseline_group", baseline_group)
        matches.insert(0, "sample_kind", sample_kind)
        matches.insert(0, "representation", representation)
        matches.insert(0, "dataset", dataset)
    else:
        matches = pd.DataFrame(
            columns=["dataset", "representation", "sample_kind", "baseline_group", "run_a", "run_b", "node_a", "node_b", "jaccard"]
        )

    shuffle_table = _metadata_shuffle_counts(
        stability,
        runs[reference_run],
        thresholds,
        n_shuffles=metadata_shuffles,
        seed=seeds[0] + 1009,
        label_col=label_col,
    )
    if not shuffle_table.empty:
        shuffle_table.insert(0, "baseline_group", baseline_group)
        shuffle_table.insert(0, "sample_kind", sample_kind)
        shuffle_table.insert(0, "representation", representation)
        shuffle_table.insert(0, "dataset", dataset)

    atlas_table = pd.DataFrame()
    if dataset == "beans_local" and sample_kind in {"observed", "raw_patches"}:
        candidates = tracks[tracks["is_stable_enriched_track"]].copy()
        if candidates.empty:
            candidates = tracks[tracks["is_stable_track"]].copy()
        sort_cols = [
            col
            for col in [
                "is_stable_enriched_track",
                "recurrence",
                "median_best_jaccard",
                "label_purity_excess",
                "reference_size",
            ]
            if col in candidates.columns
        ]
        if not candidates.empty and sort_cols:
            candidates = candidates.sort_values(sort_cols, ascending=False).head(10)
            node_ids = candidates["reference_node_id"].astype(int).tolist()
            atlas_table = patch_atlas_records(runs[reference_run].graph, cloud, node_ids, patches_per_node=6)
            if not atlas_table.empty:
                atlas_table = atlas_table.rename(columns={"node_id": "reference_node_id"})
                context_cols = [
                    "reference_node_id",
                    "recurrence",
                    "median_best_jaccard",
                    "reference_size",
                    "label_purity",
                    "label_purity_excess",
                    "dominant_image_fraction",
                    "spatial_radius",
                    "is_stable_track",
                    "is_stable_enriched_track",
                ]
                context_cols = [col for col in context_cols if col in tracks.columns]
                atlas_table = atlas_table.merge(tracks[context_cols], on="reference_node_id", how="left")
                atlas_table.insert(0, "baseline_group", baseline_group)
                atlas_table.insert(0, "sample_kind", sample_kind)
                atlas_table.insert(0, "representation", representation)
                atlas_table.insert(0, "dataset", dataset)

    counts = _count_summary(tracks, label_col=label_col)
    counts.update(
        {
            "dataset": dataset,
            "representation": representation,
            "sample_kind": sample_kind,
            "baseline_group": baseline_group,
            "reference_run": reference_run,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }
    )
    if not shuffle_table.empty:
        counts["metadata_shuffle_stable_enriched_count_mean"] = float(shuffle_table["stable_enriched_track_count"].mean())
        counts["metadata_shuffle_stable_enriched_count_q95"] = float(shuffle_table["stable_enriched_track_count"].quantile(0.95))
        counts["metadata_shuffle_win"] = bool(counts["stable_enriched_track_count"] > counts["metadata_shuffle_stable_enriched_count_q95"])
    count_table = pd.DataFrame([counts])

    condition_dir = output_dir / "conditions" / dataset / sample_kind
    _safe_write_csv(run_stats, condition_dir / "run_stats.csv")
    _safe_write_csv(tracks, condition_dir / "tracks.csv")
    _safe_write_csv(matches, condition_dir / "matches.csv")
    if not shuffle_table.empty:
        _safe_write_csv(shuffle_table, condition_dir / "metadata_shuffle_counts.csv")
    if not atlas_table.empty:
        _safe_write_csv(atlas_table, condition_dir / "patch_atlas_records.csv")
    _safe_write_csv(count_table, condition_dir / "counts.csv")

    return run_stats, tracks, matches, count_table, shuffle_table, atlas_table


def _paired_primary(counts: pd.DataFrame, config: dict[str, Any], bootstrap_samples: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    control_kinds = list(config["controls"].get("representation_controls", []))
    primary_rows = []
    target_rows = counts[(counts["representation"] == "flux_vae") & (counts["baseline_group"].isin(["observed", "representation_control"]))]
    for dataset, group in target_rows.groupby("dataset"):
        observed = group[group["sample_kind"] == "observed"]
        controls = group[group["sample_kind"].isin(control_kinds)]
        if observed.empty:
            continue
        observed_count = int(observed["stable_enriched_track_count"].iloc[0])
        max_control_count = int(controls["stable_enriched_track_count"].max()) if not controls.empty else 0
        hardest_control = (
            str(controls.sort_values("stable_enriched_track_count", ascending=False)["sample_kind"].iloc[0])
            if not controls.empty
            else "none"
        )
        metadata_q95 = float(observed.get("metadata_shuffle_stable_enriched_count_q95", pd.Series([np.nan])).iloc[0])
        primary_rows.append(
            {
                "dataset": dataset,
                "observed_count": observed_count,
                "max_control_count": max_control_count,
                "hardest_control": hardest_control,
                "stable_enriched_track_count_delta": observed_count - max_control_count,
                "metadata_shuffle_q95": metadata_q95,
                "beats_metadata_shuffle": bool(observed_count > metadata_q95) if not np.isnan(metadata_q95) else False,
            }
        )
    paired = pd.DataFrame(primary_rows).sort_values("dataset").reset_index(drop=True)
    if paired.empty:
        verdict = {
            "status": "insufficient_data",
            "reason": "no paired observed/control rows",
        }
        return paired, verdict

    deltas = paired["stable_enriched_track_count_delta"].to_numpy(dtype=float)
    rng = np.random.default_rng(72)
    if bootstrap_samples > 0 and len(deltas) > 0:
        boot = rng.choice(deltas, size=(bootstrap_samples, len(deltas)), replace=True).mean(axis=1)
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    else:
        ci_low = ci_high = float("nan")
    positive_datasets = int((deltas > 0).sum())
    mean_delta = float(np.mean(deltas))
    metadata_wins = int(paired["beats_metadata_shuffle"].sum())
    pass_rule = bool(mean_delta > 0 and ci_low > 0 and positive_datasets >= 2)
    verdict = {
        "status": "pass" if pass_rule else "fail",
        "mean_delta": mean_delta,
        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "positive_datasets": positive_datasets,
        "n_datasets": int(len(paired)),
        "metadata_shuffle_wins": metadata_wins,
        "primary_statistic": config["primary_statistic"],
        "primary_question": config["primary_question"],
        "decision_rule": config["decision_rule"],
    }
    return paired, verdict


def _write_summary(
    output_dir: Path,
    config: dict[str, Any],
    counts: pd.DataFrame,
    paired: pd.DataFrame,
    verdict: dict[str, Any],
) -> None:
    lines = [
        f"# {config['name']} Summary",
        "",
        f"Status: **{verdict.get('status', 'unknown')}**",
        "",
        f"Primary question: {config['primary_question']}",
        "",
        f"Primary statistic: `{config['primary_statistic']}`",
        "",
    ]
    if paired.empty:
        lines.append("No paired primary rows were available.")
    else:
        lines.extend(
            [
                "## Primary Readout",
                "",
                _table_to_markdown(paired),
                "",
                "## Verdict Math",
                "",
                f"- Mean delta: `{verdict['mean_delta']:.3f}`",
                f"- Bootstrap 95% CI: `[{verdict['bootstrap_ci_95'][0]:.3f}, {verdict['bootstrap_ci_95'][1]:.3f}]`",
                f"- Positive datasets: `{verdict['positive_datasets']} / {verdict['n_datasets']}`",
                f"- Observed beats metadata-shuffle q95: `{verdict['metadata_shuffle_wins']} / {verdict['n_datasets']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Counts By Condition",
            "",
            _table_to_markdown(
                counts[
                    [
                        "dataset",
                        "representation",
                        "sample_kind",
                        "baseline_group",
                        "stable_track_count",
                        "stable_enriched_track_count",
                        "metadata_shuffle_stable_enriched_count_q95",
                    ]
                ]
            ),
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2))

    mapper_config = _mapper_config(config, args.smoke)
    thresholds = _thresholds(config)
    metadata_shuffles = min(args.metadata_shuffles, 8) if args.smoke else args.metadata_shuffles
    bootstrap_samples = min(args.bootstrap_samples, 1000) if args.smoke else args.bootstrap_samples

    clouds = load_cached_flux_clouds()
    target_datasets = [str(name) for name in config["target_object"]["datasets"]]
    available = [dataset for dataset in target_datasets if dataset in clouds]
    missing = sorted(set(target_datasets) - set(available))
    if not available:
        raise FileNotFoundError(f"No requested FLUX caches are available. Missing: {missing}")
    if args.smoke:
        available = available[:1]

    all_run_stats = []
    all_tracks = []
    all_matches = []
    all_counts = []
    all_shuffles = []
    all_atlas = []

    print(f"Running {config['name']} on datasets: {available}")
    if missing:
        print(f"Skipping missing datasets: {missing}")

    control_kinds = list(config["controls"].get("representation_controls", []))
    for dataset in available:
        cloud = clouds[dataset]
        conditions = [("observed", "observed", cloud)]
        for i, control_kind in enumerate(control_kinds):
            conditions.append(
                (
                    control_kind,
                    "representation_control",
                    _control_cloud(cloud, control_kind=control_kind, seed=10_000 + i),
                )
            )
        for sample_kind, baseline_group, condition_cloud in conditions:
            print(f"  {dataset} / {sample_kind}")
            run_stats, tracks, matches, counts, shuffles, atlas = _condition_run(
                dataset=dataset,
                representation="flux_vae",
                sample_kind=sample_kind,
                baseline_group=baseline_group,
                cloud=condition_cloud,
                config=config,
                mapper_config=mapper_config,
                thresholds=thresholds,
                smoke=args.smoke,
                metadata_shuffles=metadata_shuffles,
                output_dir=output_dir,
            )
            all_run_stats.append(run_stats)
            all_tracks.append(tracks)
            all_matches.append(matches)
            all_counts.append(counts)
            if not shuffles.empty:
                all_shuffles.append(shuffles)
            if not atlas.empty:
                all_atlas.append(atlas)

    if not args.smoke and "raw_patches_beans_local" in config["controls"].get("specificity_baselines", []):
        raw_cloud = maybe_load_raw_patch_cloud(n_images=48, image_size=args.image_size, image_dir=args.image_dir)
        if raw_cloud is not None:
            print("  beans_local / raw_patches")
            run_stats, tracks, matches, counts, shuffles, atlas = _condition_run(
                dataset="beans_local",
                representation="raw_patches",
                sample_kind="raw_patches",
                baseline_group="specificity_baseline",
                cloud=raw_cloud,
                config=config,
                mapper_config=mapper_config,
                thresholds=thresholds,
                smoke=args.smoke,
                metadata_shuffles=metadata_shuffles,
                output_dir=output_dir,
            )
            all_run_stats.append(run_stats)
            all_tracks.append(tracks)
            all_matches.append(matches)
            all_counts.append(counts)
            if not shuffles.empty:
                all_shuffles.append(shuffles)
            if not atlas.empty:
                all_atlas.append(atlas)

    run_stats = pd.concat(all_run_stats, ignore_index=True) if all_run_stats else pd.DataFrame()
    tracks = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    counts = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()
    shuffles = pd.concat(all_shuffles, ignore_index=True) if all_shuffles else pd.DataFrame()
    atlas = pd.concat(all_atlas, ignore_index=True) if all_atlas else pd.DataFrame()

    paired, verdict = _paired_primary(counts, config, bootstrap_samples=bootstrap_samples)

    _safe_write_csv(run_stats, output_dir / "run_stats.csv")
    _safe_write_csv(tracks, output_dir / "tracks.csv")
    _safe_write_csv(matches, output_dir / "matches.csv")
    _safe_write_csv(counts, output_dir / "counts.csv")
    _safe_write_csv(paired, output_dir / "paired_primary.csv")
    if not shuffles.empty:
        _safe_write_csv(shuffles, output_dir / "metadata_shuffle_counts.csv")
    if not atlas.empty:
        _safe_write_csv(atlas, output_dir / "patch_atlas_records.csv")
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    _write_summary(output_dir, config, counts, paired, verdict)
    print(f"Wrote {output_dir.relative_to(ROOT)}")
    print(f"Verdict: {verdict.get('status')} | mean delta={verdict.get('mean_delta')}")


if __name__ == "__main__":
    main()
