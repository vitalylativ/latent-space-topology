"""Run a witness-complex version of the FLUX latent H1 cycle hunt.

This script matches the pipeline described by Vitaly's collaborator more closely
than the Rips notebooks:

1. take FLUX VAE spatial tokens shaped as 16-dimensional vectors;
2. project all tokens to the unit sphere S^15;
3. keep the densest tokens as witnesses;
4. choose 30-35 landmarks from those dense witnesses;
5. build a GUDHI weak witness complex and inspect H1 persistence.

The default preset is intentionally small enough to run on a laptop. Use
``--preset full`` to get closer to the collaborator's 10k-20k witness setup.
Outputs are written under outputs/cycle_hunt/witness_pipeline/, which is
ignored by Git. The script can consume cached FLUX token clouds from previous
sweeps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebook_utils.encoder_explorer import l2_normalize, sample_indices
from notebook_utils.flux_tda import farthest_point_indices


DEFAULT_CACHE_DIR = ROOT / "outputs" / "cycle_hunt" / "data_sweep" / "cache"
OUT_DIR = ROOT / "outputs" / "cycle_hunt" / "witness_pipeline"

PRESETS = {
    "smoke": {
        "datasets": "beans_local",
        "witnesses": "1000",
        "landmarks": "30",
        "seeds": "72",
        "controls": "random_tokens",
        "density_mode": "anchors",
        "density_probes": 512,
        "max_candidates": 4000,
    },
    "local": {
        "datasets": "beans_local",
        "witnesses": "1000,2000,5000",
        "landmarks": "30,35",
        "seeds": "72,73",
        "controls": "random_tokens,uniform_sphere",
        "density_mode": "anchors",
        "density_probes": 1024,
        "max_candidates": 8000,
    },
    "medium": {
        "datasets": "beans_local,cifar10",
        "witnesses": "5000,10000",
        "landmarks": "30,35",
        "seeds": "72,73,74",
        "controls": "random_tokens,uniform_sphere,channel_shuffle",
        "density_mode": "anchors",
        "density_probes": 2048,
        "max_candidates": 20000,
    },
    "full": {
        "datasets": "beans_local,cifar10,fashion_mnist",
        "witnesses": "10000,20000",
        "landmarks": "30,35",
        "seeds": "72,73,74,75,76",
        "controls": "random_tokens,uniform_sphere,channel_shuffle",
        "density_mode": "anchors",
        "density_probes": 4096,
        "max_candidates": 50000,
    },
}

CONTROL_KINDS = {"random_tokens", "uniform_sphere", "channel_shuffle"}


@dataclass
class TokenCloudLite:
    name: str
    tokens: np.ndarray
    grid_shape: tuple[int, int]
    channel_dim: int


def load_cached_cloud(path: Path, name: str | None = None) -> TokenCloudLite:
    data = np.load(path)
    tokens = data["tokens"].astype(np.float32)
    grid_shape = tuple(int(x) for x in data["grid_shape"])
    channel_dim = int(data["channel_dim"][0])
    return TokenCloudLite(name=name or path.stem, tokens=tokens, grid_shape=grid_shape, channel_dim=channel_dim)


def kth_density_order_exact(x: np.ndarray, k: int, max_candidates: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    candidate_idx = sample_indices(len(x), min(max_candidates, len(x)), seed=seed)
    candidates = x[candidate_idx]
    n_neighbors = min(k + 1, len(candidates))
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(candidates).kneighbors(candidates)
    kth = distances[:, -1]
    order = np.argsort(kth)
    return candidate_idx[order], kth[order]


def kth_density_order_anchors(
    x: np.ndarray,
    k: int,
    max_candidates: int,
    seed: int,
    n_probes: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_idx = sample_indices(len(x), min(max_candidates, len(x)), seed=seed)
    candidates = x[candidate_idx].astype(np.float32, copy=False)
    probe_count = min(n_probes, len(candidates))
    probe_local = sample_indices(len(candidates), probe_count, seed=seed + 997)
    probes = candidates[probe_local].astype(np.float32, copy=False)
    probe_norms = np.sum(probes * probes, axis=1)
    kth_index = min(k, probe_count - 1)
    kth_sq = np.empty(len(candidates), dtype=np.float32)

    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        chunk = candidates[start:stop]
        chunk_norms = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances_sq = np.maximum(chunk_norms + probe_norms[None, :] - 2.0 * (chunk @ probes.T), 0.0)
        kth_sq[start:stop] = np.partition(distances_sq, kth_index, axis=1)[:, kth_index]

    order = np.argsort(kth_sq)
    return candidate_idx[order], np.sqrt(kth_sq[order])


def kth_density_order(
    x: np.ndarray,
    k: int,
    max_candidates: int,
    seed: int,
    density_mode: str,
    density_probes: int,
    density_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if density_mode == "exact":
        return kth_density_order_exact(x, k=k, max_candidates=max_candidates, seed=seed)
    if density_mode == "anchors":
        return kth_density_order_anchors(
            x,
            k=k,
            max_candidates=max_candidates,
            seed=seed,
            n_probes=density_probes,
            chunk_size=density_chunk_size,
        )
    raise ValueError(f"unknown density mode: {density_mode}")


def dense_witness_landmarks(
    x: np.ndarray,
    n_witnesses: int,
    n_landmarks: int,
    k_density: int,
    max_candidates: int,
    seed: int,
    landmark_strategy: str,
    density_mode: str,
    density_probes: int,
    density_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    density_order, kth = kth_density_order(
        x,
        k=k_density,
        max_candidates=max_candidates,
        seed=seed,
        density_mode=density_mode,
        density_probes=density_probes,
        density_chunk_size=density_chunk_size,
    )
    witness_idx = density_order[: min(n_witnesses, len(density_order))]
    witnesses = x[witness_idx]

    if landmark_strategy == "fps":
        local_landmarks = farthest_point_indices(witnesses, n_landmarks=n_landmarks, seed=seed)
    elif landmark_strategy == "random":
        local_landmarks = sample_indices(len(witnesses), min(n_landmarks, len(witnesses)), seed=seed)
    elif landmark_strategy == "density_top":
        local_landmarks = np.arange(min(n_landmarks, len(witnesses)))
    else:
        raise ValueError(f"unknown landmark strategy: {landmark_strategy}")

    landmark_idx = witness_idx[local_landmarks]
    notes = {
        "dense_kth_median": float(np.median(kth[: len(witness_idx)])),
        "dense_kth_max": float(np.max(kth[: len(witness_idx)])),
        "candidate_count": int(len(density_order)),
        "density_mode": density_mode,
        "density_probes": int(density_probes),
    }
    return witness_idx, landmark_idx, notes


def weak_witness_h1(
    witnesses: np.ndarray,
    landmarks: np.ndarray,
    max_alpha_square: float,
    limit_dimension: int = 2,
) -> dict[str, Any]:
    import gudhi

    complex_ = gudhi.EuclideanWitnessComplex(landmarks=landmarks, witnesses=witnesses)
    st = complex_.create_simplex_tree(max_alpha_square=max_alpha_square, limit_dimension=limit_dimension)
    st.compute_persistence()
    h1 = np.asarray(st.persistence_intervals_in_dimension(1), dtype=float)
    finite = h1[np.isfinite(h1[:, 1])] if len(h1) else np.empty((0, 2))
    infinite = h1[~np.isfinite(h1[:, 1])] if len(h1) else np.empty((0, 2))
    # GUDHI does not expose max filtration directly; compute it from filtration iterator.
    filtration_values = [float(value) for _, value in st.get_filtration()]
    filtration_max = max(filtration_values) if filtration_values else 0.0
    capped_deaths = np.full(len(infinite), filtration_max)
    finite_persistence = finite[:, 1] - finite[:, 0] if len(finite) else np.array([])
    capped_persistence = capped_deaths - infinite[:, 0] if len(infinite) else np.array([])
    all_persistence = np.concatenate([finite_persistence, capped_persistence]) if len(capped_persistence) else finite_persistence

    sqrt_finite_persistence = (
        np.sqrt(np.maximum(finite[:, 1], 0.0)) - np.sqrt(np.maximum(finite[:, 0], 0.0))
        if len(finite)
        else np.array([])
    )
    sqrt_capped_persistence = (
        np.sqrt(np.maximum(capped_deaths, 0.0)) - np.sqrt(np.maximum(infinite[:, 0], 0.0))
        if len(infinite)
        else np.array([])
    )
    sqrt_all_persistence = (
        np.concatenate([sqrt_finite_persistence, sqrt_capped_persistence])
        if len(sqrt_capped_persistence)
        else sqrt_finite_persistence
    )
    sqrt_filtration_max = float(np.sqrt(max(filtration_max, 0.0)))

    return {
        "num_vertices": st.num_vertices(),
        "num_simplices": st.num_simplices(),
        "filtration_max_alpha_square": filtration_max,
        "filtration_max_alpha": sqrt_filtration_max,
        "h1_finite_count": int(len(finite)),
        "h1_infinite_count": int(len(infinite)),
        "h1_max_persistence_alpha_square": float(all_persistence.max()) if len(all_persistence) else 0.0,
        "h1_top3_persistence_alpha_square": float(np.sort(all_persistence)[-3:].sum()) if len(all_persistence) else 0.0,
        "h1_max_persistence_alpha": float(sqrt_all_persistence.max()) if len(sqrt_all_persistence) else 0.0,
        "h1_top3_persistence_alpha": float(np.sort(sqrt_all_persistence)[-3:].sum()) if len(sqrt_all_persistence) else 0.0,
        "h1_max_fraction_alpha": float(sqrt_all_persistence.max() / sqrt_filtration_max) if len(sqrt_all_persistence) and sqrt_filtration_max > 0 else 0.0,
    }


def shuffled_columns(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = x.copy()
    for j in range(y.shape[1]):
        rng.shuffle(y[:, j])
    return y


def uniform_sphere(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=(n, dim)).astype(np.float32)).astype(np.float32)


def run_condition(
    cloud: TokenCloudLite,
    dataset: str,
    n_witnesses: int,
    n_landmarks: int,
    k_density: int,
    max_candidates: int,
    seed: int,
    landmark_strategy: str,
    max_alpha_square: float,
    density_mode: str,
    density_probes: int,
    density_chunk_size: int,
    controls: set[str],
) -> list[dict[str, Any]]:
    sphere = l2_normalize(cloud.tokens.astype(np.float32)).astype(np.float32)
    witness_idx, landmark_idx, notes = dense_witness_landmarks(
        sphere,
        n_witnesses=n_witnesses,
        n_landmarks=n_landmarks,
        k_density=k_density,
        max_candidates=max_candidates,
        seed=seed,
        landmark_strategy=landmark_strategy,
        density_mode=density_mode,
        density_probes=density_probes,
        density_chunk_size=density_chunk_size,
    )

    samples = {
        "observed": (sphere[witness_idx], sphere[landmark_idx]),
    }

    if "random_tokens" in controls:
        random_idx = sample_indices(len(sphere), len(witness_idx), seed=seed + 101)
        random_witnesses = sphere[random_idx]
        random_landmark_local = farthest_point_indices(random_witnesses, n_landmarks=n_landmarks, seed=seed + 101)
        samples["random_tokens"] = (random_witnesses, random_witnesses[random_landmark_local])

    if "uniform_sphere" in controls:
        uniform_witnesses = uniform_sphere(len(witness_idx), sphere.shape[1], seed=seed + 102)
        uniform_landmark_local = farthest_point_indices(uniform_witnesses, n_landmarks=n_landmarks, seed=seed + 102)
        samples["uniform_sphere"] = (uniform_witnesses, uniform_witnesses[uniform_landmark_local])

    if "channel_shuffle" in controls:
        shuffled = shuffled_columns(sphere, seed=seed + 103)
        sh_witness_idx, sh_landmark_idx, _ = dense_witness_landmarks(
            shuffled,
            n_witnesses=n_witnesses,
            n_landmarks=n_landmarks,
            k_density=k_density,
            max_candidates=max_candidates,
            seed=seed + 103,
            landmark_strategy=landmark_strategy,
            density_mode=density_mode,
            density_probes=density_probes,
            density_chunk_size=density_chunk_size,
        )
        samples["channel_shuffle"] = (shuffled[sh_witness_idx], shuffled[sh_landmark_idx])

    rows = []
    for sample_kind, (witnesses, landmarks) in samples.items():
        started = time.perf_counter()
        result = weak_witness_h1(
            witnesses=witnesses,
            landmarks=landmarks,
            max_alpha_square=max_alpha_square,
            limit_dimension=2,
        )
        rows.append(
            {
                "dataset": dataset,
                "sample_kind": sample_kind,
                "seed": seed,
                "n_tokens": len(cloud.tokens),
                "n_witnesses": len(witnesses),
                "n_landmarks": len(landmarks),
                "k_density": k_density,
                "max_candidates": max_candidates,
                "landmark_strategy": landmark_strategy,
                "max_alpha_square": max_alpha_square,
                "runtime_seconds": time.perf_counter() - started,
                **notes,
                **result,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--preset", default="local", choices=sorted(PRESETS))
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--witnesses", default=None)
    parser.add_argument("--landmarks", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--controls", default=None)
    parser.add_argument("--k-density", type=int, default=16)
    parser.add_argument("--density-mode", default=None, choices=["anchors", "exact"])
    parser.add_argument("--density-probes", type=int, default=None)
    parser.add_argument("--density-chunk-size", type=int, default=2048)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--landmark-strategy", default="fps", choices=["fps", "random", "density_top"])
    parser.add_argument("--max-alpha-square", type=float, default=float("inf"))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    for key in (
        "datasets",
        "witnesses",
        "landmarks",
        "seeds",
        "controls",
        "density_mode",
        "density_probes",
        "max_candidates",
    ):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    return args


def parse_csv_set(value: str, allowed: set[str]) -> set[str]:
    requested = {part.strip() for part in value.split(",") if part.strip()}
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"unknown controls: {sorted(unknown)}; choose from {sorted(allowed)}")
    return requested


def main() -> None:
    args = apply_preset(parse_args())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    witness_counts = [int(part) for part in args.witnesses.split(",") if part.strip()]
    landmark_counts = [int(part) for part in args.landmarks.split(",") if part.strip()]
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    controls = parse_csv_set(args.controls, CONTROL_KINDS)
    print(
        f"preset={args.preset} datasets={datasets} witnesses={witness_counts} "
        f"landmarks={landmark_counts} seeds={seeds} controls={sorted(controls)} "
        f"density={args.density_mode}:{args.density_probes}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        cache_path = args.cache_dir / f"flux_vae_{dataset}_n48_px256.npz"
        if not cache_path.exists() and dataset == "beans_local":
            alt = ROOT / "outputs" / "cycle_hunt" / "beans_param_sweep" / "flux_cloud_48_images.npz"
            cache_path = alt if alt.exists() else cache_path
        if not cache_path.exists():
            print(f"skip {dataset}: missing cache {cache_path}")
            continue
        cloud = load_cached_cloud(cache_path, name=dataset)
        print(f"dataset={dataset} tokens={cloud.tokens.shape}", flush=True)
        for n_witnesses in witness_counts:
            for n_landmarks in landmark_counts:
                for seed in seeds:
                    print(f"  witnesses={n_witnesses} landmarks={n_landmarks} seed={seed}", flush=True)
                    rows.extend(
                        run_condition(
                            cloud,
                            dataset=dataset,
                            n_witnesses=n_witnesses,
                            n_landmarks=n_landmarks,
                            k_density=args.k_density,
                            max_candidates=args.max_candidates,
                            seed=seed,
                            landmark_strategy=args.landmark_strategy,
                            max_alpha_square=args.max_alpha_square,
                            density_mode=args.density_mode,
                            density_probes=args.density_probes,
                            density_chunk_size=args.density_chunk_size,
                            controls=controls,
                        )
                    )

    if not rows:
        raise SystemExit("no runs completed; check cache paths and dataset names")

    df = pd.DataFrame(rows)
    raw_path = args.out_dir / "witness_pipeline_runs.csv"
    df.to_csv(raw_path, index=False)

    observed = df[df["sample_kind"] == "observed"].copy()
    controls = df[df["sample_kind"] != "observed"].copy()
    control_max = (
        controls.groupby(["dataset", "seed", "n_witnesses", "n_landmarks"], as_index=False)["h1_max_fraction_alpha"]
        .max()
        .rename(columns={"h1_max_fraction_alpha": "best_control_h1_fraction_alpha"})
    )
    paired = observed.merge(control_max, on=["dataset", "seed", "n_witnesses", "n_landmarks"], how="left")
    paired["delta_vs_best_control"] = paired["h1_max_fraction_alpha"] - paired["best_control_h1_fraction_alpha"]
    paired_path = args.out_dir / "witness_pipeline_paired.csv"
    paired.to_csv(paired_path, index=False)

    aggregate = (
        paired.groupby(["dataset", "n_witnesses", "n_landmarks"], as_index=False)
        .agg(
            observed_mean=("h1_max_fraction_alpha", "mean"),
            observed_max=("h1_max_fraction_alpha", "max"),
            control_mean=("best_control_h1_fraction_alpha", "mean"),
            delta_mean=("delta_vs_best_control", "mean"),
            delta_max=("delta_vs_best_control", "max"),
            win_rate=("delta_vs_best_control", lambda x: float((x > 0).mean())),
            runs=("delta_vs_best_control", "size"),
        )
        .sort_values("delta_mean", ascending=False)
    )
    aggregate_path = args.out_dir / "witness_pipeline_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    summary_path = args.out_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Witness Pipeline H1 Summary\n\n")
        handle.write(
            "Pipeline: sphere-normalized FLUX tokens -> dense witnesses -> 30/35 FPS landmarks -> weak witness complex.\n\n"
        )
        handle.write(
            f"Preset: `{args.preset}`. Density mode: `{args.density_mode}` "
            f"with {args.density_probes} probes over at most {args.max_candidates} candidates.\n\n"
        )
        handle.write("## Aggregate observed vs controls\n\n")
        handle.write(aggregate.to_string(index=False))
        handle.write("\n\n## Top observed paired runs\n\n")
        top_cols = [
            "dataset",
            "seed",
            "n_witnesses",
            "n_landmarks",
            "h1_max_fraction_alpha",
            "best_control_h1_fraction_alpha",
            "delta_vs_best_control",
            "h1_finite_count",
            "h1_infinite_count",
            "num_simplices",
        ]
        handle.write(paired.sort_values("h1_max_fraction_alpha", ascending=False)[top_cols].head(20).to_string(index=False))
        handle.write("\n")

    config_path = args.out_dir / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": datasets,
                "witness_counts": witness_counts,
                "landmark_counts": landmark_counts,
                "seeds": seeds,
                "preset": args.preset,
                "controls": sorted(controls),
                "k_density": args.k_density,
                "density_mode": args.density_mode,
                "density_probes": args.density_probes,
                "density_chunk_size": args.density_chunk_size,
                "max_candidates": args.max_candidates,
                "landmark_strategy": args.landmark_strategy,
                "max_alpha_square": args.max_alpha_square,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {raw_path}")
    print(f"wrote {paired_path}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {summary_path}")
    print(aggregate.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
