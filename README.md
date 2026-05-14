# Latent Space Topology

Exploratory notebooks on the geometry and topology of image encoder latents,
with a current focus on FLUX-style VAE spatial tokens.

The current experiment treats each spatial latent vector as a point-cloud
sample, compares raw and normalized views, adds simple controls, and then runs
persistent homology as a diagnostic rather than as a final claim about "the"
topology of the model.

For context, see [project_note.md](project_note.md) and
[literature_review.md](literature_review.md).

## Notebook Map

The notebooks are committed with executed cells so collaborators can inspect
the current run without recomputing model weights.

| Notebook | Purpose |
| --- | --- |
| `03_understand_tokenizers_encoders.ipynb` | Compare image tokenizer and encoder families. |
| `04_geometric_intuition_before_topology.ipynb` | Check latent norms, PCA, density, and spatial effects before TDA. |
| `05_flux_encoder_deep_dive.ipynb` | Focus on FLUX VAE token geometry and nearest-neighbor structure. |
| `06_flux_tda_exploration.ipynb` | Run the first persistent-homology probe with controls. |
| `07_stability_and_artifact_controls.ipynb` | Repeat the TDA pipeline across seeds, views, knobs, and controls. |
| `08_representation_comparison.ipynb` | Compare FLUX, SD VAE, VQ, ViT, CLIP, and raw-patch token clouds. |
| `09_interpreting_persistent_features.ipynb` | Map selected FLUX persistent-feature supports back to image patches. |
| `10_natural_patch_baseline.ipynb` | Build a raw natural-image-patch baseline with controls. |
| `11_metric_preprocessing_sensitivity.ipynb` | Test how FLUX summaries change under metric and preprocessing choices. |
| `12_cycle_hunt_long_lived_h1.ipynb` | Summarize the search for the claimed long-lived FLUX H1 cycle. |
| `13_confirmatory_h1_sweep.ipynb` | Report the held-out confirmatory H1 sweep and verdict. |
| `14_mapper_flux_intuition.ipynb` | Consolidated Mapper exploration: intuition, controls, node interpretation, dataset checks, raw-patch baseline, and synthesis. |

## Installation

This project uses Python 3.12 and `uv`.

```bash
git clone git@github.com:vitalylativ/latent-space-topology.git
cd latent-space-topology
uv sync
```

The notebooks download model weights from Hugging Face on first use. On Apple
Silicon or CUDA machines they try to use the available accelerator. Set
`TOKENIZER_FORCE_CPU=1` to force CPU execution.

## Data

Local data is intentionally not committed. The notebooks look for images in
`data/images/beans` by default and fall back to the Hugging Face `beans` dataset
if that folder is missing.

To create the local data folder used by the notebooks:

```bash
./scripts/download_beans_data.sh
```

Example with a larger subset:

```bash
TOKENIZER_N_IMAGES=128 ./scripts/download_beans_data.sh
```

The script writes images plus `data/images/beans/metadata.csv`, which is the
format expected by `notebook_utils.encoder_explorer.load_project_images`.

## Running

Start Jupyter:

```bash
uv run jupyter lab
```

To rerun a quick topology smoke pass from the terminal:

```bash
TOKENIZER_SMOKE=1 uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/06_flux_tda_exploration.ipynb \
  --inplace
```

Notebook source builders are kept in `scripts/`. For example:

```bash
uv run python scripts/build_flux_tda_notebook.py
uv run python scripts/build_mapper_intuition_notebook.py
```
