# Latent Space Topology

Exploratory notebooks for studying the geometry and topology of image encoder
latents, with a current focus on FLUX-style VAE spatial tokens. The project
treats each spatial latent vector as a point cloud sample, then compares raw,
normalized, density-filtered, and control clouds before running persistent
homology.

The longer motivation and caveats live in [project_note.md](project_note.md);
the literature map is in [literature_review.md](literature_review.md).

## Installation

This project uses Python 3.12 and `uv`.

```bash
git clone git@github.com:vitalylativ/latent-space-topology.git
cd latent-space-topology
uv sync
```

The notebooks download model weights from Hugging Face on first use. On Apple
Silicon or CUDA machines they will try to use the available accelerator; set
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

## Running Notebooks

Start Jupyter:

```bash
uv run jupyter lab
```

For a quick topology smoke run:

```bash
TOKENIZER_SMOKE=1 uv run jupyter nbconvert \
  --to notebook \
  --execute notebooks/06_flux_tda_exploration.ipynb \
  --inplace
```

Notebook source builders are kept in `scripts/`. For example:

```bash
uv run python scripts/build_flux_tda_notebook.py
```
