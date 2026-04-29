# Literature Review: Latent-Token Topology

This note collects the literature context separately from the project motivation and execution plan.

## Literature Map

### Natural Image Patches

The closest classical precedent is the topological study of natural image patches by Carlsson, Ishkhanov, de Silva, and Zomorodian: [On the Local Behavior of Spaces of Natural Images](https://www.math.uchicago.edu/~shmuel/AAT-readings/Data%20Analysis%20/mumford-carlsson%20et%20al.pdf).

Why it matters here:

- they studied local image patches as points in a high-dimensional space;
- they normalized patches onto a sphere-like space;
- they selected dense/high-contrast regions;
- they used witness complexes for computational tractability;
- they found structured topology, including a Klein-bottle model for a high-density patch subset.

This is very close in spirit to studying latent spatial tokens. It also warns us that preprocessing is not a detail: contrast selection, normalization, density filtering, and landmark choice are part of the mathematical object being studied.

### TDA For Neural Representations

Several papers use persistent homology or Mapper to analyze activations and representation spaces.

Useful sources:

- [Topology of Deep Neural Networks](https://jmlr.org/beta/papers/v21/20-345.html), Naitzat, Zhitnikov, Lim, JMLR 2020.
- [Topological Data Analysis of Neural Network Layer Representations](https://arxiv.org/abs/2208.06438), Shahidullah 2022.
- [Experimental Observations of the Topology of Convolutional Neural Network Activations](https://arxiv.org/abs/2212.00222), Purvine et al. 2022.
- [Activation Landscapes as a Topological Summary of Neural Network Performance](https://arxiv.org/abs/2110.10136), Wheeler, Bouza, Bubenik 2021.
- [TopoAct](https://arxiv.org/abs/1912.06332), Rathore et al. 2019.

What is known:

- activation point clouds can have measurable topological structure;
- topology can change across layers, training, or adversarial conditions;
- topological summaries can be used as exploratory diagnostics or model fingerprints.

What remains delicate:

- the result depends strongly on what is considered a point;
- flattening a tensor, choosing a metric, or aggregating channels changes the studied object;
- "topological simplification" is not universal and should not be assumed.

### Generative Latent Geometry

There is substantial work arguing that latent spaces of generative models are not simply Euclidean coordinate systems.

Useful sources:

- [Latent Space Oddity](https://arxiv.org/abs/1710.11379), Arvanitidis, Hansen, Hauberg 2018.
- [Explorations in Homeomorphic Variational Auto-Encoding](https://arxiv.org/abs/1807.04689), Falorsi et al. 2018.
- [Topological Autoencoders](https://proceedings.mlr.press/v119/moor20a.html), Moor, Horn, Rieck, Borgwardt 2020.
- [Learning Flat Latent Manifolds with VAEs](https://proceedings.mlr.press/v119/chen20i.html), Chen et al. 2020.
- [The Geometry of Deep Generative Image Models and its Applications](https://arxiv.org/abs/2101.06006), Wang and Ponce 2021.
- [Implications of Data Topology for Deep Generative Models](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1260604/full), Jin et al. 2024.

What is known:

- decoder maps induce non-Euclidean geometry on latent spaces;
- simple Gaussian/Euclidean latents may be topologically mismatched with nontrivial data manifolds;
- autoencoders can be explicitly regularized to preserve topology;
- topology and geometry can affect interpolation, sampling, and generation quality.

Implication for us:

> Euclidean persistent homology of autoencoder tokens is only one view. It should be compared with cosine, normalized, whitened, and possibly decoder-aware metrics.

### Image Tokenizers And Autoencoders

Tokenizer type matters.

Useful sources:

- [VQ-VAE](https://arxiv.org/abs/1711.00937), van den Oord, Vinyals, Kavukcuoglu 2017.
- [VQGAN](https://arxiv.org/abs/2012.09841), Esser, Rombach, Ommer 2021.
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752), Rombach et al. 2022.
- [FLUX official inference repo](https://github.com/black-forest-labs/flux).
- [FLUX.1 VAE diffusers config](https://huggingface.co/diffusers/FLUX.1-vae/blob/main/config.json).

Important distinction:

- VQ-style tokenizers produce discrete codebook indices plus learned codebook embeddings.
- LDM/Stable-Diffusion/FLUX-style autoencoders produce continuous spatial latent tensors.

For FLUX-style latents, the public diffusers config identifies an `AutoencoderKL` with `latent_channels: 16`, scaling and shift factors, and no quantization convolutions. This supports treating the current tensors as continuous autoencoder features, not discrete tokens.

Implication for us:

> The current project is not really about language-like discrete token IDs. It is about the geometry of continuous spatial autoencoder features.

### Statistical Persistent Homology

The statistical literature gives the rules of engagement.

Useful sources:

- [Stability of Persistence Diagrams](https://doi.org/10.1007/s00454-006-1276-5), Cohen-Steiner, Edelsbrunner, Harer 2007.
- [Confidence Sets for Persistence Diagrams](https://arxiv.org/abs/1303.7117), Fasy et al. 2014.
- [Subsampling Methods for Persistent Homology](https://proceedings.mlr.press/v37/chazal15.html), Chazal et al. 2015.
- [Robust Topological Inference: Distance To a Measure and Kernel Distance](https://jmlr.org/beta/papers/v18/15-484.html), Chazal et al. 2018.
- [Statistical Topological Data Analysis using Persistence Landscapes](https://www.jmlr.org/papers/v16/bubenik15a.html), Bubenik 2015.
- [Persistence Images](https://arxiv.org/abs/1507.06217), Adams et al. 2017.
- [Topological Estimation Using Witness Complexes](https://doi.org/10.2312/SPBG/SPBG04/157-166), de Silva and Carlsson 2004.

What is known:

- persistence diagrams are stable under appropriate perturbations of the filtration;
- raw point-cloud topology is sensitive to outliers;
- confidence sets, bootstrap, subsampling, DTM, kernel distance, landscapes, and persistence images give ways to make PH more statistical and less anecdotal.

Implication for us:

> A persistence diagram should be treated as an estimator with uncertainty, not as a picture of truth.

## Synthesis

### What Seems Known

- Local image patches can have nontrivial topology after carefully chosen preprocessing.
- Neural activations and learned representations can be studied as point clouds, but the conclusions depend on the representation object.
- Generative latent spaces have meaningful geometry, but Euclidean geometry is often not the right default.
- Autoencoder/tokenizer training objectives shape latent geometry in ways that are not purely semantic.
- Persistent homology has a solid stability and statistical literature, but only after the filtration, metric, sampling scheme, and target distribution are specified.

### What Seems Unknown

- Whether modern diffusion autoencoder spatial tokens have stable topological structure beyond preprocessing artifacts.
- Whether FLUX-style 16-channel tokens preserve natural-image-patch topology, transform it, or create new geometry.
- Whether token topology distinguishes tokenizers with similar reconstruction scores.
- Whether topological features correlate with visual semantics, reconstruction error, generation failures, prompt sensitivity, or data-domain shifts.
- Whether latent-token norms are meaningful or should be factored out.
- Whether treating spatial tokens as an unordered bag destroys important structure.
- Which metric is most appropriate: Euclidean, cosine, whitened Euclidean, decoder-aware, or something spatially structured.

