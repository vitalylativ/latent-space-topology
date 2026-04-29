# Latent-Space Topology: Motivation, Caveats, and Next Steps

## Motivation

The experiment studies the geometry of latent tokens produced by a generative-model encoder. Each image is encoded into a tensor of shape `(16, 32, 32)`, and each spatial position is treated as a point in `R^16`. After normalization, these points are interpreted as directions on `S^15`.

Why study this topology in the first place? Because the latent space is where the encoder has compressed the image distribution into a representation the generative model can manipulate. If the encoder is meaningful, nearby latent tokens should often correspond to related local visual patterns, and the global organization of these tokens may reveal how the model represents variation in images.

The hope is that the encoder does not place tokens arbitrarily. It may organize local visual patterns into regions, modes, loops, shells, or other geometric structures. Persistent homology can then be used as a coarse diagnostic of this organization.

Topology is useful here because it asks questions that PCA, covariance, and clustering do not directly answer:

- is the latent-token distribution one connected cloud or several separated regimes?
- are there stable cycles, shells, or voids suggesting constrained families of representations?
- do two tokenizers produce similar local statistics but different global organization?
- are some image domains represented as separate regions?
- do failure cases live near sparse zones, boundaries, holes, or disconnected pieces?
- does training, fine-tuning, checkpoint choice, or model size change the latent geometry?

The useful output is not necessarily "the true topology of images." A more modest and defensible goal is:

> Extract stable geometric/topological summaries of the encoder-induced token distribution, and test whether these summaries distinguish encoders, datasets, image domains, or model behaviors.

This can be useful if the summaries tell us something that ordinary statistics miss: for example, whether two tokenizers have similar reconstruction quality but different latent organization, whether some image domains occupy different regions of token space, or whether failure cases correspond to unusual latent geometry.

In this framing, topology gives us a diagnostic layer between raw latent statistics and model behavior. It becomes useful if it helps compare encoders, explain failures, track training, or reveal representation structure that simpler tools miss.

Possible uses:

- compare encoders or tokenizers as representation-level fingerprints;
- study dataset coverage and identify sparse or separated regions;
- detect latent regimes associated with bad reconstructions or bad generations;
- track how representation geometry changes during training or fine-tuning;
- build numerical features from persistence diagrams, Betti curves, or persistence images;
- map stable topological features back to image patches and interpret what visual variation they represent.

The key point is:

> We are not trying to prove that the latent space has an intrinsic topology independent of choices. We are trying to find stable, interpretable, and useful geometric signatures of the encoder-induced representation.

## Refined Problem Statement

The current project should be framed as a study of an empirical distribution induced by an encoder:

> Given an image dataset and a fixed tokenizer/autoencoder, study the geometry and topology of the spatial latent-token distribution it produces, and test whether the resulting summaries are stable, interpretable, and useful for comparing models, datasets, or failure regimes.

The central object is not the full latent space in the abstract. It is a sampled object:

```text
images -> encoder -> latent tensor z in R^{C x H x W}
       -> spatial token vectors z_{h,w} in R^C
       -> point cloud / field / distribution
```

For the current FLUX-style experiment, this means a point cloud of 16-dimensional spatial latent tokens. Several related objects should be kept separate:

- raw token vectors in `R^16`;
- normalized token directions on `S^15`;
- whitened or covariance-normalized tokens;
- per-image latent tensors as structured fields, not just bags of tokens;
- per-image summaries such as token means, covariances, norms, or diagrams;
- internal diffusion representations or denoising trajectories, which are different from autoencoder tokens.

The first research question is therefore:

> Which of these objects carries stable geometric/topological information, and what does that information tell us about the encoder, the data, or model behavior?

The detailed source map is kept separately in [literature_review.md](literature_review.md). The project note keeps the motivation, problem framing, caveats, and working plan.

## Most Interesting Direction

The most promising project is not:

> "Find the topology of FLUX latent space."

A stronger version is:

> Compare the topology of local visual representations before and after tokenization, and test whether stable topological summaries of latent-token distributions reveal tokenizer-specific organization, dataset coverage, or failure regimes.

This connects directly to the natural-image-patch literature. We can ask:

- Do raw image patches and encoded latent tokens have related topology?
- Does the encoder simplify, preserve, or reorganize classical patch geometry?
- Are stable cycles or components associated with interpretable visual attributes?
- Do different tokenizers induce different topological fingerprints?
- Do failure cases occupy sparse, boundary-like, or topologically unusual regions?

## Revised Exploration Plan

1. Start with geometry, not topology.

   Measure norm distributions, PCA spectra, covariance, intrinsic dimension, nearest-neighbor distances, density variation, and spatial correlations.

2. Define several candidate point clouds.

   Compare raw tokens, normalized token directions, whitened tokens, per-image summaries, and possibly raw pixel patches.

3. Reproduce the natural-image-patch baseline.

   Use small raw image patches with the Carlsson-style preprocessing as a positive reference point. This gives us a known nearby phenomenon and helps calibrate the pipeline.

4. Run PH as a stability experiment.

   Repeat across seeds, image subsets, token subsets, density thresholds, landmark counts, metrics, and complexes.

5. Add null and control clouds.

   Use uniform sphere samples, Gaussian matched covariance samples, shuffled channels, shuffled spatial positions, and density-matched synthetic samples.

6. Compare tokenizers and datasets.

   Use the same protocol for different autoencoders, datasets, classes, or image domains.

7. Interpret through back-mapping.

   For any stable feature, identify supporting landmarks, recover source images and spatial locations, and inspect corresponding patches.

8. Convert diagrams into statistical features.

   Use persistence landscapes, persistence images, Betti curves, bottleneck/Wasserstein distances, and bootstrap confidence bands.

## Meeting Framing

Good way to phrase the project in conversation:

> We want to understand whether image autoencoders induce stable and interpretable geometry on local latent tokens. TDA is one probe of this geometry, but the real target is not a single persistence diagram. The target is a controlled comparison of representation geometry across preprocessing choices, tokenizers, datasets, and model behaviors.

Good meeting questions:

- Are we interested in continuous autoencoder tokens, discrete tokenizer codes, or both?
- Should the object be local tokens, whole-image latents, or spatial latent fields?
- What model behavior should the topology explain or predict?
- Are latent norms meaningful for this autoencoder?
- Can we recover image patches corresponding to latent landmarks?
- Which controls would make a topological feature convincing to the group?

## What Can Be Extracted

Possible features include:

- persistence diagrams or barcodes in dimensions 0, 1, 2;
- persistent Betti curves across filtration scales;
- persistence images, landscapes, or vectorized diagram summaries;
- density-aware summaries of the most populated latent-token regions;
- comparisons between tokenizers using bottleneck or Wasserstein distances;
- stability statistics across image subsamples, random seeds, landmark choices, and preprocessing choices.

These features should be treated as empirical descriptors of the chosen representation and pipeline, not as direct ontological claims about the data.

## Main Caveat

The pipeline can easily invent topology.

For example, normalizing every token to the unit sphere removes norm information and imposes spherical geometry. Selecting only the densest points changes the support of the distribution. Landmark selection and witness-complex construction can create or destroy features. Spatial tokens from the same image are strongly correlated, so the effective sample size is much smaller than the raw number of token vectors.

Thus, a visible loop or void in a persistence diagram is not automatically evidence for a real latent-space structure. It may be an artifact of normalization, density thresholding, sampling, filtration cutoff, metric choice, or landmark selection.

## When Is A Feature Meaningful?

There is no absolute guarantee that an extracted topological feature is "real." The best we can do is define the target carefully and then accumulate evidence.

A meaningful claim should specify:

1. The target object: for example, the distribution of normalized encoder tokens for a given dataset.
2. The metric and preprocessing: Euclidean distance on `S^15`, raw `R^16`, cosine distance, whitening, density filtering, etc.
3. The estimator: Rips complex, witness complex, alpha complex if applicable, subsampling strategy.
4. The stability evidence: whether the same feature survives reasonable perturbations of the pipeline.
5. The baseline evidence: whether the feature is absent or different under appropriate null models.
6. The interpretability evidence: whether the feature can be connected back to image patches, semantic categories, reconstruction behavior, or model failures.
7. The utility evidence: whether the feature predicts, explains, or distinguishes something we care about.

In this framing, topology is meaningful if it is stable, nontrivial relative to controls, interpretable in terms of the encoder/data, and useful for comparison or prediction.

## Controls And Robustness Checks

Important controls:

- uniform random points on `S^15`;
- Gaussian points in `R^16` with matched mean and covariance;
- shuffled latent channels;
- shuffled spatial positions;
- random subsets of images and tokens;
- different image domains or classes;
- different encoders or tokenizer checkpoints;
- positive controls with synthetic data whose topology is known.

Important robustness checks:

- repeat over many random seeds;
- vary the number of images;
- vary the number of selected dense points;
- vary the nearest-neighbor density parameter `k`;
- vary the number of landmarks;
- compare witness complexes with Rips complexes on smaller subsamples;
- compare raw vectors, normalized vectors, whitened vectors, and cosine geometry;
- compute confidence intervals or bootstrap distributions for diagram summaries.

A feature that appears only for one arbitrary parameter choice should be treated as a warning sign, not as a discovery.

## Interpretation

If stable `H_0` structure appears, it may indicate separated token regimes or clusters.

If stable `H_1` structure appears, it may indicate a circular or cyclic organization of token directions. This could correspond to continuous variation in local visual attributes, but that interpretation must be checked by mapping representative tokens back to image patches.

If stable `H_2` or higher-dimensional structure appears, interpretation becomes harder. It may still be useful as a fingerprint of the encoder distribution, but semantic claims require stronger evidence.

The most convincing interpretation loop is:

1. Find a stable topological feature.
2. Identify tokens or landmarks responsible for the feature.
3. Map them back to spatial locations and source images.
4. Inspect whether they correspond to coherent visual patterns.
5. Test whether the feature changes under controlled changes of dataset, encoder, or preprocessing.

## Next Steps

The immediate next step is to turn the notebook into a reproducible experiment:

1. Fix all random seeds and save the selected tensor filenames.
2. Save all pipeline parameters with each run.
3. Run many subsamples instead of one sample.
4. Add null models and positive controls.
5. Compare both tokenizers or data parts using the same pipeline.
6. Summarize diagrams numerically, not only visually.
7. Map important landmarks back to image patches.
8. Report which features are stable and which disappear under perturbation.

Only after this should we make stronger claims about the encoder. Until then, the current result should be described as an exploratory probe of latent-token geometry.

## Preparation Plan

Guiding goal for the next discussion:

> Understand what it would mean to study tokenizer latent topology, what has already been done nearby, and what kind of evidence would make such a study credible.

### 1. Orient The Problem

Clarify the object of study:

- whole-image latents vs spatial latent tokens;
- raw latent vectors vs normalized directions;
- one encoder/tokenizer vs comparison between tokenizers;
- geometry of the data distribution vs geometry imposed by preprocessing.

Key question:

> What exactly is the point cloud whose topology we claim to study?

### 2. Broad Literature Search

Search several nearby fields:

- TDA for neural-network representations;
- topology and geometry of latent spaces in VAEs, GANs, and diffusion models;
- representation geometry of vision models;
- image tokenizers and latent diffusion autoencoders;
- persistent homology under noise, subsampling, and density filtering;
- Mapper, witness complexes, and density-based TDA for high-dimensional data.

The goal is to build a map of the field, not to read everything deeply at first.

### 3. Understand Tokenizers

For each relevant tokenizer family, understand:

- what the encoder is trained to optimize;
- what latent shape it produces;
- whether latents are continuous, discrete, quantized, normalized, or regularized;
- whether latent norms are meaningful;
- what each spatial token roughly corresponds to;
- how decoder usage constrains latent geometry.

Families worth comparing:

- VAE / KL autoencoders used in latent diffusion;
- VQ-VAE / vector-quantized tokenizers;
- diffusion-model VAEs such as Stable Diffusion / FLUX-style encoders;
- patch embeddings from vision transformers as a comparison point.

### 4. Build Geometric Intuition Before Topology

Before computing persistent homology, inspect simpler geometry:

- norm distributions;
- channel means and covariances;
- PCA spectra;
- intrinsic dimension estimates;
- nearest-neighbor distances;
- density variation;
- spatial correlations between neighboring tokens;
- token clusters and their corresponding image patches.

This prevents topology from becoming a black-box explanation for everything.

### 5. Study Representation Geometry / Topology

Ask what TDA can add:

- connected components as separated token regimes;
- loops as cyclic variation in local visual features;
- voids or shells as constrained latent directions;
- persistence diagrams as tokenizer fingerprints;
- Betti curves or persistence images as numerical summaries.

The right phrasing is:

> We study the topology of an encoder-induced empirical distribution under a specified pipeline.

### 6. Artifact-Control Plan

Prepare controls:

- uniform samples on `S^15`;
- Gaussian samples with matched covariance;
- shuffled latent channels;
- shuffled spatial positions;
- random image subsets;
- different density thresholds;
- different landmark counts;
- raw vs normalized vs whitened latents;
- Rips vs witness complex comparison on small samples.

Stability is the bridge from "pretty diagram" to "possible signal."

### 7. Interpretability Plan

If a topological feature appears, ask:

- which landmarks support it?
- which original images and patches do they come from?
- do they correspond to coherent visual patterns?
- does the feature change across classes, domains, or tokenizers?
- does it correlate with reconstruction quality or generation failures?

### 8. Questions For The Group

- What latent object is most interesting: tokens, whole images, trajectories, or something else?
- Are latent norms meaningful for this tokenizer?
- Do we expect tokenizer geometry to encode semantics, texture, frequency, color, or reconstruction constraints?
- What baselines would convince us that a topological feature is not an artifact?
- Is the goal interpretation, comparison of tokenizers, anomaly detection, or downstream prediction?
- Can we map latent tokens back to patches/images reliably?

End-of-day deliverables:

- a small annotated literature map;
- a tokenizer glossary;
- a diagram of the current experimental pipeline;
- a list of artifact risks;
- a list of controls;
- a short list of concrete research questions for the group chat.
