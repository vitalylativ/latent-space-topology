# Latent-Space Topology: Project Note

## Current Thesis

We are not trying to prove that FLUX latent space has an intrinsic topology.
We are studying an empirical distribution induced by an encoder:

```text
images -> encoder -> latent tensor z in R^{C x H x W}
       -> spatial token vectors z_{h,w} in R^C
       -> point cloud / spatial field / empirical distribution
```

For the current FLUX-style experiments, each image gives a `(16, 32, 32)`
latent tensor, and each spatial position is treated as a token in `R^16`.
Depending on the question, we may study raw tokens, normalized directions on
`S^15`, whitened tokens, per-image summaries, or the spatial latent field.

The defensible goal is:

> Find stable, interpretable, and useful geometric summaries of
> encoder-induced representations, then test whether they distinguish datasets,
> tokenizers, image domains, or model behaviors.

Topology is one diagnostic layer. It is useful only if it adds something beyond
PCA, covariance, density, clustering, nearest-neighbor geometry, and simple
dataset statistics.

## Completed Result: Confirmatory H1 Sweep

The original promising observation was a long-lived `H1` feature under selected
preprocessing settings. We ran a confirmatory sweep to test whether that signal
survived held-out datasets, fixed seeds, matched controls, and pre-registered
decision rules.

Primary pipelines:

- `pca8_sphere_rips`: PCA to 8D, spherical normalization, dense selection,
  landmarks, and Rips persistent homology.
- `s15_witness`: raw FLUX token directions on `S^15`, density selection,
  landmarks, and a weak witness complex.

The matched controls included random tokens from the same transformed view,
uniform sphere samples, channel-shuffled tokens, matched Gaussian samples, and
a synthetic positive control excluded from the FLUX verdict.

The main comparison was:

```text
h1_norm = longest H1 lifetime / filtration normalizer
delta = observed_h1_norm - max(control_h1_norms)
ratio = observed_h1_norm / max(control_h1_norms)
win = delta > 0
```

A primary pipeline needed positive mean `delta`, a bootstrap CI above zero, high
win rate, cross-dataset support, and neighboring-setting stability. Secondary
settings could explain a result, but could not rescue a failed primary test.

Verdict:

> The held-out primary pipelines did not survive controls; the
> preprocessing-overfit hypothesis is supported.

Key numbers:

- `pca8_sphere_rips`: mean `delta = -0.210`, bootstrap 95% CI
  `[-0.226, -0.194]`, win rate `0%`, positive datasets `0 / 4`.
- `s15_witness`: mean `delta = 0.035`, bootstrap 95% CI
  `[-0.007, 0.080]`, win rate `55.8%`, positive datasets `2 / 4`.

Interpretation:

- The PCA8/Rips candidate is ruled out cleanly.
- The S15/witness candidate is not a robust cross-dataset loop.
- The remaining interesting signal is dataset-specific: stronger on STL-10 and
  CIFAR-10, weak or negative on Beans and Fashion-MNIST.

The result is a useful negative result. It says we should stop trying to rescue
the exact long-lived `H1` loop by further parameter sweeps. TDA may still be
useful, but as a controlled comparative diagnostic rather than a claim that a
universal FLUX latent loop has been found.

## What This Tells Us

The next project direction should shift from discovery-by-sweep to explanation
and comparison.

The most interesting remaining question is:

> Why do some natural-image datasets produce stronger witness-complex `H1`
> summaries than matched controls, while others do not?

Possible explanations:

- natural-image texture, color, frequency, or edge statistics;
- dataset diversity and class composition;
- resolution or preprocessing differences;
- density structure in token directions;
- spatial dependence between tokens from the same image;
- witness-complex or density-selection artifacts.

The current evidence does not decide between these explanations. The next
experiments should be designed to distinguish them, not to maximize `H1`.

## Evidence Standard

A topological feature is meaningful only if the claim is specific and the
feature survives controls. A credible claim should specify:

1. Target object: raw tokens, normalized directions, whitened tokens, image
   patches, per-image summaries, or spatial fields.
2. Metric and preprocessing: Euclidean, cosine, spherical, whitening, density
   selection, subsampling, and normalization.
3. Estimator: Rips, witness, alpha, Mapper, Betti curves, persistence images,
   or another summary.
4. Stability: seeds, image subsets, token subsets, density thresholds,
   landmarks, and neighboring preprocessing settings.
5. Controls: matched Gaussian, uniform sphere, random tokens, channel shuffle,
   spatial/block perturbations, and positive synthetic controls.
6. Interpretability: whether responsible landmarks map back to coherent image
   patches or model behaviors.
7. Utility: whether the summary predicts or distinguishes something useful.

Main artifact risks:

- normalization to `S^15` removes norm information and imposes spherical
  geometry;
- dense selection changes the support of the distribution;
- landmark and witness choices can create or remove features;
- spatial tokens from the same image are correlated, so raw token count
  overstates effective sample size;
- a visible persistence feature is not evidence unless it beats matched
  controls.

## Next Directions

1. Explain the dataset-specific witness signal.

   Focus on STL-10 and CIFAR-10 as positive cases, and Beans/Fashion-MNIST as
   negative cases. Compare simple geometry first: norms, PCA spectra,
   covariance, intrinsic dimension, nearest-neighbor distances, density, and
   spatial autocorrelation.

2. Back-map landmarks to image patches.

   For positive S15/witness runs, recover source image IDs and spatial
   locations for landmarks or cocycle-supporting tokens. Inspect whether the
   patches form a coherent visual progression. If they do not, treat the signal
   as an estimator or dataset-statistic effect rather than an interpretable
   loop.

3. Add dependence-preserving controls.

   Future controls should preserve more image-level structure: within-image
   token resampling, block/channel perturbations, per-image covariance controls,
   and controls that keep token norms or spatial autocorrelation fixed.

4. Compare representations and tokenizers.

   Use small, fixed, pre-registered comparisons across raw `R^16` tokens,
   normalized `S^15` directions, whitened tokens, raw image patches, and other
   tokenizers. The target is a comparative fingerprint, not one universal
   topology.

5. Convert topology into statistical features.

   Use Betti curves, persistence images, landscapes, bottleneck/Wasserstein
   distances, and bootstrap confidence bands. Test whether these features
   predict dataset, class, tokenizer, reconstruction quality, or failure modes.

6. Keep compute bounded.

   Every remote run should have a cost estimate, a stopping rule, a minimal
   primary test, and a written statement of what result would change our belief.

## Good Meeting Questions

- Are we studying local tokens, whole-image latents, spatial fields, or
  denoising trajectories?
- Are latent norms meaningful for this tokenizer, or should we study only
  directions?
- What would make a topological feature useful: interpretation, tokenizer
  comparison, anomaly detection, or downstream prediction?
- Which controls preserve enough structure to be fair?
- Can we map the responsible latent tokens back to coherent patches or image
  behaviors?
- Should the next result be a positive discovery, or a reliable fingerprinting
  method even if no literal loop survives?
