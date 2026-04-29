from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageOps
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "beans"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class TokenCloud:
    name: str
    model_id: str
    family: str
    token_kind: str
    tokens: np.ndarray
    token_metadata: pd.DataFrame
    grid_shape: tuple[int, int]
    channel_dim: int
    notes: dict[str, Any]
    reconstructions: list[Image.Image] | None = None
    code_indices: np.ndarray | None = None


@dataclass
class EncoderSpec:
    name: str
    model_id: str
    family: str
    short_description: str
    run: Callable[[list[Image.Image], pd.DataFrame], TokenCloud]


@dataclass
class CloudView:
    name: str
    cloud_name: str
    family: str
    view_kind: str
    tokens: np.ndarray
    token_metadata: pd.DataFrame
    grid_shape: tuple[int, int]
    notes: dict[str, Any]


def seed_everything(seed: int = 72) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def center_crop_resize(image: Image.Image, size: int, resample: int = Image.Resampling.BICUBIC) -> Image.Image:
    image = rgb(image)
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), resample=resample)


def _resolve_metadata_path(path_text: str, image_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    candidate = image_dir / path.name
    if candidate.exists():
        return candidate
    return path


def load_project_images(n_images: int = 12, image_dir: str | os.PathLike[str] | None = None) -> tuple[list[Image.Image], pd.DataFrame]:
    """Load local downloaded images by default, falling back to the HF beans dataset."""
    folder = Path(image_dir).expanduser() if image_dir else DEFAULT_IMAGE_DIR
    if not folder.is_absolute() and not folder.exists():
        project_relative = PROJECT_ROOT / folder
        if project_relative.exists():
            folder = project_relative
    if folder.exists():
        metadata_path = folder / "metadata.csv"
        if metadata_path.exists():
            meta = pd.read_csv(metadata_path).head(n_images).copy()
            images = []
            rows = []
            for i, row in meta.iterrows():
                path = _resolve_metadata_path(str(row["path"]), folder)
                images.append(rgb(Image.open(path)))
                rows.append(
                    {
                        "image_id": len(rows),
                        "source": row.get("dataset", "local"),
                        "path": str(path),
                        "label": row.get("label"),
                        "dataset_index": row.get("dataset_index"),
                        "width": images[-1].width,
                        "height": images[-1].height,
                    }
                )
            return images, pd.DataFrame(rows)

        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)[:n_images]
        if paths:
            images = [rgb(Image.open(path)) for path in paths]
            rows = [
                {
                    "image_id": i,
                    "source": "local_folder",
                    "path": str(path),
                    "label": None,
                    "dataset_index": None,
                    "width": img.width,
                    "height": img.height,
                }
                for i, (path, img) in enumerate(zip(paths, images))
            ]
            return images, pd.DataFrame(rows)

    from datasets import load_dataset

    ds = load_dataset("beans", split=f"train[:{n_images}]")
    names = ds.features["labels"].names
    images = []
    rows = []
    for i, item in enumerate(ds):
        images.append(rgb(item["image"]))
        label_id = int(item["labels"])
        rows.append(
            {
                "image_id": i,
                "source": "beans",
                "path": None,
                "label": names[label_id],
                "dataset_index": i,
                "width": images[-1].width,
                "height": images[-1].height,
            }
        )
    return images, pd.DataFrame(rows)


def show_image_grid(images: list[Image.Image], metadata: pd.DataFrame, n: int = 9, title: str | None = None) -> None:
    n = min(n, len(images))
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.1 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, img, (_, row) in zip(axes, images[:n], metadata.head(n).iterrows()):
        ax.imshow(img)
        label = row.get("label")
        ax.set_title(str(label) if pd.notna(label) else f"image {row['image_id']}", fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def chunked(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def pil_to_vae_tensor(batch: list[Image.Image], size: int) -> torch.Tensor:
    arrays = []
    for img in batch:
        arr = np.asarray(center_crop_resize(img, size), dtype=np.float32) / 255.0
        arrays.append(arr.transpose(2, 0, 1))
    tensor = torch.from_numpy(np.stack(arrays))
    return tensor * 2.0 - 1.0


def tensor_to_pil_list(tensor: torch.Tensor, max_images: int = 4) -> list[Image.Image]:
    tensor = tensor.detach().cpu().float().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    output = []
    for item in tensor[:max_images]:
        arr = (item.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        output.append(Image.fromarray(arr))
    return output


def flatten_spatial(z: torch.Tensor) -> tuple[np.ndarray, tuple[int, int], int]:
    z = z.detach().cpu().float()
    b, c, h, w = z.shape
    return z.permute(0, 2, 3, 1).reshape(b * h * w, c).numpy(), (h, w), c


def flatten_sequence(tokens: torch.Tensor, grid_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int], int]:
    tokens = tokens.detach().cpu().float()
    b, n, c = tokens.shape
    return tokens.reshape(b * n, c).numpy(), grid_shape, c


def build_token_metadata(base_metadata: pd.DataFrame, model_id: str, family: str, token_kind: str, grid_shape: tuple[int, int]) -> pd.DataFrame:
    h, w = grid_shape
    rows = []
    for _, base in base_metadata.iterrows():
        for yy in range(h):
            for xx in range(w):
                rows.append(
                    {
                        "image_id": int(base["image_id"]),
                        "label": base.get("label"),
                        "source": base.get("source"),
                        "model_id": model_id,
                        "family": family,
                        "token_kind": token_kind,
                        "h": yy,
                        "w": xx,
                    }
                )
    return pd.DataFrame(rows)


def safe_to_device(model: torch.nn.Module, preferred_device: str) -> tuple[torch.nn.Module, str]:
    try:
        return model.to(preferred_device), preferred_device
    except Exception as exc:
        print(f"Could not move model to {preferred_device}; using CPU. Reason: {type(exc).__name__}: {exc}")
        return model.to("cpu"), "cpu"


def pipeline_scaled_latents(vae: Any, z: torch.Tensor) -> torch.Tensor:
    scaling = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    shift = getattr(vae.config, "shift_factor", None)
    return z * scaling if shift is None else (z - float(shift)) * scaling


def run_autoencoder_kl(
    model_id: str,
    name: str,
    images: list[Image.Image],
    metadata: pd.DataFrame,
    device: str,
    batch_size: int,
    image_size: int,
) -> TokenCloud:
    from diffusers import AutoencoderKL

    model = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True)
    model, used_device = safe_to_device(model.eval(), device)
    latent_chunks = []
    scaled_means = []
    scaled_stds = []
    recons: list[Image.Image] = []
    with torch.inference_mode():
        for _, batch_images in chunked(images, batch_size):
            batch = pil_to_vae_tensor(batch_images, image_size).to(used_device, dtype=torch.float32)
            z = model.encode(batch).latent_dist.mean
            z_scaled = pipeline_scaled_latents(model, z)
            latent_chunks.append(z.detach().cpu())
            scaled_means.append(float(z_scaled.detach().cpu().mean()))
            scaled_stds.append(float(z_scaled.detach().cpu().std()))
            if len(recons) < 4:
                n_recon = min(4 - len(recons), z.shape[0])
                decoded = model.decode(z[:n_recon]).sample
                recons.extend(tensor_to_pil_list(decoded, n_recon))
    z_all = torch.cat(latent_chunks, dim=0)
    tokens, grid_shape, channel_dim = flatten_spatial(z_all)
    token_metadata = build_token_metadata(metadata, model_id, "AutoencoderKL", "posterior_mean", grid_shape)
    notes = {
        "device": used_device,
        "latent_shape_bchw": tuple(z_all.shape),
        "scaling_factor": float(getattr(model.config, "scaling_factor", 1.0) or 1.0),
        "shift_factor": getattr(model.config, "shift_factor", None),
        "scaled_latent_mean": float(np.mean(scaled_means)),
        "scaled_latent_std": float(np.mean(scaled_stds)),
    }
    return TokenCloud(name, model_id, "AutoencoderKL", "posterior_mean", tokens, token_metadata, grid_shape, channel_dim, notes, recons)


def collect_tensors(obj: Any) -> list[torch.Tensor]:
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, dict):
        out = []
        for value in obj.values():
            out.extend(collect_tensors(value))
        return out
    if isinstance(obj, (tuple, list)):
        out = []
        for value in obj:
            out.extend(collect_tensors(value))
        return out
    if hasattr(obj, "__dict__"):
        out = []
        for value in obj.__dict__.values():
            out.extend(collect_tensors(value))
        return out
    return []


def parse_vq_quantizer_output(output: Any, latents: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    tensors = collect_tensors(output)
    quantized = None
    indices = None
    for tensor in tensors:
        if tensor.ndim == latents.ndim and tuple(tensor.shape[-2:]) == tuple(latents.shape[-2:]):
            quantized = tensor
            break
    for tensor in tensors:
        if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            indices = tensor
            break
    return quantized, indices


def normalize_code_indices(indices: torch.Tensor | None, latents: torch.Tensor) -> np.ndarray | None:
    if indices is None:
        return None
    idx = indices.detach().cpu()
    if idx.ndim == 1 and idx.numel() == latents.shape[0] * latents.shape[-2] * latents.shape[-1]:
        idx = idx.reshape(latents.shape[0], -1)
    elif idx.ndim > 2:
        idx = idx.reshape(idx.shape[0], -1)
    return idx.numpy()


def run_vq_model(
    images: list[Image.Image],
    metadata: pd.DataFrame,
    device: str,
    batch_size: int,
    image_size: int,
) -> TokenCloud:
    from diffusers import VQModel

    model_id = "kandinsky-community/kandinsky-2-2-decoder"
    subfolder = "movq"
    model = VQModel.from_pretrained(model_id, subfolder=subfolder, torch_dtype=torch.float32, use_safetensors=True)
    model, used_device = safe_to_device(model.eval(), device)
    token_chunks = []
    index_chunks = []
    recons: list[Image.Image] = []
    with torch.inference_mode():
        for _, batch_images in chunked(images, batch_size):
            batch = pil_to_vae_tensor(batch_images, image_size).to(used_device, dtype=torch.float32)
            encoded = model.encode(batch)
            latents = getattr(encoded, "latents", None)
            if latents is None and isinstance(encoded, (tuple, list)):
                latents = encoded[0]
            if latents is None:
                raise RuntimeError("Could not find VQ encoder latents")
            quantized, indices = parse_vq_quantizer_output(model.quantize(latents), latents)
            token_tensor = quantized if quantized is not None else latents
            token_chunks.append(token_tensor.detach().cpu())
            code_idx = normalize_code_indices(indices, latents)
            if code_idx is not None:
                index_chunks.append(code_idx)
            if len(recons) < 4:
                n_recon = min(4 - len(recons), token_tensor.shape[0])
                try:
                    decoded = model.decode(token_tensor[:n_recon]).sample
                    recons.extend(tensor_to_pil_list(decoded, n_recon))
                except Exception:
                    pass
    token_all = torch.cat(token_chunks, dim=0)
    tokens, grid_shape, channel_dim = flatten_spatial(token_all)
    full_model_id = f"{model_id}/{subfolder}"
    token_metadata = build_token_metadata(metadata, full_model_id, "VQModel", "quantized_embedding", grid_shape)
    code_indices = np.concatenate(index_chunks, axis=0) if index_chunks else None
    notes = {
        "device": used_device,
        "latent_shape_bchw": tuple(token_all.shape),
        "has_code_indices": code_indices is not None,
        "num_vq_embeddings": getattr(model.config, "num_vq_embeddings", None),
        "vq_embed_dim": getattr(model.config, "vq_embed_dim", None),
    }
    return TokenCloud("kandinsky_movq", full_model_id, "VQModel", "quantized_embedding", tokens, token_metadata, grid_shape, channel_dim, notes, recons, code_indices)


def run_vit_model(
    images: list[Image.Image],
    metadata: pd.DataFrame,
    device: str,
    batch_size: int,
    image_size: int,
) -> TokenCloud:
    from transformers import ViTImageProcessor, ViTModel

    model_id = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTModel.from_pretrained(model_id)
    model, used_device = safe_to_device(model.eval(), device)
    token_chunks = []
    with torch.inference_mode():
        for _, batch_images in chunked(images, batch_size):
            prepared = [center_crop_resize(img, image_size) for img in batch_images]
            inputs = processor(images=prepared, return_tensors="pt")
            inputs = {k: v.to(used_device) for k, v in inputs.items()}
            token_chunks.append(model(**inputs).last_hidden_state[:, 1:, :].detach().cpu())
    all_tokens = torch.cat(token_chunks, dim=0)
    n = all_tokens.shape[1]
    side = int(math.sqrt(n))
    grid_shape = (side, side) if side * side == n else (1, n)
    tokens, grid_shape, channel_dim = flatten_sequence(all_tokens, grid_shape)
    token_metadata = build_token_metadata(metadata, model_id, "ViT", "contextual_patch_embedding", grid_shape)
    notes = {"device": used_device, "sequence_shape_bnc": tuple(all_tokens.shape), "patch_size": getattr(model.config, "patch_size", None)}
    return TokenCloud("vit_base_patch16", model_id, "ViT", "contextual_patch_embedding", tokens, token_metadata, grid_shape, channel_dim, notes)


def run_clip_vision_model(
    images: list[Image.Image],
    metadata: pd.DataFrame,
    device: str,
    batch_size: int,
    image_size: int,
) -> TokenCloud:
    from transformers import CLIPImageProcessor, CLIPModel

    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model, used_device = safe_to_device(model.eval(), device)
    token_chunks = []
    with torch.inference_mode():
        for _, batch_images in chunked(images, batch_size):
            prepared = [center_crop_resize(img, image_size) for img in batch_images]
            inputs = processor(images=prepared, return_tensors="pt")
            inputs = {k: v.to(used_device) for k, v in inputs.items()}
            token_chunks.append(model.vision_model(**inputs).last_hidden_state[:, 1:, :].detach().cpu())
    all_tokens = torch.cat(token_chunks, dim=0)
    n = all_tokens.shape[1]
    side = int(math.sqrt(n))
    grid_shape = (side, side) if side * side == n else (1, n)
    tokens, grid_shape, channel_dim = flatten_sequence(all_tokens, grid_shape)
    token_metadata = build_token_metadata(metadata, model_id, "CLIPVision", "contextual_patch_embedding", grid_shape)
    patch_size = getattr(getattr(model.config, "vision_config", None), "patch_size", None)
    notes = {"device": used_device, "sequence_shape_bnc": tuple(all_tokens.shape), "patch_size": patch_size}
    return TokenCloud("clip_vit_base_patch32", model_id, "CLIPVision", "contextual_patch_embedding", tokens, token_metadata, grid_shape, channel_dim, notes)


def run_raw_patches(images: list[Image.Image], metadata: pd.DataFrame, image_size: int, patch_size: int = 16) -> TokenCloud:
    rows = []
    for img in images:
        arr = np.asarray(center_crop_resize(img, image_size), dtype=np.float32) / 255.0
        for yy in range(0, image_size, patch_size):
            for xx in range(0, image_size, patch_size):
                patch = arr[yy : yy + patch_size, xx : xx + patch_size, :]
                vec = patch.reshape(-1)
                vec = vec - vec.mean()
                denom = np.linalg.norm(vec)
                rows.append(vec / denom if denom > 1e-8 else vec)
    tokens = np.stack(rows).astype(np.float32)
    grid_shape = (image_size // patch_size, image_size // patch_size)
    token_metadata = build_token_metadata(metadata, "raw_image_patches", "RawPatches", "centered_contrast_patch", grid_shape)
    notes = {"patch_size": patch_size, "image_size": image_size, "preprocessing": "patch mean subtraction plus L2 normalization"}
    return TokenCloud("raw_patches", "raw_image_patches", "RawPatches", "centered_contrast_patch", tokens, token_metadata, grid_shape, tokens.shape[1], notes)


def default_encoder_specs(device: str, batch_size: int, autoencoder_size: int, vit_size: int) -> list[EncoderSpec]:
    return [
        EncoderSpec(
            "flux_vae",
            "diffusers/FLUX.1-vae",
            "AutoencoderKL",
            "A latent-diffusion VAE with 16 continuous channels. This is closest to the current project object.",
            lambda imgs, meta: run_autoencoder_kl("diffusers/FLUX.1-vae", "flux_vae", imgs, meta, device, batch_size, autoencoder_size),
        ),
        EncoderSpec(
            "sd_vae_ft_mse",
            "stabilityai/sd-vae-ft-mse",
            "AutoencoderKL",
            "A Stable-Diffusion-style KL autoencoder with 4 continuous latent channels.",
            lambda imgs, meta: run_autoencoder_kl("stabilityai/sd-vae-ft-mse", "sd_vae_ft_mse", imgs, meta, device, batch_size, autoencoder_size),
        ),
        EncoderSpec(
            "kandinsky_movq",
            "kandinsky-community/kandinsky-2-2-decoder/movq",
            "VQModel",
            "A vector-quantized image tokenizer: encoder output is snapped to codebook vectors.",
            lambda imgs, meta: run_vq_model(imgs, meta, device, batch_size, autoencoder_size),
        ),
        EncoderSpec(
            "vit_base_patch16",
            "google/vit-base-patch16-224-in21k",
            "ViT",
            "A transformer vision encoder. Its patch tokens are contextual representation features, not decoder latents.",
            lambda imgs, meta: run_vit_model(imgs, meta, device, batch_size, vit_size),
        ),
        EncoderSpec(
            "clip_vit_base_patch32",
            "openai/clip-vit-base-patch32",
            "CLIPVision",
            "A contrastive image-text vision encoder. Its geometry is shaped by alignment with text.",
            lambda imgs, meta: run_clip_vision_model(imgs, meta, device, batch_size, vit_size),
        ),
        EncoderSpec(
            "raw_patches",
            "raw_image_patches",
            "RawPatches",
            "A non-learned baseline: local pixel patches after centering and contrast normalization.",
            lambda imgs, meta: run_raw_patches(imgs, meta, autoencoder_size),
        ),
    ]


def extract_token_clouds(
    images: list[Image.Image],
    metadata: pd.DataFrame,
    device: str,
    batch_size: int = 4,
    autoencoder_size: int = 256,
    vit_size: int = 224,
    selected: list[str] | None = None,
) -> tuple[dict[str, TokenCloud], pd.DataFrame]:
    specs = default_encoder_specs(device, batch_size, autoencoder_size, vit_size)
    if selected:
        keep = set(selected)
        specs = [spec for spec in specs if spec.name in keep]
    clouds: dict[str, TokenCloud] = {}
    failures = []
    for spec in specs:
        print(f"Running {spec.name}: {spec.model_id}")
        start = time.time()
        try:
            cloud = spec.run(images, metadata)
            clouds[cloud.name] = cloud
            print(f"  ok: {cloud.tokens.shape}, grid={cloud.grid_shape}, elapsed={time.time() - start:.1f}s")
        except Exception as exc:
            print(f"  skipped: {type(exc).__name__}: {exc}")
            failures.append({"name": spec.name, "model_id": spec.model_id, "error_type": type(exc).__name__, "error": str(exc)})
    return clouds, pd.DataFrame(failures)


def encoder_story_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "encoder": "FLUX VAE",
                "what_it_is": "Continuous autoencoder used to compress images before diffusion operates in latent space.",
                "training_pressure": "Reconstruct images while keeping latents in a distribution the diffusion model can use.",
                "tokens_mean": "Each spatial token is a 16D local latent vector on a 32x32 grid for 256x256 images.",
                "why_we_care": "This matches the current project's 16-channel latent-token object.",
            },
            {
                "encoder": "Stable Diffusion VAE",
                "what_it_is": "Older latent-diffusion KL autoencoder.",
                "training_pressure": "Reconstruction/perceptual quality plus KL regularization.",
                "tokens_mean": "Each spatial token is a 4D continuous latent vector.",
                "why_we_care": "Useful contrast: same family as FLUX-style VAEs, but lower channel dimension.",
            },
            {
                "encoder": "VQ / MoVQ",
                "what_it_is": "Discrete image tokenizer with a learned codebook.",
                "training_pressure": "Reconstruct images after replacing encoder vectors by nearest codebook embeddings.",
                "tokens_mean": "Each spatial location maps to a code index and a learned embedding vector.",
                "why_we_care": "Topology may reflect codebook discreteness and duplicate embeddings, not just visual continuity.",
            },
            {
                "encoder": "ViT",
                "what_it_is": "Vision transformer representation model.",
                "training_pressure": "Classification or representation learning, not image reconstruction.",
                "tokens_mean": "Patch tokens become contextual after attention; a token sees more than its original patch.",
                "why_we_care": "A strong comparison point for representation geometry.",
            },
            {
                "encoder": "CLIP vision",
                "what_it_is": "Vision encoder trained by image-text contrastive learning.",
                "training_pressure": "Place matching images and text nearby in representation space.",
                "tokens_mean": "Patch tokens are contextual features; pooled output is aligned to text.",
                "why_we_care": "Shows geometry shaped by semantic/contrastive objectives rather than reconstruction.",
            },
            {
                "encoder": "Raw patches",
                "what_it_is": "No learned encoder.",
                "training_pressure": "None.",
                "tokens_mean": "Centered and L2-normalized pixel patches.",
                "why_we_care": "Connects to the natural-image-patch literature and exposes preprocessing artifacts.",
            },
        ]
    )


def shape_summary(clouds: dict[str, TokenCloud]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": name,
                "family": cloud.family,
                "token_kind": cloud.token_kind,
                "tokens_shape": tuple(cloud.tokens.shape),
                "grid_shape": cloud.grid_shape,
                "channel_dim": cloud.channel_dim,
                "model_id": cloud.model_id,
            }
            for name, cloud in clouds.items()
        ]
    )


def show_reconstruction_grid(images: list[Image.Image], clouds: dict[str, TokenCloud], image_size: int = 256, max_images: int = 4) -> None:
    recon_clouds = [cloud for cloud in clouds.values() if cloud.reconstructions]
    if not recon_clouds:
        print("No reconstruction-capable clouds were available.")
        return
    n = min(max_images, len(images))
    fig, axes = plt.subplots(len(recon_clouds) + 1, n, figsize=(2.4 * n, 2.35 * (len(recon_clouds) + 1)))
    axes = np.asarray(axes).reshape(len(recon_clouds) + 1, n)
    for j in range(n):
        axes[0, j].imshow(center_crop_resize(images[j], image_size))
        axes[0, j].set_title("original", fontsize=9)
        axes[0, j].axis("off")
    for row, cloud in enumerate(recon_clouds, start=1):
        for j in range(n):
            axes[row, j].imshow(cloud.reconstructions[j])
            axes[row, j].set_title(cloud.name, fontsize=9)
            axes[row, j].axis("off")
    plt.tight_layout()
    plt.show()


def token_norm_table(clouds: dict[str, TokenCloud]) -> pd.DataFrame:
    rows = []
    for name, cloud in clouds.items():
        norms = np.linalg.norm(cloud.tokens, axis=1)
        rows.append(
            {
                "name": name,
                "family": cloud.family,
                "mean_norm": norms.mean(),
                "std_norm": norms.std(),
                "cv_norm": norms.std() / max(norms.mean(), 1e-12),
                "min_norm": norms.min(),
                "max_norm": norms.max(),
            }
        )
    return pd.DataFrame(rows)


def safe_hist(ax: plt.Axes, values: np.ndarray, bins: int = 40, **kwargs: Any) -> None:
    values = np.asarray(values)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    span = hi - lo
    scale = max(1.0, abs(lo), abs(hi))
    if span <= 1e-7 * scale:
        ax.axvline(float(values[0]), linewidth=2)
        ax.set_ylabel("constant")
    else:
        n_bins = min(bins, max(5, int(np.sqrt(len(values)))))
        ax.hist(values, bins=np.linspace(lo, hi, n_bins + 1), **kwargs)


def plot_norm_distributions(clouds: dict[str, TokenCloud]) -> None:
    if not clouds:
        return
    cols = 2
    rows = math.ceil(len(clouds) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, cloud) in zip(axes, clouds.items()):
        norms = np.linalg.norm(cloud.tokens, axis=1)
        safe_hist(ax, norms, alpha=0.85)
        ax.set_title(f"{name}: token norm")
        ax.set_xlabel("L2 norm")
    for ax in axes[len(clouds) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def norm_map(cloud: TokenCloud, image_id: int = 0) -> np.ndarray:
    h, w = cloud.grid_shape
    d = cloud.channel_dim
    b = int(cloud.tokens.shape[0] / (h * w))
    x = cloud.tokens.reshape(b, h, w, d)
    return np.linalg.norm(x[image_id], axis=-1)


def plot_norm_maps(clouds: dict[str, TokenCloud], image_id: int = 0) -> None:
    if not clouds:
        return
    cols = min(3, len(clouds))
    rows = math.ceil(len(clouds) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, cloud) in zip(axes, clouds.items()):
        im = ax.imshow(norm_map(cloud, image_id=image_id), cmap="magma")
        ax.set_title(f"{name}: norm map")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(clouds) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def pca_projection(cloud: TokenCloud, max_points: int = 2500, seed: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cloud.tokens))
    if len(idx) > max_points:
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    x = cloud.tokens[idx]
    xy = PCA(n_components=2, random_state=seed).fit_transform(x)
    meta = cloud.token_metadata.iloc[idx].reset_index(drop=True)
    return pd.DataFrame(
        {
            "pc1": xy[:, 0],
            "pc2": xy[:, 1],
            "image_id": meta["image_id"].astype(str),
            "label": meta["label"].astype(str),
            "h": meta["h"],
            "w": meta["w"],
        }
    )


def plot_pca_by_label(cloud: TokenCloud, max_points: int = 2500, seed: int = 72) -> None:
    df = pca_projection(cloud, max_points=max_points, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df, x="pc1", y="pc2", hue="label", s=12, alpha=0.6, ax=axes[0])
    axes[0].set_title(f"{cloud.name}: PCA colored by label")
    sns.scatterplot(data=df, x="pc1", y="pc2", hue="image_id", s=12, alpha=0.6, ax=axes[1], legend=False)
    axes[1].set_title(f"{cloud.name}: PCA colored by source image")
    plt.tight_layout()
    plt.show()


def code_usage_table(cloud: TokenCloud) -> pd.DataFrame:
    if cloud.code_indices is None:
        return pd.DataFrame()
    flat = cloud.code_indices.reshape(-1)
    counts = pd.Series(flat).value_counts().sort_values(ascending=False)
    probs = counts / counts.sum()
    return pd.DataFrame({"code": counts.index.astype(int), "count": counts.values, "frequency": probs.values})


def plot_code_map(cloud: TokenCloud, image_id: int = 0) -> None:
    if cloud.code_indices is None:
        print(f"{cloud.name} has no code indices.")
        return
    h, w = cloud.grid_shape
    codes = cloud.code_indices.reshape(-1, h, w)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(codes[image_id], cmap="tab20")
    ax.set_title(f"{cloud.name}: VQ code ids for image {image_id}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def pca_whiten(x: np.ndarray, max_dim: int = 64, seed: int = 72) -> tuple[np.ndarray, dict[str, Any]]:
    n_components = min(max_dim, x.shape[0] - 1, x.shape[1])
    if n_components < 1:
        return x.copy(), {"whiten_components": x.shape[1], "explained_variance_sum": None}
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    z = pca.fit_transform(x)
    return z.astype(np.float32), {
        "whiten_components": n_components,
        "explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
    }


def make_cloud_views(clouds: dict[str, TokenCloud], max_whiten_dim: int = 64, seed: int = 72) -> dict[str, CloudView]:
    views: dict[str, CloudView] = {}
    for cloud_name, cloud in clouds.items():
        raw = cloud.tokens.astype(np.float32)
        views[f"{cloud_name}:raw"] = CloudView(f"{cloud_name}:raw", cloud_name, cloud.family, "raw", raw, cloud.token_metadata, cloud.grid_shape, {})
        views[f"{cloud_name}:unit"] = CloudView(f"{cloud_name}:unit", cloud_name, cloud.family, "unit_norm", l2_normalize(raw).astype(np.float32), cloud.token_metadata, cloud.grid_shape, {})
        white, notes = pca_whiten(raw, max_dim=max_whiten_dim, seed=seed)
        views[f"{cloud_name}:whitened"] = CloudView(f"{cloud_name}:whitened", cloud_name, cloud.family, "pca_whitened", white, cloud.token_metadata, cloud.grid_shape, notes)
    return views


def sample_indices(n: int, max_n: int, seed: int = 72) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def participation_ratio(pca: PCA) -> float:
    vals = np.maximum(pca.explained_variance_, 0)
    if vals.sum() <= 0:
        return float("nan")
    return float((vals.sum() ** 2) / np.sum(vals ** 2))


def twonn_intrinsic_dimension(x: np.ndarray) -> float:
    if len(x) < 4:
        return float("nan")
    distances, _ = NearestNeighbors(n_neighbors=3).fit(x).kneighbors(x)
    scale = max(1.0, float(np.median(np.linalg.norm(x, axis=1))))
    eps = 1e-8 * scale
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = (r1 > eps) & (r2 > r1 * (1.0 + 1e-8))
    if valid.sum() < max(10, len(x) // 2):
        return float("nan")
    logs = np.log((r2[valid] / r1[valid]))
    if len(logs) == 0 or logs.mean() <= 0:
        return float("nan")
    return float(1.0 / logs.mean())


def spatial_neighbor_cosine(view: CloudView, n_images: int) -> dict[str, float]:
    h, w = view.grid_shape
    d = view.tokens.shape[1]
    if h * w * n_images != len(view.tokens) or h < 2 or w < 2:
        return {"spatial_cosine_mean": np.nan}
    x = view.tokens.reshape(n_images, h, w, d)
    vals = []
    for a, c in [(x[:, :, :-1, :], x[:, :, 1:, :]), (x[:, :-1, :, :], x[:, 1:, :, :])]:
        aa = a.reshape(-1, d)
        cc = c.reshape(-1, d)
        denom = np.linalg.norm(aa, axis=1) * np.linalg.norm(cc, axis=1)
        vals.append(np.sum(aa * cc, axis=1) / np.maximum(denom, 1e-8))
    return {"spatial_cosine_mean": float(np.mean(np.concatenate(vals)))}


def label_silhouette(x: np.ndarray, metadata: pd.DataFrame) -> float:
    labels = metadata["label"].astype(str).to_numpy()
    valid = np.array([label not in {"None", "nan"} for label in labels])
    if valid.sum() < 10 or len(np.unique(labels[valid])) < 2:
        return float("nan")
    if min(pd.Series(labels[valid]).value_counts()) < 2:
        return float("nan")
    try:
        return float(silhouette_score(x[valid], labels[valid], metric="euclidean"))
    except Exception:
        return float("nan")


def geometry_metrics(
    views: dict[str, CloudView],
    n_images: int,
    max_points: int = 5000,
    seed: int = 72,
) -> pd.DataFrame:
    rows = []
    for name, view in views.items():
        idx = sample_indices(len(view.tokens), max_points, seed=seed + len(rows))
        x = view.tokens[idx]
        meta = view.token_metadata.iloc[idx].reset_index(drop=True)
        norms = np.linalg.norm(x, axis=1)
        n_components = min(32, x.shape[0] - 1, x.shape[1])
        pca = PCA(n_components=n_components, random_state=seed).fit(x)
        ratios = pca.explained_variance_ratio_
        nbr_k = min(16, len(x))
        distances, _ = NearestNeighbors(n_neighbors=nbr_k).fit(x).kneighbors(x)
        kth = distances[:, -1]
        nearest = distances[:, 1] if distances.shape[1] > 1 else np.full(len(x), np.nan)
        duplicate_eps = 1e-8 * max(1.0, float(np.median(norms)))
        row = {
            "view": name,
            "cloud": view.cloud_name,
            "family": view.family,
            "view_kind": view.view_kind,
            "n_points": len(x),
            "dim": x.shape[1],
            "norm_cv": float(norms.std() / max(norms.mean(), 1e-12)),
            "pc1": float(ratios[0]),
            "pca_80_components": int(np.searchsorted(np.cumsum(ratios), 0.80) + 1),
            "participation_ratio": participation_ratio(pca),
            "twonn_id": twonn_intrinsic_dimension(x),
            "density_q90_q10": float(np.quantile(kth, 0.90) / max(np.quantile(kth, 0.10), 1e-12)),
            "near_duplicate_fraction": float(np.mean(nearest <= duplicate_eps)),
            "label_silhouette": label_silhouette(x, meta),
        }
        row.update(spatial_neighbor_cosine(view, n_images))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cloud", "view_kind"]).reset_index(drop=True)


def distance_preprocessing_effects(views: dict[str, CloudView], max_points: int = 700, seed: int = 72) -> pd.DataFrame:
    rows = []
    clouds = sorted({view.cloud_name for view in views.values()})
    for cloud_name in clouds:
        raw = views.get(f"{cloud_name}:raw")
        if raw is None:
            continue
        idx = sample_indices(len(raw.tokens), max_points, seed=seed)
        raw_dist = pdist(raw.tokens[idx], metric="euclidean")
        for kind in ["unit", "whitened"]:
            view = views.get(f"{cloud_name}:{kind}")
            if view is None:
                continue
            idx2 = sample_indices(len(view.tokens), max_points, seed=seed)
            dist = pdist(view.tokens[idx2], metric="euclidean")
            n = min(len(raw_dist), len(dist))
            rows.append(
                {
                    "cloud": cloud_name,
                    "comparison": f"raw_vs_{kind}",
                    "spearman_distance_corr": float(spearmanr(raw_dist[:n], dist[:n]).statistic),
                    "raw_dist_mean": float(raw_dist.mean()),
                    "comparison_dist_mean": float(dist.mean()),
                }
            )
    return pd.DataFrame(rows)


def main_effect_table(metrics: pd.DataFrame, distance_effects: pd.DataFrame) -> pd.DataFrame:
    raw = metrics[metrics["view_kind"] == "raw"].copy()
    unit = distance_effects[distance_effects["comparison"] == "raw_vs_unit"][["cloud", "spearman_distance_corr"]].rename(columns={"spearman_distance_corr": "raw_unit_spearman"})
    white = distance_effects[distance_effects["comparison"] == "raw_vs_whitened"][["cloud", "spearman_distance_corr"]].rename(columns={"spearman_distance_corr": "raw_whitened_spearman"})
    table = raw.merge(unit, on="cloud", how="left").merge(white, on="cloud", how="left")

    def notes(row: pd.Series) -> str:
        out = []
        if row["norm_cv"] > 0.25:
            out.append("large norm variation")
        if row["pc1"] > 0.35:
            out.append("dominant first PC")
        if row["density_q90_q10"] > 2.0:
            out.append("strong density variation")
        if row["near_duplicate_fraction"] > 0.05:
            out.append("many duplicate or quantized-neighbor tokens")
        if row["spatial_cosine_mean"] > 0.55:
            out.append("strong spatial autocorrelation")
        if row.get("raw_unit_spearman", 1.0) < 0.85:
            out.append("unit normalization changes distances")
        if row.get("raw_whitened_spearman", 1.0) < 0.85:
            out.append("whitening changes distances")
        return "; ".join(out) if out else "no single coarse effect dominates"

    table["takeaway"] = table.apply(notes, axis=1)
    return table


def approximate_patch(cloud: TokenCloud, images: list[Image.Image], token_index: int, image_size: int = 256, context_cells: int = 1) -> Image.Image:
    row = cloud.token_metadata.iloc[int(token_index)]
    image_id = int(row["image_id"])
    img = center_crop_resize(images[image_id], image_size)
    h, w = cloud.grid_shape
    yy, xx = int(row["h"]), int(row["w"])
    cell_w = image_size / w
    cell_h = image_size / h
    x0 = max(0, int((xx - context_cells) * cell_w))
    x1 = min(image_size, int((xx + context_cells + 1) * cell_w))
    y0 = max(0, int((yy - context_cells) * cell_h))
    y1 = min(image_size, int((yy + context_cells + 1) * cell_h))
    return img.crop((x0, y0, x1, y1)).resize((96, 96), Image.Resampling.NEAREST)


def representative_patch_indices(cloud: TokenCloud, max_points: int = 5000, seed: int = 72) -> dict[str, list[int]]:
    x = cloud.tokens
    norms = np.linalg.norm(x, axis=1)
    idx = sample_indices(len(x), min(max_points, len(x)), seed=seed)
    xs = x[idx]
    nbr_k = min(16, len(xs))
    kth = NearestNeighbors(n_neighbors=nbr_k).fit(xs).kneighbors(xs)[0][:, -1]
    return {
        "high norm": np.argsort(norms)[-4:].tolist(),
        "low norm": np.argsort(norms)[:4].tolist(),
        "dense": idx[np.argsort(kth)[:4]].tolist(),
        "sparse": idx[np.argsort(kth)[-4:]].tolist(),
    }


def show_representative_patches(cloud: TokenCloud, images: list[Image.Image], image_size: int = 256) -> None:
    reps = representative_patch_indices(cloud)
    fig, axes = plt.subplots(len(reps), 4, figsize=(8.2, 2.15 * len(reps)))
    axes = np.asarray(axes)
    context = 2 if cloud.grid_shape[0] >= 16 else 1
    for row_i, (kind, indices) in enumerate(reps.items()):
        for col_i, token_idx in enumerate(indices):
            axes[row_i, col_i].imshow(approximate_patch(cloud, images, token_idx, image_size=image_size, context_cells=context))
            axes[row_i, col_i].set_title(kind, fontsize=9)
            axes[row_i, col_i].axis("off")
    fig.suptitle(f"Approximate source patches: {cloud.name}", y=1.02)
    plt.tight_layout()
    plt.show()
