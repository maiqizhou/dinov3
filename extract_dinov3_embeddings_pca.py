#!/usr/bin/env python3
"""
Extract frozen DINOv3 embeddings for CAMELYON16 patches
and visualize tumor vs normal separation using PCA.

Supports DINOv3 backbones: ViT-S/16 (vits16), ViT-B/16 (vitb16),
ViT-H/16+ (vith16plus), ViT-L/16 (vitl16).
NO classifier, NO training.
"""

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# -------------------------
# Data structure
# -------------------------
@dataclass(frozen=True)
class PatchItem:
    path: str
    slide_id: str
    label: int  # 0 = normal, 1 = tumor


# -------------------------
# Utility
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def iter_slide_dirs(root: str, *, prefixes: Optional[Sequence[str]] = None) -> List[Path]:
    p = Path(root)
    if not p.is_dir():
        raise FileNotFoundError(root)

    dirs = [d for d in p.iterdir() if d.is_dir()]
    if prefixes:
        prefixes = tuple(prefixes)
        dirs = [d for d in dirs if d.name.startswith(prefixes)]
    dirs.sort(key=lambda x: x.name)
    return dirs


def list_images(dir_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    imgs = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    imgs.sort()
    return imgs


def sample_patches(
    slide_dirs: Sequence[Path],
    *,
    label: int,
    max_total: int,
    max_per_slide: Optional[int],
    seed: int,
) -> List[PatchItem]:
    rng = random.Random(seed)
    items: List[PatchItem] = []

    for slide_dir in slide_dirs:
        imgs = list_images(slide_dir)
        if not imgs:
            continue

        if max_per_slide is not None and len(imgs) > max_per_slide:
            imgs = rng.sample(imgs, max_per_slide)
            imgs.sort()

        for p in imgs:
            items.append(PatchItem(str(p), slide_dir.name, label))

    if len(items) > max_total:
        items = rng.sample(items, max_total)
        items.sort(key=lambda it: it.path)

    return items


# -------------------------
# Image preprocessing
# -------------------------
def pil_to_tensor(pil_img, *, out_size: int):
    import numpy as np
    import torch
    from PIL import Image

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    if pil_img.size != (out_size, out_size):
        # Pillow compatibility: some versions prefer Image.BICUBIC, newer have Image.Resampling.BICUBIC
        resampling = getattr(Image, "Resampling", Image)
        pil_img = pil_img.resize((out_size, out_size), resample=resampling.BICUBIC)

    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    x = torch.from_numpy(arr)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    return (x - mean) / std


# -------------------------
# DINOv3 embedding extraction
# -------------------------
def extract_embeddings(
    items: Sequence[PatchItem],
    *,
    weights_path: str,
    dinov3_repo: str,
    model_name: str,
    batch_size: int,
    input_size: int,
    device: str,
):
    import numpy as np
    import torch
    from PIL import Image

    dinov3_repo = str(Path(dinov3_repo).resolve())
    if dinov3_repo not in sys.path:
        sys.path.insert(0, dinov3_repo)

    # Use torch.hub and dinov3's hubconf backbones; load local weights via file://
    import torch.hub as _hub
    name_map = {
        "vits16": "dinov3_vits16",
        "vitb16": "dinov3_vitb16",
        "vith16plus": "dinov3_vith16plus",
        "vitl16": "dinov3_vitl16",
    }
    if model_name not in name_map:
        raise ValueError(
            f"Unsupported model_name: {model_name}. Use 'vits16', 'vitb16', 'vith16plus', or 'vitl16'."
        )
    # Build model skeleton without downloading
    model = _hub.load(repo_or_dir=dinov3_repo, model=name_map[model_name], source="local", pretrained=False)
    # Load local weights
    url = weights_path if weights_path.startswith("file://") else f"file://{weights_path}"
    sd = _hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded with non-strict: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    model.to(device)

    X = []
    y = []
    paths = []
    slide_ids = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        imgs = []

        for it in batch:
            img = Image.open(it.path)
            imgs.append(pil_to_tensor(img, out_size=input_size))

        x = torch.stack(imgs).to(device)

        with torch.inference_mode():
            feats = model.forward_features(x)
            cls = feats["x_norm_clstoken"].float().cpu().numpy()
            X.append(cls)

        y.extend([it.label for it in batch])
        paths.extend([it.path for it in batch])
        slide_ids.extend([it.slide_id for it in batch])

    return np.concatenate(X), np.array(y), paths, slide_ids


# -------------------------
# PCA + visualization
# -------------------------
def run_pca_and_plot(X, y, out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required to save the PCA scatter plot. "
            "Install it in your dinov3 env, e.g.:\n"
            "  conda run -p /home/ubuntu/data/conda_envs/dinov3 pip install matplotlib\n"
            f"Original error: {type(e).__name__}: {e}"
        )
    import numpy as np
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        Z[y == 0, 0],
        Z[y == 0, 1],
        s=8,
        alpha=0.6,
        label="Normal",
    )
    plt.scatter(
        Z[y == 1, 0],
        Z[y == 1, 1],
        s=8,
        alpha=0.6,
        label="Tumor ROI",
    )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("DINOv3 Patch Embeddings (PCA)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=300)
    plt.close()

    return Z, pca.explained_variance_ratio_


def run_multi_model_compare(model_to_results, out_dir: str):
    """
    Create a 2x2 comparison scatter plot across models.
    model_to_results: dict {model_name: (Z, y)}
    """
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    # Determine global axis limits for fair comparison
    xs = []
    ys = []
    for (Z, y) in model_to_results.values():
        xs.append(Z[:, 0])
        ys.append(Z[:, 1])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Order models for layout
    order = [m for m in ["vits16", "vitb16", "vith16plus", "vitl16"] if m in model_to_results]
    n = len(order)
    rows = 2
    cols = 2 if n >= 2 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    axes = np.atleast_1d(axes).flatten()

    titles = {
        "vits16": "ViT-S/16",
        "vitb16": "ViT-B/16",
        "vith16plus": "ViT-H/16+",
        "vitl16": "ViT-L/16",
    }

    for i, m in enumerate(order):
        ax = axes[i]
        Z, y = model_to_results[m]
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=6, alpha=0.6, label="Normal")
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=6, alpha=0.6, label="Tumor ROI")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(titles.get(m, m))
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_pca_scatter.png"), dpi=300)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Frozen DINOv3 embedding extraction + PCA visualization (NO training)"
    )

    # Single-model run (backward compatible)
    parser.add_argument("--weights", help="Path to model weights (.pth) matching --model")

    parser.add_argument(
        "--dinov3_repo",
        default="/home/ubuntu/dinov3",
        help="Path to local dinov3 repo root (so dinov3 can be imported)",
    )

    parser.add_argument(
        "--model",
        choices=["vits16", "vitb16", "vith16plus", "vitl16"],
        help="DINOv3 backbone to use for single run",
    )

    # Multi-model comparison
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["vits16", "vitb16", "vith16plus", "vitl16"],
        help="Run PCA for multiple backbones and generate a comparison figure",
    )
    parser.add_argument("--weights_vits16", help="Weights for vits16 (.pth)")
    parser.add_argument("--weights_vitb16", help="Weights for vitb16 (.pth)")
    parser.add_argument("--weights_vith16plus", help="Weights for vith16plus (.pth)")
    parser.add_argument("--weights_vitl16", help="Weights for vitl16 (.pth)")

    parser.add_argument("--normal_root", required=True)
    parser.add_argument("--tumor_roi_root", required=True)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--max_normal", type=int, default=5000)
    parser.add_argument("--max_tumor", type=int, default=5000)
    parser.add_argument("--max_per_slide", type=int, default=300)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torch_home",
        default="/home/ubuntu/data/tmp/torch_cache",
        help="Directory for torch hub cache to avoid low-space defaults",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # Redirect torch hub cache directory if provided
    os.makedirs(args.torch_home, exist_ok=True)
    os.environ["TORCH_HOME"] = args.torch_home

    normal_slides = iter_slide_dirs(args.normal_root, prefixes=["normal_"])
    tumor_slides = iter_slide_dirs(args.tumor_roi_root, prefixes=["tumor_"])

    normal_items = sample_patches(
        normal_slides,
        label=0,
        max_total=args.max_normal,
        max_per_slide=args.max_per_slide,
        seed=args.seed,
    )

    tumor_items = sample_patches(
        tumor_slides,
        label=1,
        max_total=args.max_tumor,
        max_per_slide=args.max_per_slide,
        seed=args.seed + 1,
    )

    items = normal_items + tumor_items
    random.shuffle(items)

    # If --models provided, run multi-model comparison
    if args.models:
        model_to_weights = {}
        if "vits16" in args.models:
            if not args.weights_vits16:
                raise SystemExit("--weights_vits16 is required when selecting vits16")
            model_to_weights["vits16"] = args.weights_vits16
        if "vitb16" in args.models:
            if not args.weights_vitb16:
                raise SystemExit("--weights_vitb16 is required when selecting vitb16")
            model_to_weights["vitb16"] = args.weights_vitb16
        if "vith16plus" in args.models:
            if not args.weights_vith16plus:
                raise SystemExit("--weights_vith16plus is required when selecting vith16plus")
            model_to_weights["vith16plus"] = args.weights_vith16plus
        if "vitl16" in args.models:
            if not args.weights_vitl16:
                raise SystemExit("--weights_vitl16 is required when selecting vitl16")
            model_to_weights["vitl16"] = args.weights_vitl16

        compare_results = {}
        for m, w in model_to_weights.items():
            sub_out = os.path.join(args.out_dir, m)
            X, y, paths, slide_ids = extract_embeddings(
                items,
                weights_path=w,
                dinov3_repo=args.dinov3_repo,
                model_name=m,
                batch_size=args.batch_size,
                input_size=args.input_size,
                device=args.device,
            )
            Z, evr = run_pca_and_plot(X, y, sub_out)
            # Save per-model CSV
            with open(os.path.join(sub_out, "embeddings_pca.csv"), "w", newline="") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["path", "slide_id", "label", "pca_x", "pca_y"])
                for i in range(len(y)):
                    wcsv.writerow([paths[i], slide_ids[i], int(y[i]), Z[i, 0], Z[i, 1]])
            print(f"{m}: n={len(y)} normal={(y==0).sum()} tumor={(y==1).sum()} PCA EVR={evr.tolist()}")
            compare_results[m] = (Z, y)

        # Combined comparison figure
        run_multi_model_compare(compare_results, args.out_dir)
        print(f"Saved comparison figure to {os.path.join(args.out_dir, 'compare_pca_scatter.png')}")
        return

    # Single-model path
    if not args.model or not args.weights:
        raise SystemExit("For single run, please provide both --model and --weights. Alternatively, use --models ...")

    X, y, paths, slide_ids = extract_embeddings(
        items,
        weights_path=args.weights,
        dinov3_repo=args.dinov3_repo,
        model_name=args.model,
        batch_size=args.batch_size,
        input_size=args.input_size,
        device=args.device,
    )

    Z, evr = run_pca_and_plot(X, y, args.out_dir)

    # Save PCA coordinates
    with open(os.path.join(args.out_dir, "embeddings_pca.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "slide_id", "label", "pca_x", "pca_y"])
        for i in range(len(y)):
            w.writerow([paths[i], slide_ids[i], int(y[i]), Z[i, 0], Z[i, 1]])

    print(
        f"Done. n={len(y)}  normal={(y==0).sum()}  tumor={(y==1).sum()} "
        f"PCA EVR={evr.tolist()}"
    )


if __name__ == "__main__":
    main()
