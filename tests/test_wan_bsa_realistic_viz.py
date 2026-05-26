"""Realistic-scale BSA attention map visualisation at 80% sparsity.

Two figures are produced:

 1. `wan_bsa_realistic_80pct_block.png`
    Block-level attend pattern (num_blocks x num_blocks bool) at
    (f, h, w) = (13, 52, 30), block = (2, 4, 4), sparse_ratio = 0.8.
    This matches the latent shape of Wan2.2-TI2V-5B at 832x480 / 49 frames /
    patch_size (1, 2, 2).

 2. `wan_bsa_realistic_80pct_block_smallcase.png`
    Block-level attend pattern at the spec's prod-scale (21, 26, 15) /
    block = (3, 2, 3), sparse_ratio = 0.8 -- same recipe used in spec Section 6.

Q/K are not raw IID gaussian. To approximate the spatial coherence that real
attention maps exhibit, each block's Q/K vector is a sum of a slowly varying
per-(t,h,w) component plus a small noise term -- this gives a similarity
matrix where temporally/spatially adjacent blocks are MORE similar, so the
top-K dropping picks "far" blocks first, producing a recognisable spatial
locality pattern in the kept set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from diffsynth.models.wan_video_dit import BlockSparseAttention


PURPLE = "#800080"
YELLOW = "#FFD700"
CMAP = ListedColormap([PURPLE, YELLOW])


def make_correlated_qk(f, h, w, n_heads, head_dim, device, dtype=torch.float32, seed=0):
    """Build (1, n_heads, L=f*h*w, head_dim) Q and K tensors where adjacent
    spatial positions share more signal than distant ones."""
    g = torch.Generator(device=device).manual_seed(seed)
    # Low-frequency component: depends smoothly on (t, h, w).
    t = torch.arange(f, device=device, dtype=dtype)
    yy = torch.arange(h, device=device, dtype=dtype)
    xx = torch.arange(w, device=device, dtype=dtype)
    tg, yg, xg = torch.meshgrid(t, yy, xx, indexing="ij")
    # 8 "frequency components" with random projections.
    freqs = torch.randn(8, 3, n_heads, head_dim, generator=g, device=device, dtype=dtype) * 0.5
    coords = torch.stack([tg.flatten() / max(f - 1, 1), yg.flatten() / max(h - 1, 1), xg.flatten() / max(w - 1, 1)], dim=-1)
    # Build a structured signal: sum over components of sin(2 pi (k . coord)) * proj.
    structured = torch.zeros(f * h * w, n_heads, head_dim, device=device, dtype=dtype)
    for k_idx in range(freqs.shape[0]):
        k_vec = freqs[k_idx]  # (3, n_heads, head_dim)
        phase = (coords @ k_vec.reshape(3, -1)).reshape(-1, n_heads, head_dim) * 2 * np.pi
        structured = structured + torch.sin(phase) * 0.4
    # IID noise component.
    noise_q = torch.randn(f * h * w, n_heads, head_dim, generator=g, device=device, dtype=dtype) * 0.6
    noise_k = torch.randn(f * h * w, n_heads, head_dim, generator=g, device=device, dtype=dtype) * 0.6
    q = (structured + noise_q).permute(1, 0, 2).unsqueeze(0)  # (1, n_heads, L, head_dim)
    k = (structured + noise_k).permute(1, 0, 2).unsqueeze(0)
    return q, k


def render_block_attend(attend_block, video_shape, block_size, sparse_ratio, title, path,
                        with_token_axes=False, info=None):
    arr = attend_block[0, 0].cpu().to(torch.int8).numpy()
    num_blocks = arr.shape[0]
    K_drop = int(num_blocks * sparse_ratio)
    f, h, w = video_shape
    fig_size = max(8, min(20, num_blocks * 0.015 + 6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(arr, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("key block index")
    ax.set_ylabel("query block index")
    ax.set_title(
        f"{title}\n"
        f"f={f}, h={h}, w={w}, block={tuple(block_size)}, ratio={sparse_ratio}, "
        f"num_blocks={num_blocks}, num_keep={num_blocks - K_drop}, K_drop={K_drop}, head=0, batch=0",
        fontsize=9, pad=12,
    )
    if num_blocks <= 40:
        ax.set_xticks(range(num_blocks))
        ax.set_yticks(range(num_blocks))
        ax.tick_params(labelsize=6)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {path}  num_blocks={num_blocks}  K_drop={K_drop}  num_keep={num_blocks - K_drop}")


def run_scene(name, video_shape, block_size, sparse_ratio, n_heads, head_dim, device, seed, out_dir):
    """Compute attend_block for a single (video_shape, block_size, sparse_ratio) and save PNG."""
    f, h, w = video_shape
    L = f * h * w
    bsa = BlockSparseAttention(n_heads, block_size, sparse_ratio, backend="sdpa_chunked").to(device)
    info = bsa._compute_block_info(video_shape, device=device)

    q, k = make_correlated_qk(f, h, w, n_heads, head_dim, device=device, seed=seed)
    # _pad_and_permute expects (B, n, L, d)
    Q_b = bsa._pad_and_permute(q, video_shape, info)
    K_b = bsa._pad_and_permute(k, video_shape, info)
    Q_mean = bsa._block_mean(Q_b, info)
    K_mean = bsa._block_mean(K_b, info)
    attend_block = bsa._compute_attend_block(Q_mean, K_mean)

    out_path = Path(out_dir) / f"wan_bsa_realistic_{name}.png"
    render_block_attend(
        attend_block,
        video_shape, block_size, sparse_ratio,
        title=f"Wan2.2-5B SelfAttention BSA  '{name}'  (correlated Q/K)",
        path=out_path,
    )
    # Sanity invariants
    num_blocks = info["num_blocks"]
    K_drop = int(num_blocks * sparse_ratio)
    assert ((~attend_block).sum(dim=-1) == K_drop).all(), (
        f"each query block should drop exactly {K_drop} keys"
    )
    assert int(info["count"].sum().item()) == f * h * w
    return out_path, attend_block, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="tests/bsa_viz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Scene 1: Wan2.2-TI2V 832x480, 49 frames -> latent (13, 52, 30) after patch (1,2,2)
    # block (2, 4, 4): boundary in t and w
    run_scene(
        "80pct_realistic_13_52_30",
        video_shape=(13, 52, 30),
        block_size=(2, 4, 4),
        sparse_ratio=0.8,
        n_heads=4, head_dim=128,
        device=device, seed=args.seed,
        out_dir=out_dir,
    )

    # Scene 2: spec's prod_scale shape with 80% sparsity (for reference vs the 50% in unit tests)
    run_scene(
        "80pct_prod_21_26_15",
        video_shape=(21, 26, 15),
        block_size=(3, 2, 3),
        sparse_ratio=0.8,
        n_heads=4, head_dim=128,
        device=device, seed=args.seed,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
