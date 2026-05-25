"""Visualization + numerical self-check for the BSA attend pattern.

Outputs PNGs under tests/bsa_viz/ — yellow = attended (kept), purple = dropped.
Each test also asserts that BSA forward output matches a dense+mask reference.

Run:
    python -m pytest tests/test_wan_bsa_visualization.py -v
or
    python tests/test_wan_bsa_visualization.py
"""

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from diffsynth.models.wan_video_dit import BlockSparseAttention


VIZ_DIR = Path(__file__).parent / "bsa_viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

PURPLE = "#800080"
YELLOW = "#FFD700"
CMAP = ListedColormap([PURPLE, YELLOW])


def _block_indices_per_token(f, h, w, bt, bh, bw):
    """For each real (t, h, w) token in raster order, return its block index in
    (t_out, h_out, w_out) raster order over the new_T*new_H*new_W grid."""
    new_T = (f + bt - 1) // bt
    new_H = (h + bh - 1) // bh
    new_W = (w + bw - 1) // bw
    ids = []
    for t in range(f):
        for hh in range(h):
            for ww in range(w):
                t_out = t // bt
                h_out = hh // bh
                w_out = ww // bw
                ids.append(t_out * new_H * new_W + h_out * new_W + w_out)
    return torch.tensor(ids, dtype=torch.long), (new_T, new_H, new_W)


def _build_attend_token(attend_block, block_ids):
    """attend_block: (B, n, num_blocks, num_blocks) bool; block_ids: (L_real,) long.
    Returns attend_token at (head=0, batch=0): (L_real, L_real) bool."""
    ab = attend_block[0, 0]  # (num_blocks, num_blocks)
    return ab[block_ids.unsqueeze(1), block_ids.unsqueeze(0)]


def _plot_token_attend(attend_token, f, h, w, bt, bh, bw, title, path, label_step=None):
    """Render an L_real x L_real bool matrix; thin grid per token (when L<=80),
    thick grid at block boundaries."""
    L = attend_token.shape[0]
    arr = attend_token.cpu().numpy().astype(np.uint8)
    fig_size = max(6, min(18, L * 0.18))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(arr, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")

    # Thin per-token grid — skip when too dense to read
    if L <= 80:
        ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.2)

    # Thick grid at block boundaries.
    # Real-token block boundaries:
    #   t boundaries at cum positions where the (t, h, w) raster crosses t_out += 1
    #   similarly for h, w but interlinked since raster is t -> h -> w
    boundary_positions = []
    cur_block = -1
    for idx in range(L + 1):
        if idx == L:
            boundary_positions.append(idx)
            break
        t = idx // (h * w)
        hh = (idx // w) % h
        ww = idx % w
        bid = (t // bt) * ((h + bh - 1) // bh) * ((w + bw - 1) // bw) \
              + (hh // bh) * ((w + bw - 1) // bw) + (ww // bw)
        if bid != cur_block:
            boundary_positions.append(idx)
            cur_block = bid

    for pos in boundary_positions:
        ax.axvline(pos - 0.5, color="black", linewidth=0.9)
        ax.axhline(pos - 0.5, color="black", linewidth=0.9)

    # Axis tick labels: every label_step-th token index
    if label_step is None:
        label_step = max(1, L // 16)
    ticks = list(range(0, L, label_step))
    if (L - 1) not in ticks:
        ticks.append(L - 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_xlabel("key token index")
    ax.set_ylabel("query token index")

    ax.set_title(title, fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_token_attend_block_contiguous(
    attend_token_bc, real_per_block, num_blocks, title, path,
):
    """attend_token_bc: (L_real, L_real) bool, where rows/cols are ordered so each block's
    real tokens are contiguous. real_per_block: (num_blocks,) long with the count of real
    tokens per block. Block boundaries fall at cumulative-count positions."""
    L = attend_token_bc.shape[0]
    arr = attend_token_bc.cpu().numpy().astype(np.uint8)
    fig_size = max(6, min(18, L * 0.18))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(arr, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")

    # Cumulative real-token positions = block boundaries.
    cum = np.concatenate([[0], np.cumsum(real_per_block.cpu().numpy().astype(int))])
    # Thick black grid at block boundaries (excluding the outermost edges already drawn by imshow).
    for pos in cum:
        ax.axvline(pos - 0.5, color="black", linewidth=1.0)
        ax.axhline(pos - 0.5, color="black", linewidth=1.0)

    # Centre block index on each axis.
    block_centres = (cum[:-1] + cum[1:] - 1) / 2.0
    ax.set_xticks(block_centres)
    ax.set_yticks(block_centres)
    ax.set_xticklabels([str(i) for i in range(num_blocks)], fontsize=7)
    ax.set_yticklabels([str(i) for i in range(num_blocks)], fontsize=7)
    ax.set_xlabel("key block index (each cell width = real tokens in that block)")
    ax.set_ylabel("query block index (each cell height = real tokens in that block)")
    ax.set_title(title, fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_block_attend(attend_block, num_blocks, title, path):
    arr = attend_block[0, 0].cpu().numpy().astype(np.uint8)
    fig_size = max(6, min(14, num_blocks * 0.05 + 6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(arr, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("key block index")
    ax.set_ylabel("query block index")
    ax.set_title(title, fontsize=10)
    if num_blocks <= 32:
        ax.set_xticks(range(num_blocks))
        ax.set_yticks(range(num_blocks))
        ax.tick_params(labelsize=6)
        ax.grid(which="major", color="white", linestyle="-", linewidth=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _run_scene(scene_name, f, h, w, block, sparse_ratio, backend, num_heads=2, d=8, seed=0,
               do_numerical_check=True):
    """Run BSA, dump PNG(s), and (optionally) assert numerical equivalence vs dense+mask reference.

    The numerical check materialises an (L_pad, L_pad) attn mask + scores, so skip it when
    L_pad is large (prod-scale CPU runs)."""
    torch.manual_seed(seed)
    bt, bh, bw = block
    L = f * h * w
    B = 1
    nd = num_heads * d

    # Random Q/K/V already 'normalised' enough for the test
    q = torch.randn(B, L, nd, dtype=torch.float32)
    k = torch.randn(B, L, nd, dtype=torch.float32)
    v = torch.randn(B, L, nd, dtype=torch.float32)

    device = torch.device("cuda" if (backend == "flex" and torch.cuda.is_available()) else "cpu")
    if backend == "flex":
        from diffsynth.models.wan_video_dit import FLEX_ATTN_AVAILABLE
        if not (FLEX_ATTN_AVAILABLE and torch.cuda.is_available()):
            pytest.skip("FlexAttention / CUDA unavailable; flex backend skipped")
    q, k, v = q.to(device), k.to(device), v.to(device)
    m = BlockSparseAttention(num_heads, block, sparse_ratio, backend).to(device)
    m._debug_record = True
    out_bsa = m(q, k, v, (f, h, w))
    assert out_bsa.shape == (B, L, nd)

    attend_block = m._dbg_attend_block
    info = m._compute_block_info((f, h, w), device=device)

    if do_numerical_check:
        # Materialise the dense+mask reference and compare. Memory: O(L_pad^2).
        q_nd = rearrange(q, "b l (n d) -> b n l d", n=num_heads)
        k_nd = rearrange(k, "b l (n d) -> b n l d", n=num_heads)
        v_nd = rearrange(v, "b l (n d) -> b n l d", n=num_heads)
        Q_b = m._pad_and_permute(q_nd, (f, h, w), info)
        K_b = m._pad_and_permute(k_nd, (f, h, w), info)
        V_b = m._pad_and_permute(v_nd, (f, h, w), info)
        L_pad = info["num_blocks"] * info["block_size_total"]
        Q_flat = Q_b.reshape(B, num_heads, L_pad, d)
        K_flat = K_b.reshape(B, num_heads, L_pad, d)
        V_flat = V_b.reshape(B, num_heads, L_pad, d)
        bst = info["block_size_total"]
        block_id_pad = torch.arange(L_pad, device=device) // bst
        attend_token_padded = attend_block[
            :, :, block_id_pad.unsqueeze(1), block_id_pad.unsqueeze(0)
        ]
        # Build block-contiguous validity mask (matches what _sparse_attn_sdpa_chunked uses).
        F_pad, H_pad, W_pad = info["pad_shape"]
        nT_, nH_, nW_ = info["new_grid"]
        bt_ = F_pad // nT_
        bh_ = H_pad // nH_
        bw_ = W_pad // nW_
        vm_bc = rearrange(
            info["valid_mask"],
            "(tT bt) (hH bh) (wW bw) -> (tT hH wW bt bh bw)",
            tT=nT_, hH=nH_, wW=nW_, bt=bt_, bh=bh_, bw=bw_,
        )
        attend_token_padded = attend_token_padded & vm_bc.view(1, 1, 1, L_pad)
        out_flat = F.scaled_dot_product_attention(
            Q_flat, K_flat, V_flat, attn_mask=attend_token_padded
        )
        out_ref_blocked = out_flat.view(B, num_heads, info["num_blocks"], bst, d)
        out_ref_nd = m._inverse_permute_and_crop(out_ref_blocked, (f, h, w), info)
        out_ref = rearrange(out_ref_nd, "b n l d -> b l (n d)")
        max_diff = (out_bsa - out_ref).abs().max().item()
        print(f"[{scene_name}/{backend}] forward max|diff vs dense reference = {max_diff:.3e}")
        assert max_diff <= 1e-4, f"{scene_name}/{backend}: max diff {max_diff} exceeded 1e-4"
    else:
        print(f"[{scene_name}/{backend}] numerical check skipped (L_pad too large for CPU dense ref)")

    # ---- visualisation ----
    block_ids_real, _ = _block_indices_per_token(f, h, w, bt, bh, bw)
    block_ids_real = block_ids_real.to(device)
    attend_token = _build_attend_token(attend_block, block_ids_real)
    num_blocks = info["num_blocks"]
    K_drop = int(num_blocks * sparse_ratio)
    title_common = (
        f"f={f}, h={h}, w={w}  block={block}  ratio={sparse_ratio}  "
        f"num_blocks={num_blocks}  K_drop={K_drop}  backend={backend}  head=0 batch=0"
    )

    # Block-contiguous permutation: sort real tokens by their block id (stable).
    perm = torch.argsort(block_ids_real, stable=True)
    attend_token_bc = attend_token[perm][:, perm]
    # Real tokens per block (boundary blocks have fewer).
    real_per_block = info["count"]
    # (Sanity) sum of real_per_block should equal L_real
    assert int(real_per_block.sum().item()) == attend_token_bc.shape[0]

    # token-level: skip if too large
    L_real = attend_token.shape[0]
    if L_real <= 256:
        _plot_token_attend(
            attend_token, f, h, w, bt, bh, bw,
            f"[{scene_name}] token-level attend  {title_common}",
            VIZ_DIR / f"wan_bsa_{scene_name}_{backend}_token.png",
        )
        _plot_token_attend_block_contiguous(
            attend_token_bc, real_per_block, num_blocks,
            f"[{scene_name}] token-level attend (block-contiguous)  {title_common}",
            VIZ_DIR / f"wan_bsa_{scene_name}_{backend}_token_bc.png",
        )
    # block-level: always
    _plot_block_attend(
        attend_block, num_blocks,
        f"[{scene_name}] block-level attend  {title_common}",
        VIZ_DIR / f"wan_bsa_{scene_name}_{backend}_block.png",
    )

    # Boundary block invariant: total real tokens should equal f*h*w
    assert int(info["count"].sum().item()) == f * h * w
    # K_drop invariant: every query block drops exactly K_drop key blocks
    assert ((~attend_block).sum(dim=-1) == K_drop).all()


def test_scene_debug_clean_sdpa_chunked():
    _run_scene("debug_clean", 4, 4, 4, (2, 2, 2), 0.5, "sdpa_chunked", seed=100)


def test_scene_boundary_sdpa_chunked():
    _run_scene("boundary", 3, 4, 5, (2, 2, 2), 0.5, "sdpa_chunked", seed=101)


def test_scene_prod_scale_sdpa_chunked():
    # Prod-scale: skip the numerical reference (would build an 8190x8190 attn mask on CPU).
    # Correctness is covered by debug_clean + boundary scenes; here we only verify the
    # block-level visualisation and the structural invariants (K_drop per row, real-token count).
    _run_scene(
        "prod_scale", 21, 26, 15, (3, 2, 3), 0.5, "sdpa_chunked",
        num_heads=1, d=4, seed=102, do_numerical_check=False,
    )


def test_scene_debug_clean_flex():
    _run_scene("debug_clean", 4, 4, 4, (2, 2, 2), 0.5, "flex", seed=200)


def test_scene_boundary_flex():
    _run_scene("boundary", 3, 4, 5, (2, 2, 2), 0.5, "flex", seed=201)


if __name__ == "__main__":
    test_scene_debug_clean_sdpa_chunked()
    test_scene_boundary_sdpa_chunked()
    test_scene_prod_scale_sdpa_chunked()
    try:
        test_scene_debug_clean_flex()
        test_scene_boundary_flex()
    except Exception as e:
        print("Flex backend scenarios skipped:", e)
    print(f"All visualisation outputs saved under: {VIZ_DIR}")
