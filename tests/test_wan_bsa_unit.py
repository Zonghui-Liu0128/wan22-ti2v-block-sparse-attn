import math
import pytest
import torch
import torch.nn.functional as F

from diffsynth.models.wan_video_dit import BlockSparseAttention


def _make_bsa(num_heads=2, block=(2, 2, 2), sparse_ratio=0.5, backend="sdpa_chunked"):
    m = BlockSparseAttention(num_heads, block, sparse_ratio, backend)
    m.eval()
    return m


# --- block info -------------------------------------------------------------


def test_block_info_clean_case():
    m = _make_bsa(block=(2, 2, 2))
    info = m._compute_block_info((4, 4, 4), device=torch.device("cpu"))
    assert info["pad_shape"] == (4, 4, 4)
    assert info["new_grid"] == (2, 2, 2)
    assert info["num_blocks"] == 8
    assert info["block_size_total"] == 8
    # Every block fully populated
    assert torch.equal(
        info["count"],
        torch.full((8,), 8, dtype=torch.long),
    )
    # All padded positions are real
    assert info["valid_mask"].all()


def test_block_info_boundary_case_3_4_5_with_block_2_2_2():
    m = _make_bsa(block=(2, 2, 2))
    info = m._compute_block_info((3, 4, 5), device=torch.device("cpu"))
    assert info["pad_shape"] == (4, 4, 6)
    assert info["new_grid"] == (2, 2, 3)
    assert info["num_blocks"] == 12
    assert info["block_size_total"] == 8

    # Per-block real-token counts (block order: (t_out, h_out, w_out) raster).
    # t_out=0 (eff_t=2): full in t. w_out=0,1 full (eff_w=2), w_out=2 boundary (eff_w=1)
    #   counts at t_out=0: [8, 8, 4,   8, 8, 4]
    # t_out=1 (eff_t=1):
    #   counts at t_out=1: [4, 4, 2,   4, 4, 2]
    expected_count = torch.tensor(
        [8, 8, 4, 8, 8, 4,   4, 4, 2, 4, 4, 2],
        dtype=torch.long,
    )
    assert torch.equal(info["count"], expected_count)
    # Total real tokens = 3*4*5
    assert int(info["count"].sum().item()) == 60
    # valid_mask: padding only at t=3, w=5 inside (4, 4, 6)
    vm = info["valid_mask"]
    assert vm.shape == (4, 4, 6)
    # Real tokens are exactly the (3, 4, 5) sub-box
    expected_vm = torch.zeros(4, 4, 6, dtype=torch.bool)
    expected_vm[:3, :4, :5] = True
    assert torch.equal(vm, expected_vm)


# --- pad/permute round-trip -------------------------------------------------


def test_pad_permute_round_trip_clean():
    torch.manual_seed(0)
    m = _make_bsa(num_heads=2, block=(2, 2, 2))
    B, n, d = 1, 2, 4
    f, h, w = 4, 4, 4
    L = f * h * w
    x = torch.randn(B, n, L, d)
    info = m._compute_block_info((f, h, w), device=x.device)
    blocked = m._pad_and_permute(x, (f, h, w), info)
    assert blocked.shape == (B, n, info["num_blocks"], info["block_size_total"], d)
    restored = m._inverse_permute_and_crop(blocked, (f, h, w), info)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)


def test_pad_permute_round_trip_boundary():
    torch.manual_seed(1)
    m = _make_bsa(num_heads=2, block=(2, 2, 2))
    B, n, d = 1, 2, 4
    f, h, w = 3, 4, 5
    L = f * h * w
    x = torch.randn(B, n, L, d)
    info = m._compute_block_info((f, h, w), device=x.device)
    blocked = m._pad_and_permute(x, (f, h, w), info)
    assert blocked.shape == (B, n, info["num_blocks"], info["block_size_total"], d)
    restored = m._inverse_permute_and_crop(blocked, (f, h, w), info)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)
    # Verify padded slots are zero in the blocked tensor
    vm = info["valid_mask"].reshape(-1)
    # Reshape blocked back to padded flat order (B, n, F_pad*H_pad*W_pad, d) via the inverse rearrange
    F_pad, H_pad, W_pad = info["pad_shape"]
    bt, bh, bw = m.block_size
    nT, nH, nW = info["new_grid"]
    from einops import rearrange as _rearr
    flat_padded = _rearr(
        blocked,
        "b n (tT hH wW) (bt bh bw) d -> b n (tT bt hH bh wW bw) d",
        tT=nT, hH=nH, wW=nW, bt=bt, bh=bh, bw=bw,
    )
    # Padded positions must be exactly zero
    assert torch.equal(
        flat_padded[:, :, ~vm, :],
        torch.zeros_like(flat_padded[:, :, ~vm, :]),
    )


# --- weighted block mean ----------------------------------------------------


def test_block_mean_clean_matches_template_mean():
    """For non-boundary case the weighted mean must equal a plain mean(dim=-2)."""
    torch.manual_seed(2)
    m = _make_bsa(num_heads=2, block=(2, 2, 2))
    B, n, d = 1, 2, 4
    f, h, w = 4, 4, 4
    L = f * h * w
    x = torch.randn(B, n, L, d)
    info = m._compute_block_info((f, h, w), device=x.device)
    blocked = m._pad_and_permute(x, (f, h, w), info)
    mean_weighted = m._block_mean(blocked, info)
    mean_plain = blocked.mean(dim=-2)
    assert torch.allclose(mean_weighted, mean_plain, atol=1e-6)


def test_block_mean_boundary_is_unbiased_real_token_mean():
    """For boundary blocks the weighted mean must equal mean over real tokens only,
    i.e. ignore zero-padded slots in the block."""
    torch.manual_seed(3)
    m = _make_bsa(num_heads=1, block=(2, 2, 2))
    B, n, d = 1, 1, 3
    f, h, w = 3, 4, 5
    L = f * h * w
    x = torch.randn(B, n, L, d)
    info = m._compute_block_info((f, h, w), device=x.device)
    blocked = m._pad_and_permute(x, (f, h, w), info)
    mean_weighted = m._block_mean(blocked, info)  # (B, n, num_blocks, d)

    # Compute reference per-block mean by manually iterating real tokens.
    bt, bh, bw = m.block_size
    nT, nH, nW = info["new_grid"]
    # x as (B, n, f, h, w, d)
    x_vol = x.view(B, n, f, h, w, d)
    expected = torch.zeros(B, n, info["num_blocks"], d)
    block_idx = 0
    for t_out in range(nT):
        for h_out in range(nH):
            for w_out in range(nW):
                t_lo, t_hi = t_out * bt, min(f, (t_out + 1) * bt)
                h_lo, h_hi = h_out * bh, min(h, (h_out + 1) * bh)
                w_lo, w_hi = w_out * bw, min(w, (w_out + 1) * bw)
                sub = x_vol[:, :, t_lo:t_hi, h_lo:h_hi, w_lo:w_hi, :]
                cnt = sub.shape[2] * sub.shape[3] * sub.shape[4]
                expected[:, :, block_idx, :] = sub.reshape(B, n, cnt, d).mean(dim=2)
                block_idx += 1
    assert torch.allclose(mean_weighted, expected, atol=1e-5)
