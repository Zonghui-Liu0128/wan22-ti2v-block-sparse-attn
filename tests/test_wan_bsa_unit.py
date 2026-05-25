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


# --- attend_block construction ---------------------------------------------


def test_attend_block_drops_least_similar():
    """sparse_ratio == drop fraction; (-sim).topk(K_drop) selects the K_drop SMALLEST sim
    values per row, those positions must be False in attend_block. The rest must be True.
    """
    torch.manual_seed(4)
    m = _make_bsa(num_heads=1, block=(2, 2, 2), sparse_ratio=0.5)
    B, n, num_blocks, d = 1, 1, 8, 4
    Q_block = torch.randn(B, n, num_blocks, d)
    K_block = torch.randn(B, n, num_blocks, d)
    attend_block = m._compute_attend_block(Q_block, K_block)
    assert attend_block.shape == (B, n, num_blocks, num_blocks)
    assert attend_block.dtype == torch.bool

    sim = Q_block @ K_block.transpose(-1, -2)
    K_drop = int(num_blocks * 0.5)
    assert ((~attend_block).sum(dim=-1) == K_drop).all()
    for i in range(num_blocks):
        # The K_drop smallest sim values must correspond to attend_block == False
        row_sim = sim[0, 0, i]
        sorted_idx = row_sim.argsort()
        drop_set = set(sorted_idx[:K_drop].tolist())
        false_set = set((~attend_block[0, 0, i]).nonzero(as_tuple=True)[0].tolist())
        assert drop_set == false_set


def test_attend_block_sparse_ratio_zero_keeps_everything():
    m = _make_bsa(num_heads=1, block=(2, 2, 2), sparse_ratio=0.0)
    torch.manual_seed(5)
    Q_block = torch.randn(1, 1, 6, 4)
    K_block = torch.randn(1, 1, 6, 4)
    attend_block = m._compute_attend_block(Q_block, K_block)
    assert attend_block.all()


# --- sdpa_chunked backend vs dense reference -------------------------------


def _dense_masked_attention(
    Q_blocked, K_blocked, V_blocked, attend_block, info
):
    """Reference: build the full padded (L_pad, L_pad) bool mask, run SDPA, return
    block-shaped output."""
    B, n, num_blocks, bst, d = Q_blocked.shape
    L_pad = num_blocks * bst
    Q_flat = Q_blocked.reshape(B, n, L_pad, d)
    K_flat = K_blocked.reshape(B, n, L_pad, d)
    V_flat = V_blocked.reshape(B, n, L_pad, d)
    # Build per-token attend mask of shape (L_pad, L_pad)
    block_id = torch.arange(L_pad, device=Q_flat.device) // bst   # (L_pad,)
    # attend_block: (B, n, num_blocks, num_blocks) bool
    # token-level: attend_token[b, n, i, j] = attend_block[b, n, block_id[i], block_id[j]]
    attend_token = attend_block[
        :, :, block_id.unsqueeze(1), block_id.unsqueeze(0)
    ]  # (B, n, L_pad, L_pad)
    # Mask out padded key positions everywhere. Q_flat/K_flat are in block-contiguous
    # order, so we need valid_mask flattened in the SAME order.
    from einops import rearrange
    F_pad, H_pad, W_pad = info["pad_shape"]
    nT_, nH_, nW_ = info["new_grid"]
    bt_ = F_pad // nT_
    bh_ = H_pad // nH_
    bw_ = W_pad // nW_
    vm_bc = rearrange(
        info["valid_mask"],
        "(tT bt) (hH bh) (wW bw) -> (tT hH wW bt bh bw)",
        tT=nT_, hH=nH_, wW=nW_, bt=bt_, bh=bh_, bw=bw_,
    )  # (L_pad,) bool, block-contiguous
    attend_token = attend_token & vm_bc.view(1, 1, 1, L_pad)
    out_flat = F.scaled_dot_product_attention(
        Q_flat, K_flat, V_flat, attn_mask=attend_token
    )
    return out_flat.view(B, n, num_blocks, bst, d)


def test_sdpa_chunked_backend_matches_dense_reference_clean():
    torch.manual_seed(6)
    m = _make_bsa(num_heads=2, block=(2, 2, 2), sparse_ratio=0.5, backend="sdpa_chunked")
    B, n, d = 1, 2, 4
    f, h, w = 4, 4, 4
    L = f * h * w
    q = torch.randn(B, n, L, d, dtype=torch.float32)
    k = torch.randn(B, n, L, d, dtype=torch.float32)
    v = torch.randn(B, n, L, d, dtype=torch.float32)

    info = m._compute_block_info((f, h, w), device=q.device)
    Q_b = m._pad_and_permute(q, (f, h, w), info)
    K_b = m._pad_and_permute(k, (f, h, w), info)
    V_b = m._pad_and_permute(v, (f, h, w), info)
    Q_mean = m._block_mean(Q_b, info)
    K_mean = m._block_mean(K_b, info)
    attend_block = m._compute_attend_block(Q_mean, K_mean)

    out_bsa = m._sparse_attn_sdpa_chunked(Q_b, K_b, V_b, attend_block, info)
    out_ref = _dense_masked_attention(Q_b, K_b, V_b, attend_block, info)
    assert torch.allclose(out_bsa, out_ref, atol=1e-5)


def test_sdpa_chunked_backend_matches_dense_reference_boundary():
    torch.manual_seed(7)
    m = _make_bsa(num_heads=2, block=(2, 2, 2), sparse_ratio=0.5, backend="sdpa_chunked")
    B, n, d = 1, 2, 4
    f, h, w = 3, 4, 5
    L = f * h * w
    q = torch.randn(B, n, L, d, dtype=torch.float32)
    k = torch.randn(B, n, L, d, dtype=torch.float32)
    v = torch.randn(B, n, L, d, dtype=torch.float32)
    info = m._compute_block_info((f, h, w), device=q.device)
    Q_b = m._pad_and_permute(q, (f, h, w), info)
    K_b = m._pad_and_permute(k, (f, h, w), info)
    V_b = m._pad_and_permute(v, (f, h, w), info)
    Q_mean = m._block_mean(Q_b, info)
    K_mean = m._block_mean(K_b, info)
    attend_block = m._compute_attend_block(Q_mean, K_mean)
    out_bsa = m._sparse_attn_sdpa_chunked(Q_b, K_b, V_b, attend_block, info)
    out_ref = _dense_masked_attention(Q_b, K_b, V_b, attend_block, info)
    # In boundary case, padded query rows may differ (they read padded keys whose validity
    # mask excludes them); we only require the **real-token** outputs to match.
    # Use the inverse permute + crop to extract real-token outputs from both.
    out_bsa_real = m._inverse_permute_and_crop(out_bsa, (f, h, w), info)
    out_ref_real = m._inverse_permute_and_crop(out_ref, (f, h, w), info)
    assert torch.allclose(out_bsa_real, out_ref_real, atol=1e-5)


# --- flex backend cross-check (CUDA + flex_attention only) ------------------

from diffsynth.models.wan_video_dit import FLEX_ATTN_AVAILABLE


_flex_skip_reason = "FlexAttention or CUDA unavailable"
_flex_skip = pytest.mark.skipif(
    not (FLEX_ATTN_AVAILABLE and torch.cuda.is_available()),
    reason=_flex_skip_reason,
)


@_flex_skip
def test_flex_backend_matches_sdpa_chunked_clean():
    torch.manual_seed(8)
    f, h, w = 4, 4, 4
    L = f * h * w
    device = torch.device("cuda")
    B, n, d = 1, 2, 4
    q = torch.randn(B, n, L, d, dtype=torch.float32, device=device)
    k = torch.randn(B, n, L, d, dtype=torch.float32, device=device)
    v = torch.randn(B, n, L, d, dtype=torch.float32, device=device)
    m_sdpa = _make_bsa(num_heads=n, block=(2, 2, 2), sparse_ratio=0.5, backend="sdpa_chunked").to(device)
    m_flex = _make_bsa(num_heads=n, block=(2, 2, 2), sparse_ratio=0.5, backend="flex").to(device)

    info = m_sdpa._compute_block_info((f, h, w), device=device)
    Q_b = m_sdpa._pad_and_permute(q, (f, h, w), info)
    K_b = m_sdpa._pad_and_permute(k, (f, h, w), info)
    V_b = m_sdpa._pad_and_permute(v, (f, h, w), info)
    Q_mean = m_sdpa._block_mean(Q_b, info)
    K_mean = m_sdpa._block_mean(K_b, info)
    attend_block = m_sdpa._compute_attend_block(Q_mean, K_mean)
    out_sdpa = m_sdpa._sparse_attn_sdpa_chunked(Q_b, K_b, V_b, attend_block, info)
    out_flex = m_flex._sparse_attn_flex(Q_b, K_b, V_b, attend_block, info)
    # FlexAttention may use different reductions; allow slightly larger tolerance.
    assert torch.allclose(out_sdpa, out_flex, atol=2e-4, rtol=1e-4)


@_flex_skip
def test_flex_backend_matches_sdpa_chunked_boundary():
    torch.manual_seed(9)
    f, h, w = 3, 4, 5
    L = f * h * w
    device = torch.device("cuda")
    B, n, d = 1, 2, 4
    q = torch.randn(B, n, L, d, dtype=torch.float32, device=device)
    k = torch.randn(B, n, L, d, dtype=torch.float32, device=device)
    v = torch.randn(B, n, L, d, dtype=torch.float32, device=device)
    m_sdpa = _make_bsa(num_heads=n, block=(2, 2, 2), sparse_ratio=0.5, backend="sdpa_chunked").to(device)
    m_flex = _make_bsa(num_heads=n, block=(2, 2, 2), sparse_ratio=0.5, backend="flex").to(device)
    info = m_sdpa._compute_block_info((f, h, w), device=device)
    Q_b = m_sdpa._pad_and_permute(q, (f, h, w), info)
    K_b = m_sdpa._pad_and_permute(k, (f, h, w), info)
    V_b = m_sdpa._pad_and_permute(v, (f, h, w), info)
    Q_mean = m_sdpa._block_mean(Q_b, info)
    K_mean = m_sdpa._block_mean(K_b, info)
    attend_block = m_sdpa._compute_attend_block(Q_mean, K_mean)
    out_sdpa_full = m_sdpa._sparse_attn_sdpa_chunked(Q_b, K_b, V_b, attend_block, info)
    out_flex_full = m_flex._sparse_attn_flex(Q_b, K_b, V_b, attend_block, info)
    # Only compare on real tokens
    out_sdpa_real = m_sdpa._inverse_permute_and_crop(out_sdpa_full, (f, h, w), info)
    out_flex_real = m_flex._inverse_permute_and_crop(out_flex_full, (f, h, w), info)
    assert torch.allclose(out_sdpa_real, out_flex_real, atol=2e-4, rtol=1e-4)
