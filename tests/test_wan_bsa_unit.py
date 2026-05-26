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


def test_sdpa_chunked_backend_caps_chunk_size_for_real_patch_grid(monkeypatch):
    """The SDPA fallback must shrink query-block chunks under a tight memory cap.

    This uses the H100 repro geometry: patchified (21,16,15), block=(3,2,3),
    sparse_ratio=0.8 -> 280 BSA blocks and 56 kept blocks per row.
    """
    import diffsynth.models.wan_video_dit as wan_dit

    monkeypatch.setenv("DIFFSYNTH_BSA_SDPA_CHUNK_BYTES", str(1024 * 1024))
    torch.manual_seed(22)
    f, h, w = 21, 16, 15
    block = (3, 2, 3)
    sparse_ratio = 0.8
    B, n, d = 1, 2, 4
    q = torch.randn(B, n, f * h * w, d, dtype=torch.float32)
    k = torch.randn(B, n, f * h * w, d, dtype=torch.float32)
    v = torch.randn(B, n, f * h * w, d, dtype=torch.float32)
    m = _make_bsa(num_heads=n, block=block, sparse_ratio=sparse_ratio, backend="sdpa_chunked")
    m.chunk_size = 64

    info = m._compute_block_info((f, h, w), device=q.device)
    Q_b = m._pad_and_permute(q, (f, h, w), info)
    K_b = m._pad_and_permute(k, (f, h, w), info)
    V_b = m._pad_and_permute(v, (f, h, w), info)
    Q_mean = m._block_mean(Q_b, info)
    K_mean = m._block_mean(K_b, info)
    attend_block = m._compute_attend_block(Q_mean, K_mean)

    call_batch_sizes = []

    def fake_sdpa(q_flat, k_flat, v_flat, attn_mask=None):
        call_batch_sizes.append(q_flat.shape[0])
        assert q_flat.shape[0] == 1
        assert k_flat.shape[-2] == 56 * 18
        return torch.zeros_like(q_flat)

    monkeypatch.setattr(wan_dit.F, "scaled_dot_product_attention", fake_sdpa)
    out = m._sparse_attn_sdpa_chunked(Q_b, K_b, V_b, attend_block, info)

    assert out.shape == Q_b.shape
    assert max(call_batch_sizes) == 1
    assert len(call_batch_sizes) == 280


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


def test_flex_backend_builds_block_mask_without_create_block_mask(monkeypatch):
    """The flex path already has block-level sparsity in attend_block.

    Rebuilding it through token-level create_block_mask calls mask_mod under vmap
    and can materialize huge dynamic-index tensors. The flex path should convert
    attend_block directly to a BlockMask.
    """
    from torch.nn.attention.flex_attention import BlockMask
    import diffsynth.models.wan_video_dit as wan_dit

    torch.manual_seed(21)
    f, h, w = 21, 16, 15
    block = (3, 2, 3)
    sparse_ratio = 0.8
    device = torch.device("cpu")
    B, n, d = 1, 2, 4
    q = torch.randn(B, n, f * h * w, d, dtype=torch.float32, device=device)
    k = torch.randn(B, n, f * h * w, d, dtype=torch.float32, device=device)
    v = torch.randn(B, n, f * h * w, d, dtype=torch.float32, device=device)
    m = _make_bsa(num_heads=n, block=block, sparse_ratio=sparse_ratio, backend="flex")

    info = m._compute_block_info((f, h, w), device=device)
    Q_b = m._pad_and_permute(q, (f, h, w), info)
    K_b = m._pad_and_permute(k, (f, h, w), info)
    V_b = m._pad_and_permute(v, (f, h, w), info)
    Q_mean = m._block_mean(Q_b, info)
    K_mean = m._block_mean(K_b, info)
    attend_block = m._compute_attend_block(Q_mean, K_mean)

    assert info["num_blocks"] == 280
    assert info["block_size_total"] == 18
    assert ((~attend_block).sum(dim=-1) == int(280 * sparse_ratio)).all()

    def fail_create_block_mask(*args, **kwargs):
        raise AssertionError("flex BSA must not call create_block_mask")

    def fake_flex_attention(Q, K, V, *, block_mask):
        assert isinstance(block_mask, BlockMask)
        assert block_mask.seq_lengths == (Q.shape[-2], K.shape[-2])
        assert block_mask.BLOCK_SIZE == (18, 18)
        assert block_mask.kv_num_blocks.shape == (B, n, 280)
        assert block_mask.kv_indices.shape == (B, n, 280, 280)
        assert torch.equal(block_mask.kv_num_blocks, torch.full((B, n, 280), 56, dtype=torch.int32))
        return torch.zeros_like(Q)

    monkeypatch.setattr(wan_dit, "_create_block_mask", fail_create_block_mask)
    monkeypatch.setattr(wan_dit, "_flex_attention", fake_flex_attention)

    out = m._sparse_attn_flex(Q_b, K_b, V_b, attend_block, info)
    assert out.shape == Q_b.shape


# --- end-to-end BlockSparseAttention.forward -------------------------------


def test_forward_end_to_end_clean_sdpa_chunked():
    torch.manual_seed(10)
    num_heads, d = 2, 4
    block = (2, 2, 2)
    f, h, w = 4, 4, 4
    L = f * h * w
    B = 1
    q = torch.randn(B, L, num_heads * d, dtype=torch.float32)
    k = torch.randn(B, L, num_heads * d, dtype=torch.float32)
    v = torch.randn(B, L, num_heads * d, dtype=torch.float32)
    m = _make_bsa(num_heads=num_heads, block=block, sparse_ratio=0.5, backend="sdpa_chunked")
    m._debug_record = True
    out = m(q, k, v, (f, h, w))
    assert out.shape == (B, L, num_heads * d)
    # Debug hooks captured
    assert m._dbg_attend_block is not None
    assert m._dbg_video_shape == (f, h, w)
    assert m._dbg_block_size == block

    # Independent reference: same lifted-mask SDPA on (B, n, L, d) layout
    from einops import rearrange
    q_nd = rearrange(q, "b l (n d) -> b n l d", n=num_heads)
    k_nd = rearrange(k, "b l (n d) -> b n l d", n=num_heads)
    v_nd = rearrange(v, "b l (n d) -> b n l d", n=num_heads)
    info = m._compute_block_info((f, h, w), device=q.device)
    Q_b = m._pad_and_permute(q_nd, (f, h, w), info)
    K_b = m._pad_and_permute(k_nd, (f, h, w), info)
    V_b = m._pad_and_permute(v_nd, (f, h, w), info)
    Q_mean = m._block_mean(Q_b, info)
    K_mean = m._block_mean(K_b, info)
    attend_block = m._compute_attend_block(Q_mean, K_mean)
    out_ref_blocked = _dense_masked_attention(Q_b, K_b, V_b, attend_block, info)
    out_ref_nd = m._inverse_permute_and_crop(out_ref_blocked, (f, h, w), info)
    out_ref = rearrange(out_ref_nd, "b n l d -> b l (n d)")
    assert torch.allclose(out, out_ref, atol=1e-5)


def test_forward_end_to_end_boundary_sdpa_chunked():
    torch.manual_seed(11)
    num_heads, d = 2, 4
    block = (2, 2, 2)
    f, h, w = 3, 4, 5
    L = f * h * w
    B = 1
    q = torch.randn(B, L, num_heads * d, dtype=torch.float32)
    k = torch.randn(B, L, num_heads * d, dtype=torch.float32)
    v = torch.randn(B, L, num_heads * d, dtype=torch.float32)
    m = _make_bsa(num_heads=num_heads, block=block, sparse_ratio=0.5, backend="sdpa_chunked")
    out = m(q, k, v, (f, h, w))
    assert out.shape == (B, L, num_heads * d)

    # Independent reference (same as above)
    from einops import rearrange
    q_nd = rearrange(q, "b l (n d) -> b n l d", n=num_heads)
    k_nd = rearrange(k, "b l (n d) -> b n l d", n=num_heads)
    v_nd = rearrange(v, "b l (n d) -> b n l d", n=num_heads)
    info = m._compute_block_info((f, h, w), device=q.device)
    Q_b = m._pad_and_permute(q_nd, (f, h, w), info)
    K_b = m._pad_and_permute(k_nd, (f, h, w), info)
    V_b = m._pad_and_permute(v_nd, (f, h, w), info)
    Q_mean = m._block_mean(Q_b, info)
    K_mean = m._block_mean(K_b, info)
    attend_block = m._compute_attend_block(Q_mean, K_mean)
    out_ref_blocked = _dense_masked_attention(Q_b, K_b, V_b, attend_block, info)
    out_ref_nd = m._inverse_permute_and_crop(out_ref_blocked, (f, h, w), info)
    out_ref = rearrange(out_ref_nd, "b n l d -> b l (n d)")
    assert torch.allclose(out, out_ref, atol=1e-5)


# --- SelfAttention integration ---------------------------------------------

from diffsynth.models.wan_video_dit import SelfAttention, AttentionModule


def test_self_attention_default_unchanged_uses_attention_module():
    sa = SelfAttention(dim=8, num_heads=2)
    assert isinstance(sa.attn, AttentionModule)


def test_self_attention_bsa_uses_block_sparse_module():
    sa = SelfAttention(
        dim=8, num_heads=2,
        bsa_enable=True, bsa_block_size=(2, 2, 2),
        bsa_sparse_ratio=0.5, bsa_backend="sdpa_chunked",
    )
    assert isinstance(sa.attn, BlockSparseAttention)
    assert sa.attn.block_size == (2, 2, 2)
    assert sa.attn.sparse_ratio == 0.5
    assert sa.attn.backend == "sdpa_chunked"


def test_self_attention_forward_bsa_disabled_matches_pre_bsa_path():
    """With bsa_enable=False, forward must take the AttentionModule path; video_shape ignored."""
    torch.manual_seed(12)
    dim = 16
    num_heads = 2
    L = 8
    sa = SelfAttention(dim=dim, num_heads=num_heads)
    sa.eval()
    x = torch.randn(1, L, dim)
    # Build a dummy freqs tensor matching the precompute_freqs_cis layout:
    # rope_apply consumes complex64 freqs of shape (L, 1, d_head/2).
    head_dim = dim // num_heads
    freqs = torch.polar(
        torch.ones(L, 1, head_dim // 2),
        torch.zeros(L, 1, head_dim // 2),
    )
    out_no_vs = sa(x, freqs)
    out_with_vs = sa(x, freqs, video_shape=(2, 2, 2))
    # Passing video_shape on a disabled BSA must not change output.
    assert torch.equal(out_no_vs, out_with_vs)


def test_self_attention_forward_bsa_enabled_runs():
    """With bsa_enable=True, forward must accept video_shape and produce a tensor of the right shape."""
    torch.manual_seed(13)
    dim = 16
    num_heads = 2
    f, h, w = 4, 4, 4
    L = f * h * w
    sa = SelfAttention(
        dim=dim, num_heads=num_heads,
        bsa_enable=True, bsa_block_size=(2, 2, 2),
        bsa_sparse_ratio=0.5, bsa_backend="sdpa_chunked",
    )
    sa.eval()
    x = torch.randn(1, L, dim)
    head_dim = dim // num_heads
    freqs = torch.polar(
        torch.ones(L, 1, head_dim // 2),
        torch.zeros(L, 1, head_dim // 2),
    )
    out = sa(x, freqs, video_shape=(f, h, w))
    assert out.shape == x.shape


# --- DiTBlock / WanModel plumbing ------------------------------------------

from diffsynth.models.wan_video_dit import DiTBlock, WanModel


def test_dit_block_default_unchanged():
    blk = DiTBlock(has_image_input=False, dim=16, num_heads=2, ffn_dim=32)
    assert isinstance(blk.self_attn.attn, AttentionModule)
    assert blk.self_attn.bsa_enable is False


def test_dit_block_bsa_propagated():
    blk = DiTBlock(
        has_image_input=False, dim=16, num_heads=2, ffn_dim=32,
        bsa_enable=True, bsa_block_size=(2, 2, 2),
        bsa_sparse_ratio=0.25, bsa_backend="sdpa_chunked",
    )
    assert isinstance(blk.self_attn.attn, BlockSparseAttention)
    assert blk.self_attn.attn.sparse_ratio == 0.25


def test_dit_block_forward_accepts_video_shape_kwarg():
    torch.manual_seed(14)
    f, h, w = 4, 4, 4
    L = f * h * w
    dim = 16
    blk = DiTBlock(
        has_image_input=False, dim=dim, num_heads=2, ffn_dim=32,
        bsa_enable=True, bsa_block_size=(2, 2, 2),
        bsa_sparse_ratio=0.5, bsa_backend="sdpa_chunked",
    )
    blk.eval()
    x = torch.randn(1, L, dim)
    context = torch.randn(1, 4, dim)
    t_mod = torch.zeros(1, 6, dim)
    head_dim = dim // 2
    freqs = torch.polar(torch.ones(L, 1, head_dim // 2), torch.zeros(L, 1, head_dim // 2))
    out = blk(x, context, t_mod, freqs, video_shape=(f, h, w))
    assert out.shape == x.shape


def test_wan_model_init_accepts_bsa_kwargs_and_propagates():
    """Smoke-test: WanModel constructs with bsa_enable=True and every block has a BSA attn."""
    model = WanModel(
        dim=16, in_dim=4, ffn_dim=32, out_dim=4,
        text_dim=8, freq_dim=8, eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2, num_layers=2,
        has_image_input=False,
        bsa_enable=True, bsa_block_size=(2, 2, 2),
        bsa_sparse_ratio=0.5, bsa_backend="sdpa_chunked",
    )
    for b in model.blocks:
        assert isinstance(b.self_attn.attn, BlockSparseAttention)


def test_wan_model_init_default_unchanged():
    """Without bsa_* kwargs, blocks must keep the standard AttentionModule."""
    model = WanModel(
        dim=16, in_dim=4, ffn_dim=32, out_dim=4,
        text_dim=8, freq_dim=8, eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2, num_layers=2,
        has_image_input=False,
    )
    for b in model.blocks:
        assert isinstance(b.self_attn.attn, AttentionModule)
