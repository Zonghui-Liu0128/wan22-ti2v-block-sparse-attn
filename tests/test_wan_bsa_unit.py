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
