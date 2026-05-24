import math
import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "diffsynth/core/attention/block_sparse_3d.py"
if not _MODULE_PATH.exists():
    pytest.fail(f"Missing block sparse attention module: {_MODULE_PATH}")
_SPEC = importlib.util.spec_from_file_location("block_sparse_3d", _MODULE_PATH)
block_sparse_3d = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(block_sparse_3d)

block_sparse_attention_3d = block_sparse_3d.block_sparse_attention_3d
build_block_sparse_3d_meta = block_sparse_3d.build_block_sparse_3d_meta
masked_block_mean = block_sparse_3d.masked_block_mean
pad_to_block_size = block_sparse_3d.pad_to_block_size
partition_3d_blocks = block_sparse_3d.partition_3d_blocks
reverse_3d_blocks = block_sparse_3d.reverse_3d_blocks
select_topk_blocks = block_sparse_3d.select_topk_blocks


def test_partition_roundtrip_preserves_wan_rope_order_for_non_divisible_grid():
    batch, frames, height, width, heads, head_dim = 1, 3, 5, 7, 2, 4
    meta = build_block_sparse_3d_meta(
        grid_shape=(frames, height, width),
        block_size=(2, 3, 4),
        device=torch.device("cpu"),
    )
    x = torch.arange(
        batch * frames * height * width * heads * head_dim,
        dtype=torch.float32,
    ).view(batch, frames, height, width, heads, head_dim)

    padded = pad_to_block_size(x, meta)
    blocks = partition_3d_blocks(padded, meta)
    recovered = reverse_3d_blocks(blocks, meta)[:, :frames, :height, :width]

    assert meta.padded_shape == (4, 6, 8)
    assert meta.num_blocks == 8
    assert torch.equal(recovered, x)

    layout_meta = build_block_sparse_3d_meta(
        grid_shape=(2, 4, 4),
        block_size=(1, 2, 2),
        device=torch.device("cpu"),
    )
    assert layout_meta.block_coords.tolist() == [
        [0, 1, 0, 2, 0, 2],
        [0, 1, 0, 2, 2, 4],
        [0, 1, 2, 4, 0, 2],
        [0, 1, 2, 4, 2, 4],
        [1, 2, 0, 2, 0, 2],
        [1, 2, 0, 2, 2, 4],
        [1, 2, 2, 4, 0, 2],
        [1, 2, 2, 4, 2, 4],
    ]


def test_masked_block_mean_ignores_padding_tokens_inside_boundary_blocks():
    meta = build_block_sparse_3d_meta(
        grid_shape=(3, 5, 7),
        block_size=(2, 3, 4),
        device=torch.device("cpu"),
    )
    x = torch.ones(1, 3, 5, 7, 1, 1)
    padded = pad_to_block_size(x, meta, pad_value=999.0)
    blocks = partition_3d_blocks(padded, meta).permute(0, 3, 1, 2, 4)
    # partition_3d_blocks returns (B, N, b, H, Dh); masked_block_mean expects
    # (B, H, N, b, Dh).
    block_mean = masked_block_mean(blocks, meta.block_token_mask)

    assert meta.valid_token_mask.sum().item() == 3 * 5 * 7
    assert (~meta.valid_token_mask).sum().item() == 4 * 6 * 8 - 3 * 5 * 7
    assert torch.allclose(block_mean, torch.ones_like(block_mean))


def test_select_topk_blocks_never_selects_invalid_kv_blocks():
    scores = torch.zeros(1, 1, 3, 4)
    scores[..., 3] = 1000.0
    block_valid = torch.tensor([True, True, True, False])

    indices, mask, stats = select_topk_blocks(
        scores,
        block_valid=block_valid,
        sparsity=0.5,
    )

    assert stats["k_keep"] == 2
    assert stats["invalid_kv_selected"] == 0
    assert not mask[..., 3].any()
    assert not (indices == 3).any()


def test_block_sparse_attention_matches_dense_attention_when_sparsity_is_zero():
    torch.manual_seed(0)
    batch, frames, height, width, heads, head_dim = 1, 2, 4, 4, 2, 8
    dim = heads * head_dim
    seq_len = frames * height * width
    q = torch.randn(batch, seq_len, dim, dtype=torch.float32, requires_grad=True)
    k = torch.randn(batch, seq_len, dim, dtype=torch.float32, requires_grad=True)
    v = torch.randn(batch, seq_len, dim, dtype=torch.float32, requires_grad=True)

    sparse_out, debug = block_sparse_attention_3d(
        q,
        k,
        v,
        grid_shape=(frames, height, width),
        block_size=(1, 2, 2),
        num_heads=heads,
        sparsity=0.0,
        q_chunk_blocks=3,
        return_debug=True,
    )
    dense_q = q.view(batch, seq_len, heads, head_dim).transpose(1, 2)
    dense_k = k.view(batch, seq_len, heads, head_dim).transpose(1, 2)
    dense_v = v.view(batch, seq_len, heads, head_dim).transpose(1, 2)
    dense_out = torch.nn.functional.scaled_dot_product_attention(
        dense_q,
        dense_k,
        dense_v,
        scale=1.0 / math.sqrt(head_dim),
    ).transpose(1, 2).reshape(batch, seq_len, dim)

    assert debug["k_keep"] == debug["num_valid_blocks"]
    assert debug["padded_token_logits_masked"]
    assert torch.allclose(sparse_out, dense_out, atol=1e-5, rtol=1e-4)

    sparse_out.sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
