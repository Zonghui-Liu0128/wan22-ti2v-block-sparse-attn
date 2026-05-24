import json
import math
import os
from typing import Dict, Optional, Tuple

import torch


class BlockSparse3DMeta:
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        block_size: Tuple[int, int, int],
        padded_shape: Tuple[int, int, int],
        block_coords: torch.Tensor,
        valid_token_mask: torch.Tensor,
        block_token_mask: torch.Tensor,
    ):
        self.grid_shape = grid_shape
        self.block_size = block_size
        self.padded_shape = padded_shape
        self.block_coords = block_coords
        self.valid_token_mask = valid_token_mask
        self.block_token_mask = block_token_mask
        self.block_valid = block_token_mask.any(dim=-1)
        self.num_blocks = int(block_token_mask.shape[0])
        self.num_valid_blocks = int(self.block_valid.sum().item())
        self.num_padded_tokens = int((~valid_token_mask).sum().item())
        self.num_valid_tokens = int(valid_token_mask.sum().item())
        self.num_blocks_t = padded_shape[0] // block_size[0]
        self.num_blocks_h = padded_shape[1] // block_size[1]
        self.num_blocks_w = padded_shape[2] // block_size[2]


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        raise ValueError(f"Grid dimensions must be positive, got {value}.")
    if multiple <= 0:
        raise ValueError(f"Block dimensions must be positive, got {multiple}.")
    return ((value + multiple - 1) // multiple) * multiple


def _normalize_shape(shape: Tuple[int, int, int], name: str) -> Tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"{name} must contain exactly three integers, got {shape}.")
    return tuple(int(dim) for dim in shape)


def _partition_token_mask(mask: torch.Tensor, block_size: Tuple[int, int, int]) -> torch.Tensor:
    bt, bh, bw = block_size
    fp, hp, wp = mask.shape
    nt, nh, nw = fp // bt, hp // bh, wp // bw
    return (
        mask.reshape(nt, bt, nh, bh, nw, bw)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(nt * nh * nw, bt * bh * bw)
    )


def build_block_sparse_3d_meta(
    grid_shape: Tuple[int, int, int],
    block_size: Tuple[int, int, int],
    device: Optional[torch.device] = None,
) -> BlockSparse3DMeta:
    grid_shape = _normalize_shape(grid_shape, "grid_shape")
    block_size = _normalize_shape(block_size, "block_size")
    f, h, w = grid_shape
    bt, bh, bw = block_size
    padded_shape = (
        _ceil_to_multiple(f, bt),
        _ceil_to_multiple(h, bh),
        _ceil_to_multiple(w, bw),
    )
    fp, hp, wp = padded_shape
    nt, nh, nw = fp // bt, hp // bh, wp // bw
    device = torch.device("cpu") if device is None else device

    valid_token_mask = torch.zeros(padded_shape, dtype=torch.bool, device=device)
    valid_token_mask[:f, :h, :w] = True

    coords = []
    for tb in range(nt):
        for hb in range(nh):
            for wb in range(nw):
                coords.append([
                    tb * bt,
                    (tb + 1) * bt,
                    hb * bh,
                    (hb + 1) * bh,
                    wb * bw,
                    (wb + 1) * bw,
                ])
    block_coords = torch.tensor(coords, dtype=torch.long, device=device)
    block_token_mask = _partition_token_mask(valid_token_mask, block_size)
    return BlockSparse3DMeta(
        grid_shape=grid_shape,
        block_size=block_size,
        padded_shape=padded_shape,
        block_coords=block_coords,
        valid_token_mask=valid_token_mask,
        block_token_mask=block_token_mask,
    )


def pad_to_block_size(
    x: torch.Tensor,
    meta: BlockSparse3DMeta,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if x.ndim != 6:
        raise ValueError(f"Expected x with shape (B, f, h, w, H, Dh), got {tuple(x.shape)}.")
    bsz, f, h, w, heads, head_dim = x.shape
    if (f, h, w) != meta.grid_shape:
        raise ValueError(f"Input grid {(f, h, w)} does not match metadata grid {meta.grid_shape}.")
    out = torch.full(
        (bsz, *meta.padded_shape, heads, head_dim),
        fill_value=pad_value,
        dtype=x.dtype,
        device=x.device,
    )
    out[:, :f, :h, :w] = x
    return out


def partition_3d_blocks(x: torch.Tensor, meta: BlockSparse3DMeta) -> torch.Tensor:
    if x.ndim != 6:
        raise ValueError(f"Expected x with shape (B, f', h', w', H, Dh), got {tuple(x.shape)}.")
    bsz, fp, hp, wp, heads, head_dim = x.shape
    if (fp, hp, wp) != meta.padded_shape:
        raise ValueError(f"Input padded grid {(fp, hp, wp)} does not match {meta.padded_shape}.")
    bt, bh, bw = meta.block_size
    nt, nh, nw = meta.num_blocks_t, meta.num_blocks_h, meta.num_blocks_w
    return (
        x.reshape(bsz, nt, bt, nh, bh, nw, bw, heads, head_dim)
        .permute(0, 1, 3, 5, 2, 4, 6, 7, 8)
        .reshape(bsz, meta.num_blocks, bt * bh * bw, heads, head_dim)
    )


def reverse_3d_blocks(blocks: torch.Tensor, meta: BlockSparse3DMeta) -> torch.Tensor:
    if blocks.ndim != 5:
        raise ValueError(f"Expected blocks with shape (B, N, b, H, Dh), got {tuple(blocks.shape)}.")
    bsz, num_blocks, block_tokens, heads, head_dim = blocks.shape
    if num_blocks != meta.num_blocks:
        raise ValueError(f"Expected {meta.num_blocks} blocks, got {num_blocks}.")
    bt, bh, bw = meta.block_size
    if block_tokens != bt * bh * bw:
        raise ValueError(f"Expected {bt * bh * bw} tokens per block, got {block_tokens}.")
    nt, nh, nw = meta.num_blocks_t, meta.num_blocks_h, meta.num_blocks_w
    return (
        blocks.reshape(bsz, nt, nh, nw, bt, bh, bw, heads, head_dim)
        .permute(0, 1, 4, 2, 5, 3, 6, 7, 8)
        .reshape(bsz, *meta.padded_shape, heads, head_dim)
    )


def masked_block_mean(x_blk: torch.Tensor, block_token_mask: torch.Tensor) -> torch.Tensor:
    if x_blk.ndim != 5:
        raise ValueError(f"Expected x_blk with shape (B, H, N, b, Dh), got {tuple(x_blk.shape)}.")
    if block_token_mask.shape != x_blk.shape[2:4]:
        raise ValueError(
            f"Mask shape {tuple(block_token_mask.shape)} does not match block/token dims {tuple(x_blk.shape[2:4])}."
        )
    weight = block_token_mask.to(dtype=x_blk.dtype, device=x_blk.device).view(1, 1, *block_token_mask.shape, 1)
    denom = weight.sum(dim=-2).clamp_min(1.0)
    mean = (x_blk * weight).sum(dim=-2) / denom
    block_valid = block_token_mask.any(dim=-1).to(device=x_blk.device).view(1, 1, -1, 1)
    return mean * block_valid.to(dtype=x_blk.dtype)


def select_topk_blocks(
    scores: torch.Tensor,
    block_valid: torch.Tensor,
    sparsity: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if scores.ndim != 4:
        raise ValueError(f"Expected scores with shape (B, H, Nq, Nkv), got {tuple(scores.shape)}.")
    if not 0.0 <= float(sparsity) < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}.")
    block_valid = block_valid.to(device=scores.device, dtype=torch.bool)
    if block_valid.ndim != 1 or block_valid.shape[0] != scores.shape[-1]:
        raise ValueError("block_valid must be a 1D bool tensor matching the KV block dimension.")
    num_valid = int(block_valid.sum().item())
    if num_valid <= 0:
        raise ValueError("At least one valid KV block is required.")
    k_keep = max(1, int(round(num_valid * (1.0 - float(sparsity)))))
    k_keep = min(k_keep, num_valid)

    masked_scores = scores.masked_fill(~block_valid.view(1, 1, 1, -1), -torch.inf)
    indices = torch.topk(masked_scores, k=k_keep, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    invalid_kv_selected = int((~block_valid[indices]).sum().item())
    stats = {
        "k_keep": k_keep,
        "num_valid_blocks": num_valid,
        "mask_density": float(mask.float().mean().item()),
        "invalid_kv_selected": invalid_kv_selected,
    }
    return indices, mask, stats


def _gather_blocks(blocks: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    bsz, heads, _, block_tokens, head_dim = blocks.shape
    _, _, q_blocks, keep_blocks = indices.shape
    expanded = blocks.unsqueeze(2).expand(bsz, heads, q_blocks, -1, block_tokens, head_dim)
    gather_idx = indices.unsqueeze(-1).unsqueeze(-1).expand(
        bsz, heads, q_blocks, keep_blocks, block_tokens, head_dim
    )
    return torch.gather(expanded, dim=3, index=gather_idx)


def block_sparse_attention_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    block_size: Tuple[int, int, int],
    num_heads: int,
    sparsity: float,
    q_chunk_blocks: int = 16,
    scale: Optional[float] = None,
    return_debug: bool = False,
):
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Q/K/V shapes must match, got {q.shape}, {k.shape}, {v.shape}.")
    if q.ndim != 3:
        raise ValueError(f"Expected Q/K/V with shape (B, S, D), got {tuple(q.shape)}.")
    if q_chunk_blocks <= 0:
        raise ValueError(f"q_chunk_blocks must be positive, got {q_chunk_blocks}.")

    bsz, seq_len, dim = q.shape
    f, h, w = _normalize_shape(grid_shape, "grid_shape")
    if seq_len != f * h * w:
        raise ValueError(f"Sequence length {seq_len} does not match grid {grid_shape}.")
    if dim % num_heads != 0:
        raise ValueError(f"Hidden dim {dim} must be divisible by num_heads {num_heads}.")
    head_dim = dim // num_heads
    scale = (1.0 / math.sqrt(head_dim)) if scale is None else scale
    meta = build_block_sparse_3d_meta(grid_shape, block_size, device=q.device)

    q_grid = q.reshape(bsz, f, h, w, num_heads, head_dim)
    k_grid = k.reshape(bsz, f, h, w, num_heads, head_dim)
    v_grid = v.reshape(bsz, f, h, w, num_heads, head_dim)
    q_blk = partition_3d_blocks(pad_to_block_size(q_grid, meta), meta).permute(0, 3, 1, 2, 4)
    k_blk = partition_3d_blocks(pad_to_block_size(k_grid, meta), meta).permute(0, 3, 1, 2, 4)
    v_blk = partition_3d_blocks(pad_to_block_size(v_grid, meta), meta).permute(0, 3, 1, 2, 4)

    with torch.no_grad():
        q_repr = masked_block_mean(q_blk, meta.block_token_mask)
        k_repr = masked_block_mean(k_blk, meta.block_token_mask)
        scores = torch.matmul(q_repr.float(), k_repr.float().transpose(-1, -2)) * float(scale)
        indices, block_mask, route_stats = select_topk_blocks(scores, meta.block_valid, sparsity)

    out_blk = torch.zeros_like(q_blk)
    token_mask = meta.block_token_mask.to(device=q.device)
    for q0 in range(0, meta.num_blocks, q_chunk_blocks):
        q1 = min(q0 + q_chunk_blocks, meta.num_blocks)
        q_c = q_blk[:, :, q0:q1]
        q_valid = token_mask[q0:q1].view(1, 1, q1 - q0, -1, 1)
        idx_c = indices[:, :, q0:q1]
        k_sel = _gather_blocks(k_blk, idx_c)
        v_sel = _gather_blocks(v_blk, idx_c)
        kv_valid = token_mask[idx_c].reshape(bsz, num_heads, q1 - q0, -1)

        k_tok = k_sel.reshape(bsz, num_heads, q1 - q0, -1, head_dim)
        v_tok = v_sel.reshape(bsz, num_heads, q1 - q0, -1, head_dim)
        logits = torch.matmul(q_c.float(), k_tok.float().transpose(-1, -2)) * float(scale)
        logits = logits.masked_fill(~kv_valid.view(bsz, num_heads, q1 - q0, 1, -1), -torch.inf)
        probs = torch.softmax(logits, dim=-1).to(dtype=v_tok.dtype)
        out_c = torch.matmul(probs, v_tok)
        out_blk[:, :, q0:q1] = out_c * q_valid.to(dtype=out_c.dtype, device=out_c.device)

    out_blocks = out_blk.permute(0, 2, 3, 1, 4).contiguous()
    out_grid = reverse_3d_blocks(out_blocks, meta)[:, :f, :h, :w]
    out = out_grid.reshape(bsz, seq_len, dim)

    if not return_debug:
        return out
    boundary_blocks = int((meta.block_token_mask.any(dim=-1) & (~meta.block_token_mask).any(dim=-1)).sum().item())
    debug = {
        "grid": list(meta.grid_shape),
        "padded_grid": list(meta.padded_shape),
        "block_size": list(meta.block_size),
        "num_blocks": meta.num_blocks,
        "num_valid_blocks": meta.num_valid_blocks,
        "num_valid_tokens": meta.num_valid_tokens,
        "num_padded_tokens": meta.num_padded_tokens,
        "boundary_blocks_with_padding": boundary_blocks,
        "sparsity": float(sparsity),
        "k_keep": int(route_stats["k_keep"]),
        "mask_density": float(route_stats["mask_density"]),
        "invalid_kv_selected": int(route_stats["invalid_kv_selected"]),
        "padded_token_logits_masked": True,
        "block_coords": meta.block_coords.detach().cpu(),
        "block_mask": block_mask.detach().cpu(),
        "selected_block_indices": indices.detach().cpu(),
    }
    return out, debug


def save_block_sparse_debug(
    debug: Dict[str, object],
    output_dir: str,
    prefix: str,
    head: int = 0,
    max_query_blocks: int = 16,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    serializable = {}
    for key, value in debug.items():
        if isinstance(value, torch.Tensor):
            continue
        serializable[key] = value
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    block_coords = debug["block_coords"]
    block_mask = debug["block_mask"]
    selected = debug["selected_block_indices"]
    torch.save(block_mask, os.path.join(output_dir, f"{prefix}_mask.pt"))
    torch.save(selected, os.path.join(output_dir, f"{prefix}_selected.pt"))

    with open(os.path.join(output_dir, f"{prefix}_block_layout.txt"), "w", encoding="utf-8") as f:
        for idx, (t0, t1, h0, h1, w0, w1) in enumerate(block_coords.tolist()):
            f.write(f"block {idx}: t[{t0}:{t1}), h[{h0}:{h1}), w[{w0}:{w1})\n")

    mask_2d = block_mask[0, min(head, block_mask.shape[1] - 1)]
    with open(os.path.join(output_dir, f"{prefix}_mask_head{head}.txt"), "w", encoding="utf-8") as f:
        for row in mask_2d.tolist():
            f.write("".join("1" if value else "." for value in row) + "\n")

    selected_2d = selected[0, min(head, selected.shape[1] - 1)]
    with open(os.path.join(output_dir, f"{prefix}_selected_head{head}.txt"), "w", encoding="utf-8") as f:
        for query_idx, kv_indices in enumerate(selected_2d[:max_query_blocks].tolist()):
            coords = [block_coords[kv_idx].tolist() for kv_idx in kv_indices]
            f.write(f"query_block {query_idx}: kv_blocks={kv_indices}, kv_coords={coords}\n")
