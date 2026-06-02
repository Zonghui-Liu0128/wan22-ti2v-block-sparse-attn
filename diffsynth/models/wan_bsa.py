from collections.abc import Sequence
from typing import Optional

from .wan_video_dit import AttentionModule, BlockSparseAttention


def parse_bsa_block_size(value, default: Optional[tuple[int, int, int]] = None) -> tuple[int, int, int]:
    if value is None:
        if default is None:
            raise ValueError("BSA block size is required.")
        return tuple(default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"BSA block size must have 3 comma-separated ints, got {value!r}.")
        block_size = tuple(int(part) for part in parts)
    elif isinstance(value, Sequence):
        if len(value) != 3:
            raise ValueError(f"BSA block size must have length 3, got {value!r}.")
        block_size = tuple(int(part) for part in value)
    else:
        raise TypeError(f"Unsupported BSA block size type: {type(value)!r}.")
    if any(part <= 0 for part in block_size):
        raise ValueError(f"BSA block dimensions must be positive, got {block_size!r}.")
    return block_size


def iter_wan_self_attention(model):
    for block in getattr(model, "blocks", []):
        if hasattr(block, "self_attn"):
            yield block.self_attn


def iter_wan_bsa_modules(model):
    for self_attn in iter_wan_self_attention(model):
        attn = getattr(self_attn, "attn", None)
        if isinstance(attn, BlockSparseAttention):
            yield attn


def validate_bsa_sparse_ratio(sparse_ratio: float) -> float:
    sparse_ratio = float(sparse_ratio)
    if not 0.0 <= sparse_ratio < 1.0:
        raise ValueError(f"BSA sparse_ratio must lie in [0, 1), got {sparse_ratio}.")
    return sparse_ratio


def configure_wan_bsa(
    model,
    enable: bool,
    block_size=(2, 4, 4),
    sparse_ratio: float = 0.5,
    backend: str = "flex",
    chunk_size: int = 64,
) -> int:
    replaced = 0
    if model is None:
        return replaced

    block_size = parse_bsa_block_size(block_size)
    sparse_ratio = validate_bsa_sparse_ratio(sparse_ratio)
    for self_attn in iter_wan_self_attention(model):
        if enable:
            self_attn.attn = BlockSparseAttention(
                num_heads=self_attn.num_heads,
                block_size=block_size,
                sparse_ratio=sparse_ratio,
                backend=backend,
                chunk_size=chunk_size,
            )
            self_attn.bsa_enable = True
        else:
            self_attn.attn = AttentionModule(self_attn.num_heads)
            self_attn.bsa_enable = False
        replaced += 1
    return replaced


def set_wan_bsa_sparse_ratio(model, sparse_ratio: float) -> int:
    sparse_ratio = validate_bsa_sparse_ratio(sparse_ratio)
    updated = 0
    for attn in iter_wan_bsa_modules(model):
        attn.sparse_ratio = sparse_ratio
        updated += 1
    return updated


def get_wan_bsa_sparse_ratio(model) -> Optional[float]:
    for attn in iter_wan_bsa_modules(model):
        return attn.sparse_ratio
    return None
