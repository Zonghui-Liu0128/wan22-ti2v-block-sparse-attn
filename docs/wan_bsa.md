# Wan2.2-5B Block Sparse Attention

This branch keeps upstream full attention unchanged by default and adds an
opt-in `BlockSparseAttention` module for Wan self-attention.

## Implementation

- `BlockSparseAttention` pads the patchified latent grid `(f,h,w)` to a multiple
  of `bsa_block_size=(bt,bh,bw)`, groups each 3D block contiguously, computes
  Q/K block means, drops `floor(num_blocks * sparse_ratio)` least-similar key
  blocks per query block, attends over the kept block pairs, then crops padding.
- Boundary blocks are normalized by their real token count, so non-divisible
  latent sizes are handled without biasing the block similarity score.
- `sdpa_chunked` gathers kept K/V blocks and runs PyTorch SDPA in memory-capped
  query-block chunks.
- `flex` builds `torch.nn.attention.flex_attention.BlockMask` directly from
  block-level keep indices and runs compiled FlexAttention on CUDA.

`bsa_enable=False` remains the default. `sparse_ratio=0.0` is the dense baseline.

## Flex Limitation

On PyTorch 2.8 CUDA, sparse `BlockMask` must use compiled FlexAttention; eager
FlexAttention does not execute the sparse block mask correctly. The compiled
kernel also requires `bt*bh*bw` to be a multiple of 32. Use `block_size=2,4,4`
for `flex`; use `sdpa_chunked` for products such as `3*2*3=18`.

## Inference Example

```bash
python examples/wanvideo/model_inference/run_bsa_test_ti2v.py \
  --image input.jpg \
  --prompt "A cinematic robot walking through a neon city at night" \
  --output outputs/bsa_r090.mp4 \
  --height 832 --width 480 --frames 81 --steps 12 --seed 42 \
  --sparse-ratio 0.9 \
  --block-size 2,4,4 \
  --bsa-backend sdpa_chunked \
  --dump-attention-png outputs/bsa_r090_mask.png
```

The mask PNG uses yellow for kept key blocks and purple for dropped key blocks.
The sidecar `.txt` records `video_shape`, `block_size`, `sparse_ratio`,
`num_blocks`, and dropped blocks per query row.

## Validation

AutoDL A800-80GB, PyTorch `2.8.0+cu128`, Wan2.2-TI2V-5B, `832x480`, 81 frames,
`block_size=2,4,4`:

- `sdpa_chunked` direct inference succeeded for sparse ratios
  `0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99`.
- Visual inspection puts the practical maximum at `sparse_ratio=0.90`; `0.95`
  starts to drift, and `0.97+` becomes heavily distorted.
- `flex` direct inference succeeded at `sparse_ratio=0.80`.
- Real 81-frame mask metadata: `video_shape=(21,26,15)`, `num_blocks=308`.

Tests:

```bash
python -m pytest tests/test_wan_bsa_unit.py -v
```
