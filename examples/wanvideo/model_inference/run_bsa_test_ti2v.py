"""Wan2.2-TI2V-5B inference with Block Sparse Attention.

After loading the pre-trained pipeline, this script swaps each DiTBlock's
self_attn.attn for a BlockSparseAttention module configured with the requested
block size, sparse ratio, and backend.  Setting sparse_ratio=0 keeps the
default dense AttentionModule (useful as a baseline).

Examples:
    # dense baseline
    python run_bsa_test_ti2v.py --image foo.jpg --output base.mp4 --sparse-ratio 0.0

    # 50% block-sparse
    python run_bsa_test_ti2v.py --image foo.jpg --output bsa50.mp4 --sparse-ratio 0.5

The first DiTBlock can optionally dump its block-level attend pattern as a PNG
via --dump-attention-png so you can verify on a real generation step.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video
from diffsynth.models.wan_video_dit import (
    AttentionModule,
    BlockSparseAttention,
)


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    p = argparse.ArgumentParser(description="Run Wan2.2-TI2V-5B with optional Block Sparse Attention.")
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sparse-ratio", type=float, default=0.0,
                   help="Block-drop fraction (0 = dense baseline, 0.5 = drop half, etc.).")
    p.add_argument("--block-size", type=str, default="2,4,4",
                   help="Block size as 'bt,bh,bw' (default 2,4,4).")
    p.add_argument("--bsa-backend", choices=["flex", "sdpa_chunked"], default="flex")
    p.add_argument("--bsa-chunk-size", type=int, default=64,
                   help="chunk_size for sdpa_chunked backend.")
    p.add_argument("--dump-attention-png", type=str, default=None,
                   help="If set, run a single dummy forward at the real shape and save the "
                        "first block's attend_block PNG to this path before generation.")
    return p.parse_args()


def parse_block_size(s):
    parts = [int(x) for x in s.split(",")]
    assert len(parts) == 3, f"block-size must have 3 ints, got {s!r}"
    return tuple(parts)


def enable_bsa(model, block_size, sparse_ratio, backend, chunk_size):
    """Replace each DiTBlock.self_attn.attn with a BlockSparseAttention module."""
    replaced = 0
    for blk in model.blocks:
        sa = blk.self_attn
        ref = sa.q.weight
        bsa = BlockSparseAttention(
            num_heads=sa.num_heads,
            block_size=block_size,
            sparse_ratio=sparse_ratio,
            backend=backend,
            chunk_size=chunk_size,
        ).to(device=ref.device, dtype=ref.dtype)
        sa.attn = bsa
        sa.bsa_enable = True
        replaced += 1
    return replaced


def maybe_dump_attention(model, png_path):
    """Trigger debug capture on block 0's BSA module after the first forward
    by toggling _debug_record and running a single dummy forward at the real
    inference shape is too expensive; instead we just turn _debug_record on
    so the next real forward stores the data, and we render after generation.

    For simplicity, this helper sets _debug_record=True on every block; the
    generation loop will fill the buffers, and we render block 0's snapshot
    afterwards.
    """
    for blk in model.blocks:
        if isinstance(blk.self_attn.attn, BlockSparseAttention):
            blk.self_attn.attn._debug_record = True


def render_block_level_png(model, png_path, block_size, sparse_ratio):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    bsa = model.blocks[0].self_attn.attn
    ab = bsa._dbg_attend_block
    if ab is None:
        print("[bsa-dump] no debug snapshot found (no real BSA forward happened)")
        return
    f, h, w = bsa._dbg_video_shape
    num_blocks = ab.shape[-1]
    K_drop = int(num_blocks * sparse_ratio)
    arr = ab[0, 0].cpu().to(torch.int8).numpy()
    cmap = ListedColormap(["#800080", "#FFD700"])
    fig_size = max(6, min(18, num_blocks * 0.02 + 6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("key block index")
    ax.set_ylabel("query block index")
    ax.set_title(
        f"Wan2.2 SelfAttention BSA  block-level attend  f={f},h={h},w={w}  "
        f"block={block_size}  ratio={sparse_ratio}  num_blocks={num_blocks}  "
        f"K_drop={K_drop}  head=0 batch=0",
        fontsize=9, pad=12,
    )
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[bsa-dump] saved {png_path}")


def main():
    args = parse_args()
    image_path = Path(args.image)
    prompt = image_path.stem

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )

    block_size = parse_block_size(args.block_size)
    if args.sparse_ratio > 0.0:
        n = enable_bsa(
            pipe.dit, block_size, args.sparse_ratio,
            backend=args.bsa_backend, chunk_size=args.bsa_chunk_size,
        )
        print(f"[bsa] enabled on {n} blocks  block_size={block_size}  "
              f"sparse_ratio={args.sparse_ratio}  backend={args.bsa_backend}")
        if args.dump_attention_png:
            maybe_dump_attention(pipe.dit, args.dump_attention_png)
    else:
        print("[bsa] sparse_ratio=0 -> dense baseline (AttentionModule untouched)")

    input_image = Image.open(image_path).convert("RGB").resize((args.width, args.height))
    video = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        seed=args.seed,
        tiled=True,
        height=args.height,
        width=args.width,
        input_image=input_image,
        num_frames=args.frames,
        num_inference_steps=args.steps,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_path), fps=15, quality=5)
    print(f"Saved BSA TI2V output to {output_path}")

    if args.sparse_ratio > 0.0 and args.dump_attention_png:
        render_block_level_png(pipe.dit, args.dump_attention_png, block_size, args.sparse_ratio)


if __name__ == "__main__":
    main()
