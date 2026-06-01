"""Wan2.2-TI2V-5B inference with optional Block Sparse Attention."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video
from diffsynth.models.wan_video_dit import BlockSparseAttention


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    p = argparse.ArgumentParser(description="Run Wan2.2-TI2V-5B with optional Block Sparse Attention.")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sparse-ratio", type=float, default=0.0,
                   help="Block-drop fraction (0 = dense baseline, 0.5 = drop half, etc.).")
    p.add_argument("--block-size", type=str, default="2,4,4",
                   help="Block size as 'bt,bh,bw' (default 2,4,4).")
    p.add_argument("--bsa-backend", choices=["flex", "sdpa_chunked"], default="flex")
    p.add_argument("--bsa-chunk-size", type=int, default=64,
                   help="chunk_size for sdpa_chunked backend.")
    p.add_argument("--dump-attention-png", default=None,
                   help="Save the first DiT block's final block-level BSA mask as a PNG.")
    return p.parse_args()


def parse_block_size(s):
    parts = [int(x) for x in s.split(",")]
    assert len(parts) == 3, f"block-size must have 3 ints, got {s!r}"
    return tuple(parts)


def enable_bsa(model, block_size, sparse_ratio, backend, chunk_size, record_attention=False):
    """Replace each DiTBlock.self_attn.attn with a BlockSparseAttention module."""
    replaced = 0
    recorder = None
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
        if record_attention and recorder is None:
            bsa.record_attend_block = True
            recorder = bsa
        sa.attn = bsa
        sa.bsa_enable = True
        replaced += 1
    return replaced, recorder


def save_attention_png(bsa, png_path, sparse_ratio):
    attend = getattr(bsa, "last_attend_block", None)
    if attend is None:
        print("[bsa] no attention mask was recorded")
        return

    keep = attend[0, 0].to(torch.bool)
    num_blocks = keep.shape[0]
    image = torch.zeros(num_blocks, num_blocks, 3, dtype=torch.uint8)
    image[keep] = torch.tensor([255, 215, 0], dtype=torch.uint8)
    image[~keep] = torch.tensor([128, 0, 128], dtype=torch.uint8)

    img = Image.fromarray(image.numpy(), mode="RGB")
    scale = max(1, min(8, 1024 // max(1, num_blocks)))
    if scale > 1:
        resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        img = img.resize((num_blocks * scale, num_blocks * scale), resample=resample)

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)

    meta_path = png_path.with_suffix(".txt")
    meta_path.write_text(
        "\n".join([
            f"video_shape={bsa.last_video_shape}",
            f"block_size={bsa.last_block_size}",
            f"sparse_ratio={sparse_ratio}",
            f"num_blocks={num_blocks}",
            f"drop_blocks_per_row={int(num_blocks * sparse_ratio)}",
            "colors=yellow:keep,purple:drop",
        ]) + "\n"
    )
    print(f"[bsa] saved attention mask to {png_path}")
    print(f"[bsa] saved attention metadata to {meta_path}")


def main():
    args = parse_args()
    image_path = Path(args.image)
    prompt = args.prompt or image_path.stem.replace("_", " ")
    block_size = parse_block_size(args.block_size)
    block_tokens = block_size[0] * block_size[1] * block_size[2]
    if args.sparse_ratio > 0.0 and args.bsa_backend == "flex" and (block_tokens < 32 or block_tokens % 32 != 0):
        raise SystemExit(
            "Torch FlexAttention on CUDA requires bt*bh*bw to be a multiple of 32; "
            f"got --block-size {args.block_size} (product={block_tokens}). "
            "Use --block-size 2,4,4 or --bsa-backend sdpa_chunked."
        )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files=False,
    )

    attention_recorder = None
    if args.sparse_ratio > 0.0:
        n, attention_recorder = enable_bsa(
            pipe.dit, block_size, args.sparse_ratio,
            backend=args.bsa_backend, chunk_size=args.bsa_chunk_size,
            record_attention=args.dump_attention_png is not None,
        )
        print(f"[bsa] enabled on {n} blocks  block_size={block_size}  "
              f"sparse_ratio={args.sparse_ratio}  backend={args.bsa_backend}")
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
    if args.dump_attention_png and attention_recorder is not None:
        save_attention_png(attention_recorder, args.dump_attention_png, args.sparse_ratio)


if __name__ == "__main__":
    main()
