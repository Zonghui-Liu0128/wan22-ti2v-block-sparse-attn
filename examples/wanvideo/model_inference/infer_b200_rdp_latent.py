import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video
from diffsynth.models.wan_video_dit import BlockSparseAttention


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_block_size(s):
    parts = [int(x) for x in s.split(",")]
    assert len(parts) == 3, f"block-size must have 3 ints, got {s!r}"
    return tuple(parts)


def enable_bsa(model, block_size, sparse_ratio, backend, chunk_size, record_attention=False):
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


def build_diffusion_latent_save_obj(
    latent,
    image_path,
    image_latent_path,
    prompt,
    negative_prompt,
    lora_checkpoint,
    lora_alpha,
    height,
    width,
    num_frames,
    seed,
    steps,
):
    return {
        "latent": latent.detach().cpu(),
        "latent_role": "wan_diffusion_latent_after_denoise_unpatchified",
        "latent_layout": "B C F H W",
        "latent_shape": tuple(latent.shape),
        "first_frame_latent_layout": "B C H W, unsqueeze dim=2 before pipeline use",
        "image_path": None if image_path is None else str(image_path),
        "image_latent_path": None if image_latent_path is None else str(image_latent_path),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "lora_checkpoint": lora_checkpoint,
        "lora_alpha": lora_alpha,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "seed": seed,
        "steps": steps,
        "dtype": str(latent.dtype),
    }


def load_prompt_from_metadata(metadata_path, image_path):
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    image_name = os.path.basename(image_path)
    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "image" not in reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError(
                "metadata.csv must contain 'image' and 'prompt' columns; "
                f"got: {reader.fieldnames}"
            )
        for row in reader:
            if os.path.basename(row["image"].strip()) == image_name:
                return row["prompt"]

    raise KeyError(f"No prompt matched image {image_name!r} in {metadata_path}")


def infer_height_width(input_image, input_image_latent_path=None, height=None, width=None, max_area=480 * 832):
    if height is not None and width is not None:
        return int(height), int(width)

    if input_image is None:
        if input_image_latent_path is None:
            raise ValueError("input_image is required unless height/width or input_image_latent_path metadata is provided")
        saved = torch.load(input_image_latent_path, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or "height" not in saved or "width" not in saved:
            raise ValueError("input_image_latent_path must contain height and width when input_image is None")
        return int(saved["height"]), int(saved["width"])

    aspect_ratio = input_image.height / input_image.width
    mod_value = 16
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return int(height), int(width)


def run_pipe_and_save_outputs(
    pipe,
    input_image,
    input_image_path,
    input_image_latent_path,
    prompt,
    negative_prompt,
    save_dir,
    save_diffusion_latent_path,
    skip_video_decode,
    height,
    width,
    num_frames,
    seed,
    num_inference_steps,
    cfg_scale,
    lora_path,
):
    if skip_video_decode and save_diffusion_latent_path is None:
        raise ValueError("skip_video_decode=True requires save_diffusion_latent_path")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        input_image_latent=input_image_latent_path,
        height=height,
        width=width,
        seed=seed,
        tiled=True,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        output_type="latent" if skip_video_decode else "quantized",
        return_latents=save_diffusion_latent_path is not None and not skip_video_decode,
    )

    if skip_video_decode:
        video = None
        diffusion_latent = result
    elif save_diffusion_latent_path is not None:
        video, diffusion_latent = result
    else:
        video = result
        diffusion_latent = None

    if save_diffusion_latent_path is not None:
        save_path = Path(save_diffusion_latent_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            build_diffusion_latent_save_obj(
                diffusion_latent,
                image_path=input_image_path,
                image_latent_path=input_image_latent_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                lora_checkpoint=lora_path,
                lora_alpha=1.0,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                steps=num_inference_steps,
            ),
            save_path,
        )
        print(f"[latent] saved denoised diffusion latent to {save_path}")

    if video is not None and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        video_path = Path(save_dir) / f"{Path(input_image_path).stem}.mp4"
        save_video(video, str(video_path), fps=15, quality=5)
        print(f"Video saved as {video_path}")

    return diffusion_latent if skip_video_decode else video


def inference_video_b200(
    input_image,
    input_image_path,
    prompt,
    save_dir=None,
    model_paths="/data_3d/w00950754/model/Wan2.2-TI2V-5B/",
    lora_path="/data_3d/y00965182/Project/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_lora_pets_circle_fixed_radius_based_zl/step-16100.safetensors",
    input_image_latent_path=None,
    save_diffusion_latent_path=None,
    skip_video_decode=False,
    block_size=(2, 2, 3),
    sparse_ratio=0.0,
    bsa_backend="sdpa_chunked",
    bsa_chunk_size=64,
    dump_attention_png=None,
    height=None,
    width=None,
    num_frames=81,
    seed=1,
    num_inference_steps=50,
    cfg_scale=5,
):
    start_time = time.time()

    block_tokens = block_size[0] * block_size[1] * block_size[2]
    if sparse_ratio > 0.0 and bsa_backend == "flex" and (block_tokens < 32 or block_tokens % 32 != 0):
        raise SystemExit(
            "Torch FlexAttention on CUDA requires bt*bh*bw to be a multiple of 32; "
            f"got block-size {block_size} (product={block_tokens}). "
            "Use --block-size 2,4,4 or --bsa-backend sdpa_chunked."
        )

    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": torch.device("cuda"),
        "onload_dtype": torch.bfloat16,
        "onload_device": torch.device("cuda"),
        "preparing_dtype": torch.bfloat16,
        "preparing_device": torch.device("cuda"),
        "computation_dtype": torch.bfloat16,
        "computation_device": torch.device("cuda"),
    }

    model_paths = os.path.join(model_paths, "")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=sorted(glob.glob(model_paths + "diffusion_pytorch_model*.safetensors")), **vram_config),
            ModelConfig(path=model_paths + "models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(path=model_paths + "Wan2.2_VAE.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(path=model_paths + "google/umt5-xxl", **vram_config),
    )

    pipe.load_lora(pipe.dit, lora_path, alpha=1)

    attention_recorder = None
    if sparse_ratio > 0.0:
        n, attention_recorder = enable_bsa(
            pipe.dit,
            block_size,
            sparse_ratio,
            backend=bsa_backend,
            chunk_size=bsa_chunk_size,
            record_attention=dump_attention_png is not None,
        )
        print(f"[BSA] enabled on {n} blocks  block_size={block_size}  sparsity={sparse_ratio}  backend={bsa_backend}")
    else:
        print("[bsa] sparse_ratio=0 -> dense baseline (AttentionModule untouched)")

    height, width = infer_height_width(input_image, input_image_latent_path, height=height, width=width)
    if input_image is not None:
        input_image = input_image.resize((width, height))
    print(f"The shape of video(H x W @ T): {height} x {width} @ {num_frames}")

    load_time = time.time()
    print(f"[wan] model loading time: {load_time - start_time} s")
    print(f"[wan] using prompt: {prompt}")

    with torch.inference_mode():
        output = run_pipe_and_save_outputs(
            pipe=pipe,
            input_image=input_image,
            input_image_path=input_image_path,
            input_image_latent_path=input_image_latent_path,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            save_dir=save_dir,
            save_diffusion_latent_path=save_diffusion_latent_path,
            skip_video_decode=skip_video_decode,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            lora_path=lora_path,
        )

    print(f"[Wan] inference time: {time.time() - load_time} s")

    if dump_attention_png and attention_recorder is not None:
        save_attention_png(attention_recorder, dump_attention_png, sparse_ratio)

    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Run simple Wan2.2-TI2V-5B B200/H100 inference.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--image_latent_path", type=str, default=None)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="examples/output")
    parser.add_argument("--save_diffusion_latent_path", type=str, default=None)
    parser.add_argument("--skip_video_decode", action="store_true")
    parser.add_argument("--model_paths", type=str, default="/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B/")
    parser.add_argument("--lora_path", type=str, default="models/step-66900.safetensors")
    parser.add_argument("--sparse-ratio", type=float, default=0.0)
    parser.add_argument("--block-size", type=str, default="2,2,3")
    parser.add_argument("--bsa-backend", choices=["flex", "sdpa_chunked"], default="sdpa_chunked")
    parser.add_argument("--bsa-chunk-size", type=int, default=64)
    parser.add_argument("--dump-attention-png", default=None)
    args = parser.parse_args()
    if args.skip_video_decode and args.save_diffusion_latent_path is None:
        parser.error("--skip_video_decode requires --save_diffusion_latent_path")
    return args


def main():
    start_time = time.time()
    args = parse_args()
    metadata_path = args.metadata_path or os.path.join(os.path.dirname(args.image_path), "metadata.csv")
    prompt = load_prompt_from_metadata(metadata_path, args.image_path)
    print(f"[meta] metadata: {metadata_path}")
    print(f"[meta] image   : {os.path.basename(args.image_path)}")
    print(f"[meta] prompt  : {prompt}")

    input_image = Image.open(args.image_path).convert("RGB")
    print(f"[image] loading time: {time.time() - start_time} s")

    inference_video_b200(
        input_image=input_image,
        input_image_path=args.image_path,
        prompt=prompt,
        save_dir=None if args.skip_video_decode else args.save_dir,
        model_paths=args.model_paths,
        lora_path=args.lora_path,
        input_image_latent_path=args.image_latent_path,
        save_diffusion_latent_path=args.save_diffusion_latent_path,
        skip_video_decode=args.skip_video_decode,
        block_size=parse_block_size(args.block_size),
        sparse_ratio=args.sparse_ratio,
        bsa_backend=args.bsa_backend,
        bsa_chunk_size=args.bsa_chunk_size,
        dump_attention_png=args.dump_attention_png,
    )


if __name__ == "__main__":
    main()
