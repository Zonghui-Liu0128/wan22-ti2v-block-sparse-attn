import argparse
import csv
import os
import time

from PIL import Image

from wan.wan_diffsynth import inference_video, inference_video_b200


def parse_block_size(s):
    parts = [int(x) for x in s.split(",")]
    assert len(parts) == 3, f"block-size must have 3 ints, got {s!r}"
    return tuple(parts)


def load_prompt_from_metadata(metadata_path, image_path):
    """
    根据 image_path 的文件名，在 metadata.csv 的 image 列中匹配，返回对应的 prompt。
    metadata.csv 需包含表头: image,prompt
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"找不到 metadata 文件: {metadata_path}")

    image_name = os.path.basename(image_path)

    matched_prompt = None
    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "image" not in reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError(
                f"metadata.csv 必须包含 'image' 和 'prompt' 两列，"
                f"实际列为: {reader.fieldnames}"
            )
        for row in reader:
            # 同时兼容 csv 中带或不带目录前缀的写法
            if os.path.basename(row["image"].strip()) == image_name:
                matched_prompt = row["prompt"]
                break

    if matched_prompt is None:
        raise KeyError(
            f"在 {metadata_path} 中未找到与图像 '{image_name}' 匹配的 prompt，"
            f"请检查 image_path 与 metadata.csv 是否对应。"
        )

    return matched_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with Wan.")

    parser.add_argument("--image_path",
                        type=str,
                        default="examples/input/example1.jpg",
                        help="Path to the input image")

    parser.add_argument("--image_latent_path",
                        type=str,
                        default=None,
                        help="Path to the .pt first-frame latent saved by the RDP image encoder.")

    parser.add_argument("--metadata_path",
                        type=str,
                        default="testset/0302_Samples_SDR-flux-segment4_seg/qwen3_prompt_modified.csv",
                        help="Path to metadata.csv (含 image,prompt 两列)。"
                             "若不指定，则默认在 image_path 所在目录下查找 metadata.csv。")

    parser.add_argument("--save_dir",
                        type=str,
                        default="examples/output",
                        help="Path to the output video")

    parser.add_argument("--save_diffusion_latent_path",
                        type=str,
                        default=None,
                        help="Save the denoised Wan diffusion embedding before video decode.")

    parser.add_argument("--skip_video_decode",
                        action="store_true",
                        help="Only save/return the denoised diffusion embedding and skip Wan VAE video decode.")

    parser.add_argument("--model_paths",
                        type=str,
                        default="/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B/",
                        help="Path to Wan2.2-TI2V-5B")

    parser.add_argument("--lora_path",
                        type=str,
                        default="models/step-66900.safetensors",
                        help="Path to the LoRA ckpt")

    parser.add_argument("--sparse-ratio", type=float, default=0.0,
                        help="Block-drop fraction (0 = dense baseline, 0.5 = drop half, etc.).")
    parser.add_argument("--block-size", type=str, default="2,2,3",
                        help="Block size as 'bt,bh,bw' (default 2,2,3).")
    parser.add_argument("--bsa-backend", choices=["flex", "sdpa_chunked"], default="sdpa_chunked")
    parser.add_argument("--bsa-chunk-size", type=int, default=64,
                        help="chunk_size for sdpa_chunked backend.")
    parser.add_argument("--dump-attention-png", default=None,
                        help="Save the first DiT block's final block-level BSA mask as a PNG.")

    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_args()

    if args.metadata_path is None:
        metadata_path = os.path.join(os.path.dirname(args.image_path), "metadata.csv")
    else:
        metadata_path = args.metadata_path

    prompt = load_prompt_from_metadata(metadata_path, args.image_path)
    print(f"[meta] metadata: {metadata_path}")
    print(f"[meta] image   : {os.path.basename(args.image_path)}")
    print(f"[meta] prompt  : {prompt}")

    input_image = Image.open(args.image_path).convert("RGB")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[flux] 加载图像总耗时: {(time.time() - start_time)} s")
    flux_time = time.time()

    block_size = parse_block_size(args.block_size)

    inference_video_b200(
        input_image=input_image,
        input_image_path=args.image_path,
        prompt=prompt,
        save_dir=args.save_dir,
        model_paths=args.model_paths,
        lora_path=args.lora_path,
        block_size=block_size,
        sparse_ratio=args.sparse_ratio,
        bsa_backend=args.bsa_backend,
        bsa_chunk_size=args.bsa_chunk_size,
        dump_attention_png=args.dump_attention_png,
        input_image_latent_path=args.image_latent_path,
        save_diffusion_latent_path=args.save_diffusion_latent_path,
        skip_video_decode=args.skip_video_decode,
    )

    print(f"[wan] 运行完毕，总耗时: {(time.time() - flux_time) / 60} min")


if __name__ == "__main__":
    main()
