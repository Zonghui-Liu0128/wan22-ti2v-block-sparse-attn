import argparse
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan2.2-TI2V-5B on a local test image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--output", required=True, help="Path to the output mp4.")
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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
    print(f"Saved TI2V output to {output_path}")


if __name__ == "__main__":
    main()
