import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "case"


def parse_tuple3(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated integers, got {value!r}.")
    return tuple(parts)


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one sparsity candidate is required.")
    return sorted(values, reverse=True)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(args):
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float16
    return WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=args.device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        vram_limit=args.vram_limit,
    )


def run_case(pipe, image_path: Path, args, sparsity: float, case_dir: Path):
    prompt = image_path.stem
    input_image = Image.open(image_path).convert("RGB").resize((args.width, args.height))
    debug_dir = case_dir / f"mask_sparsity_{sparsity:.3f}"
    pipe.enable_block_sparse_attention(
        sparsity=sparsity,
        block_size=args.block_size,
        q_chunk_blocks=args.q_chunk_blocks,
        dense_fallback_threshold=args.dense_fallback_threshold,
        debug=True,
        debug_layer=args.debug_layer,
        debug_head=args.debug_head,
        debug_output=str(debug_dir),
    )
    video = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        tiled=args.tiled,
        height=args.height,
        width=args.width,
        input_image=input_image,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
    )
    video_path = case_dir / f"video_sparsity_{sparsity:.3f}.mp4"
    save_video(video, str(video_path), fps=args.fps, quality=args.quality)
    return {
        "case": image_path.name,
        "prompt": prompt,
        "sparsity": sparsity,
        "video_path": str(video_path),
        "debug_dir": str(debug_dir),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_dir", type=Path, default=Path("inputs"))
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--block_size", type=parse_tuple3, default=(2, 8, 8))
    parser.add_argument("--q_chunk_blocks", type=int, default=16)
    parser.add_argument("--dense_fallback_threshold", type=int, default=0)
    parser.add_argument("--sparsity_candidates", type=parse_float_list, default=parse_float_list("0.99,0.975,0.95,0.925,0.9,0.875,0.85,0.8"))
    parser.add_argument("--debug_layer", type=int, default=0)
    parser.add_argument("--debug_head", type=int, default=0)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--vram_limit", type=float, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留",
    )
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in args.inputs_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {args.inputs_dir}.")

    pipe = load_pipeline(args)
    results = []
    for image_path in image_paths:
        case_dir = args.outputs_dir / slugify(image_path.stem)
        case_dir.mkdir(parents=True, exist_ok=True)
        case_result = None
        errors = []
        for sparsity in args.sparsity_candidates:
            try:
                case_result = run_case(pipe, image_path, args, sparsity, case_dir)
                break
            except Exception as exc:
                errors.append({"sparsity": sparsity, "error": repr(exc)})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if case_result is None:
            case_result = {
                "case": image_path.name,
                "prompt": image_path.stem,
                "sparsity": None,
                "video_path": None,
                "debug_dir": None,
                "errors": errors,
            }
        else:
            case_result["errors"] = errors
        results.append(case_result)

    summary_path = args.outputs_dir / "wan22_ti2v_block_sparse_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
