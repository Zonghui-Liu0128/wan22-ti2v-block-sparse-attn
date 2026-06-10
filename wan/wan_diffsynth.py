import glob
import time
from pathlib import Path

import numpy as np
import torch

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video
from examples.wanvideo.model_inference.run_bsa_test_ti2v import (
    NEGATIVE_PROMPT,
    build_diffusion_latent_save_obj,
    enable_bsa,
    save_attention_png,
)


def _save_diffusion_latent(
    latent,
    save_path,
    input_image_path,
    input_image_latent_path,
    prompt,
    lora_path,
    height,
    width,
    num_frames,
    seed,
    steps,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_diffusion_latent_save_obj(
            latent,
            image_path=input_image_path,
            image_latent_path=input_image_latent_path,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            lora_checkpoint=lora_path,
            lora_alpha=1,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            steps=steps,
        ),
        save_path,
    )
    print(f"[latent] saved denoised diffusion latent to {save_path}")


def inference_video(*args, **kwargs):
    return inference_video_b200(*args, **kwargs)


def inference_video_b200(input_image,
                    input_image_path,
                    prompt,
                    save_dir=None,
                    model_paths="/data_3d/w00950754/model/Wan2.2-TI2V-5B/",
                    lora_path="/data_3d/y00965182/Project/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_lora_pets_circle_fixed_radius_based_zl/step-16100.safetensors",
                    block_size=(2, 2, 3), sparse_ratio=0.0, bsa_backend="sdpa_chunked", bsa_chunk_size=64, dump_attention_png=None,
                    input_image_latent_path=None, save_diffusion_latent_path=None, skip_video_decode=False):

    start_time = time.time()

    block_tokens = block_size[0] * block_size[1] * block_size[2]
    if sparse_ratio > 0.0 and bsa_backend == "flex" and (block_tokens < 32 or block_tokens % 32 != 0):
        raise SystemExit(
            "Torch FlexAttention on CUDA requires bt*bh*bw to be a multiple of 32; "
            f"got block-size {block_size} (product={block_tokens}). "
            "Use --block-size 2,4,4 or --bsa-backend sdpa_chunked."
        )

    # -------------------- VRAM 配置 --------------------
    # B200 原生适合 bf16，建议全链路统一 bf16，避免 fp16/bf16 来回转换。
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

    # -------------------- 创建 pipeline --------------------
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=sorted(glob.glob(model_paths + "diffusion_pytorch_model*.safetensors")), **vram_config),
            ModelConfig(path=model_paths + "models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(path=model_paths + "Wan2.2_VAE.pth", **vram_config),
            # ModelConfig(path=model_paths+"models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(path=model_paths + "google/umt5-xxl", **vram_config)
    )

    # -------------------- 加载 LoRA --------------------
    if lora_path is not None:
        pipe.load_lora(pipe.dit, lora_path, alpha=1)

    # -------------------- 应用 BSA 稀疏化 --------------------
    attention_recorder = None
    if sparse_ratio > 0.0:
        n, attention_recorder = enable_bsa(
            pipe.dit, block_size, sparse_ratio,
            backend=bsa_backend, chunk_size=bsa_chunk_size,
            record_attention=dump_attention_png is not None,
        )
        print(f"[BSA] enabled on {n} blocks  block_size={block_size}  "
              f"sparsity={sparse_ratio}  backend={bsa_backend}")
    else:
        print("[bsa] sparse_ratio=0 -> dense baseline (AttentionModule untouched)")

    # -------------------- 计算高度宽度 --------------------
    max_area = 480 * 832
    aspect_ratio = input_image.height / input_image.width
    num_frames = 81
    mod_value = 8 * 2
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    input_image = input_image.resize((width, height))
    print(f"The shape of video(H x W @ T): {height} x {width} @ {num_frames}")

    load_time = time.time()
    print(f"[wan] 加载模型，总耗时: {(load_time - start_time)} s")

    # -------------------- 推理 --------------------
    # 只关闭 autograd，不改变采样参数和模型效果设置。
    print(f"[wan] using prompt: {prompt}")
    seed = 1
    steps = 50
    should_save_latent = save_diffusion_latent_path is not None
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            input_image=input_image,
            input_image_latent=input_image_latent_path,
            height=height,
            width=width,
            seed=seed,
            tiled=True,
            num_inference_steps=steps,
            cfg_scale=5,
            output_type="latent" if skip_video_decode else "quantized",
            return_latents=should_save_latent and not skip_video_decode,
        )

    if skip_video_decode:
        video = None
        diffusion_latent = result
    elif should_save_latent:
        video, diffusion_latent = result
    else:
        video = result
        diffusion_latent = None

    print(f"[Wan] 模型推理，总耗时: {(time.time() - load_time)} s")

    if should_save_latent:
        _save_diffusion_latent(
            diffusion_latent,
            save_diffusion_latent_path,
            input_image_path,
            input_image_latent_path,
            prompt,
            lora_path,
            height,
            width,
            num_frames,
            seed,
            steps,
        )

    if dump_attention_png and attention_recorder is not None:
        save_attention_png(attention_recorder, dump_attention_png, sparse_ratio)

    if video is not None and save_dir is not None:
        save_video(video, f"{save_dir}/{Path(input_image_path).stem}.mp4", fps=15, quality=5)
        print(f"Video saved as {save_dir}/{Path(input_image_path).stem}.mp4")

    return diffusion_latent if skip_video_decode else video
