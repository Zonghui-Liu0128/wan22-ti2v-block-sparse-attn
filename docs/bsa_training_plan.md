# BSA Training Plan for Wan2.2-TI2V-5B

## Goal

Train a LoRA that is robust under BSA inference for 480x832@81 horizontal 360-degree toy rotation videos, while preserving detail and color at `bsa_sparse_ratio=0.95`.

## Recommended Experiment

1. Cache training inputs first.
   - Run `task=sft:data_process` on the 480x832@81 dataset.
   - This removes VAE and text-encoder work from the hot training loop and lets the A800/H100 spend more time inside DiT + BSA.

2. Train BSA LoRA from the best dense LoRA checkpoint.
   - `lora_rank=32`.
   - `lora_target_modules=q,k,v,o,ffn.0,ffn.2`.
   - `bsa_block_size=2,2,3`.
   - `bsa_backend=sdpa_chunked`.
   - Start sparse ratio at `0.90`, ramp to `0.95`.
   - Use bf16, gradient checkpointing, cached data, high dataloader prefetch, and fused AdamW when CUDA supports it.

3. Keep only three validation sparsity points.
   - `0.90`: quality reference, expected detail/color stable.
   - `0.93`: transition point.
   - `0.95`: target deployment point.
   - Do not spend long runs on `(2,8,8)` except as an ablation, because the coarse block grid has too few kept key blocks at high sparsity.

4. Use a short stabilization tail.
   - After reaching `0.95`, continue with lower LR for 5-10% of total steps.
   - Save the best checkpoint by validation visual score and loss trend, not only final step.

## Shape and Throughput Accounting

Wan2.2-TI2V-5B uses `WanVideoVAE38` with spatial factor 16 and DiT patch size `(1,2,2)`.

For 480x832@81:

- latent frames: `(81 - 1) / 4 + 1 = 21`
- DiT patch grid: `480 / 32 = 15`, `832 / 32 = 26`
- tokens/video: `21 * 15 * 26 = 8190`

Training logs should report loss plus:

- tokens/s, tokens/hour, tokens/day
- videos/s, videos/hour, videos/day
- current BSA sparse ratio
- step time and elapsed time

## AutoDL Notes

Use the AutoDL instance with `tmux` or `screen` before long training. Dependencies are installed with conda/pip and persist on the instance system disk. Smoke data can be uploaded with `scp -rP` or a tar stream into `/root/autodl-tmp`.

## Default A800 80G Command Shape

Use cached data for the main run:

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --task "sft:train" \
  --dataset_base_path /root/autodl-tmp/bsa_cache \
  --dataset_repeat 200 \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 3 \
  --output_path "./models/train/Wan2.2-TI2V-5B_bsa_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --lora_checkpoint ./models/train/Wan2.2-TI2V-5B_lora/step-best.safetensors \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 4 \
  --dataset_pin_memory \
  --dataset_persistent_workers \
  --dataset_prefetch_factor 4 \
  --optimizer_fused \
  --bsa_enable \
  --bsa_block_size 2,2,3 \
  --bsa_backend sdpa_chunked \
  --bsa_sparse_ratio_start 0.90 \
  --bsa_sparse_ratio 0.95 \
  --bsa_sparse_ratio_warmup_steps 500 \
  --bsa_sdpa_chunk_size 64 \
  --log_steps 1 \
  --enable_tensorboard
```
