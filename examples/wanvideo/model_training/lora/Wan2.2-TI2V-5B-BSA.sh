#!/usr/bin/env bash
set -euo pipefail

DATASET_BASE_PATH="${DATASET_BASE_PATH:-/root/autodl-tmp/hq_vsr_bsa_smoke}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}"
CACHE_PATH="${CACHE_PATH:-/root/autodl-tmp/bsa_ti2v_480x832x81_cache}"
OUTPUT_PATH="${OUTPUT_PATH:-/root/autodl-tmp/models/train/Wan2.2-TI2V-5B_bsa_lora}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"

MODEL_CONFIGS="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
DIT_MODEL_CONFIG="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors"

accelerate launch examples/wanvideo/model_training/train.py \
  --task "sft:data_process" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_num_workers 2 \
  --model_id_with_origin_paths "${MODEL_CONFIGS}" \
  --disable_common_file_redirect \
  --output_path "${CACHE_PATH}" \
  --extra_inputs "input_image"

TRAIN_CMD=(
  accelerate launch examples/wanvideo/model_training/train.py
  --task "sft:train"
  --dataset_base_path "${CACHE_PATH}"
  --height 480
  --width 832
  --num_frames 81
  --dataset_repeat 100
  --model_id_with_origin_paths "${DIT_MODEL_CONFIG}"
  --disable_common_file_redirect
  --learning_rate 1e-4
  --num_epochs 3
  --remove_prefix_in_ckpt "pipe.dit."
  --output_path "${OUTPUT_PATH}"
  --lora_base_model "dit"
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
  --lora_rank 32
  --extra_inputs "input_image"
  --use_gradient_checkpointing
  --gradient_accumulation_steps 1
  --dataset_num_workers 4
  --dataset_pin_memory
  --dataset_persistent_workers
  --dataset_prefetch_factor 4
  --optimizer_fused
  --bsa_enable
  --bsa_block_size "2,2,3"
  --bsa_backend "sdpa_chunked"
  --bsa_sparse_ratio_start 0.90
  --bsa_sparse_ratio 0.95
  --bsa_sparse_ratio_warmup_steps 500
  --bsa_sdpa_chunk_size 64
  --log_steps 1
  --enable_tensorboard
)

if [[ -n "${LORA_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--lora_checkpoint "${LORA_CHECKPOINT}")
fi

"${TRAIN_CMD[@]}"
