#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
#export ASCEND_LAUNCH_BLOCKING=1

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-44556}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-./config_wan22_5B.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-examples/wanvideo/model_training/train.py}"

DATASET_BASE_PATH="${DATASET_BASE_PATH:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/Smart_Doll_Pet-master/datasets/dataset_dolls_480p}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/Smart_Doll_Pet-master/datasets/metadata/metadata_dataset_dolls_480p.csv}"
MODEL_PATHS="${MODEL_PATHS:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/Smart_Doll_Pet-master/outputs/train/Wan2.2-TI2V-5B_lora_pets/bsa_${BSA_SPARSE_RATIO:-0.85}}"
CACHE_PATH="${CACHE_PATH:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/Smart_Doll_Pet-master/outputs/cache/Wan2.2-TI2V-5B_lora_pets_832x480x81}"

HEIGHT="${HEIGHT:-832}"
WIDTH="${WIDTH:-480}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-400}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"

LORA_BASE_MODEL="${LORA_BASE_MODEL:-dit}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q,k,v,o,ffn.0,ffn.2}"
LORA_RANK="${LORA_RANK:-32}"
LORA_CHECKPOINT="${LORA_CHECKPOINT-models/step-66900.safetensors}"
EXTRA_INPUTS="${EXTRA_INPUTS:-input_image}"

DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-4}"
DATASET_PREFETCH_FACTOR="${DATASET_PREFETCH_FACTOR-4}"
DATASET_PIN_MEMORY="${DATASET_PIN_MEMORY:-1}"
DATASET_PERSISTENT_WORKERS="${DATASET_PERSISTENT_WORKERS:-1}"
OPTIMIZER_FUSED="${OPTIMIZER_FUSED:-1}"
ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD:-1}"
LOG_STEPS="${LOG_STEPS:-1}"

BSA_ENABLE="${BSA_ENABLE:-1}"
BSA_BLOCK_SIZE="${BSA_BLOCK_SIZE:-3,7,3}"
BSA_SPARSE_RATIO="${BSA_SPARSE_RATIO:-0.85}"
BSA_SPARSE_RATIO_START="${BSA_SPARSE_RATIO_START:-}"
BSA_SPARSE_RATIO_WARMUP_STEPS="${BSA_SPARSE_RATIO_WARMUP_STEPS:-0}"
BSA_BACKEND="${BSA_BACKEND:-sdpa_chunked}"
BSA_SDPA_CHUNK_SIZE="${BSA_SDPA_CHUNK_SIZE:-64}"

USE_CACHE="${USE_CACHE:-0}"

launch_args=(
  accelerate launch
  --main_process_port "${MAIN_PROCESS_PORT}"
  --config_file "${ACCELERATE_CONFIG}"
  "${TRAIN_SCRIPT}"
)

video_args=(
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
)

source_dataset_args=(
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  "${video_args[@]}"
  --dataset_repeat "${DATASET_REPEAT}"
)

loader_args=(
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
)
if [[ "${DATASET_PIN_MEMORY}" == "1" ]]; then
  loader_args+=(--dataset_pin_memory)
fi
if [[ "${DATASET_PERSISTENT_WORKERS}" == "1" ]]; then
  loader_args+=(--dataset_persistent_workers)
fi
if [[ -n "${DATASET_PREFETCH_FACTOR}" ]]; then
  loader_args+=(--dataset_prefetch_factor "${DATASET_PREFETCH_FACTOR}")
fi

lora_args=(
  --lora_base_model "${LORA_BASE_MODEL}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --lora_rank "${LORA_RANK}"
)
if [[ -n "${LORA_CHECKPOINT}" ]]; then
  lora_args+=(--lora_checkpoint "${LORA_CHECKPOINT}")
fi

model_args=(
  --model_paths "${MODEL_PATHS}"
)
if [[ -n "${TOKENIZER_PATH}" ]]; then
  model_args+=(--tokenizer_path "${TOKENIZER_PATH}")
fi

bsa_args=()
if [[ "${BSA_ENABLE}" == "1" ]]; then
  bsa_args+=(
    --bsa_enable
    --bsa_block_size "${BSA_BLOCK_SIZE}"
    --bsa_backend "${BSA_BACKEND}"
    --bsa_sparse_ratio "${BSA_SPARSE_RATIO}"
    --bsa_sdpa_chunk_size "${BSA_SDPA_CHUNK_SIZE}"
  )
  if [[ -n "${BSA_SPARSE_RATIO_START}" ]]; then
    bsa_args+=(--bsa_sparse_ratio_start "${BSA_SPARSE_RATIO_START}")
  fi
  if [[ "${BSA_SPARSE_RATIO_WARMUP_STEPS}" != "0" ]]; then
    bsa_args+=(--bsa_sparse_ratio_warmup_steps "${BSA_SPARSE_RATIO_WARMUP_STEPS}")
  fi
fi

train_args=(
  "${model_args[@]}"
  --num_epochs "${NUM_EPOCHS}"
  --save_steps "${SAVE_STEPS}"
  --remove_prefix_in_ckpt "pipe.dit."
  --output_path "${OUTPUT_PATH}"
  "${lora_args[@]}"
  --extra_inputs "${EXTRA_INPUTS}"
  --use_gradient_checkpointing
  "${loader_args[@]}"
  "${bsa_args[@]}"
  --log_steps "${LOG_STEPS}"
)
if [[ "${OPTIMIZER_FUSED}" == "1" ]]; then
  train_args+=(--optimizer_fused)
fi
if [[ "${ENABLE_TENSORBOARD}" == "1" ]]; then
  train_args+=(--enable_tensorboard)
fi
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  train_args+=(--max_train_steps "${MAX_TRAIN_STEPS}")
fi

if [[ "${USE_CACHE}" == "1" ]]; then
  "${launch_args[@]}" \
    --task "sft:data_process" \
    "${source_dataset_args[@]}" \
    "${model_args[@]}" \
    --output_path "${CACHE_PATH}" \
    --extra_inputs "${EXTRA_INPUTS}" \
    --use_gradient_checkpointing \
    "${loader_args[@]}"

  "${launch_args[@]}" \
    --task "sft:train" \
    --dataset_base_path "${CACHE_PATH}" \
    "${video_args[@]}" \
    --dataset_repeat "${DATASET_REPEAT}" \
    "${train_args[@]}"
else
  "${launch_args[@]}" \
    --task "sft" \
    "${source_dataset_args[@]}" \
    "${train_args[@]}"
fi
