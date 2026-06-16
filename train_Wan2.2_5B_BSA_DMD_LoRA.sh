#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

: "${MAIN_PROCESS_PORT:=44556}"
: "${ACCELERATE_CONFIG:=./config_wan22_5B.yaml}"
: "${TRAIN_SCRIPT:=examples/wanvideo/model_training/train.py}"

: "${DATASET_BASE_PATH:=/root/autodl-tmp/BridgeData2-Smoke}"
: "${DATASET_METADATA_PATH:=/root/autodl-tmp/BridgeData2-Smoke/metadata.csv}"
: "${MODEL_PATHS:=/root/autodl-tmp/models/Wan2.2-TI2V-5B}"
: "${TEACHER_MODEL_PATHS:=${MODEL_PATHS}}"
: "${TOKENIZER_PATH:=}"

: "${TEACHER_LORA_CHECKPOINT:=}"
: "${STUDENT_INIT_LORA_CHECKPOINT:=${TEACHER_LORA_CHECKPOINT}}"
: "${FAKE_SCORE_INIT_LORA_CHECKPOINT:=${TEACHER_LORA_CHECKPOINT}}"

: "${HEIGHT:=256}"
: "${WIDTH:=256}"
: "${NUM_FRAMES:=81}"
: "${DATASET_REPEAT:=1}"
: "${NUM_EPOCHS:=1000}"
: "${SAVE_STEPS:=50}"
: "${MAX_TRAIN_STEPS:=100}"
: "${WEIGHT_DECAY:=0.0}"

: "${LORA_TARGET_MODULES:=q,k,v,o,ffn.0,ffn.2}"
: "${LORA_RANK:=32}"
: "${STUDENT_LR:=5e-7}"
: "${FAKE_SCORE_LR:=1e-7}"
: "${STUDENT_BETA1:=0.9}"
: "${STUDENT_BETA2:=0.999}"
: "${FAKE_SCORE_BETA1:=0.9}"
: "${FAKE_SCORE_BETA2:=0.999}"

: "${DMD_DENOISING_STEPS:=1000,750,500,250}"
: "${REAL_GUIDANCE_SCALE:=6.0}"
: "${FAKE_GUIDANCE_SCALE:=0.0}"
: "${FAKE_SCORE_UPDATES_PER_GENERATOR_UPDATE:=5}"
: "${FAKE_SCORE_LOSS_TYPE:=x0}"

: "${EXTRA_INPUTS:=input_image}"
: "${DATASET_NUM_WORKERS:=4}"
: "${DATASET_PREFETCH_FACTOR:=4}"
: "${DATASET_PIN_MEMORY:=1}"
: "${DATASET_PERSISTENT_WORKERS:=1}"
: "${ENABLE_TENSORBOARD:=1}"
: "${LOG_STEPS:=1}"

: "${BSA_ENABLE:=1}"
: "${BSA_BLOCK_SIZE:=3,7,3}"
: "${BSA_SPARSE_RATIO:=0.85}"
: "${BSA_SPARSE_RATIO_START:=}"
: "${BSA_SPARSE_RATIO_WARMUP_STEPS:=0}"
: "${BSA_BACKEND:=sdpa_chunked}"
: "${BSA_SDPA_CHUNK_SIZE:=64}"

: "${OUTPUT_PATH:=/root/autodl-tmp/outputs/Wan2.2-TI2V-5B_bsa_dmd_lora}"

add_flag() {
  local args_name="$1"
  local enabled="$2"
  local flag="$3"
  [[ "${enabled}" == "1" ]] && eval "${args_name}+=(\"\${flag}\")"
  return 0
}

add_value() {
  local args_name="$1"
  local flag="$2"
  local value="$3"
  [[ -n "${value}" ]] && eval "${args_name}+=(\"\${flag}\" \"\${value}\")"
  return 0
}

launch_train() {
  accelerate launch \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --config_file "${ACCELERATE_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    "$@"
}

dataset_args=(
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --dataset_repeat "${DATASET_REPEAT}"
)

loader_args=(--dataset_num_workers "${DATASET_NUM_WORKERS}")
add_flag loader_args "${DATASET_PIN_MEMORY}" --dataset_pin_memory
add_flag loader_args "${DATASET_PERSISTENT_WORKERS}" --dataset_persistent_workers
add_value loader_args --dataset_prefetch_factor "${DATASET_PREFETCH_FACTOR}"

model_args=(
  --model_paths "${MODEL_PATHS}"
  --teacher_model_paths "${TEACHER_MODEL_PATHS}"
)
add_value model_args --tokenizer_path "${TOKENIZER_PATH}"

dmd_lora_args=(
  --student_lora_rank "${LORA_RANK}"
  --fake_score_lora_rank "${LORA_RANK}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --dmd_denoising_steps "${DMD_DENOISING_STEPS}"
  --real_guidance_scale "${REAL_GUIDANCE_SCALE}"
  --fake_guidance_scale "${FAKE_GUIDANCE_SCALE}"
  --fake_score_updates_per_generator_update "${FAKE_SCORE_UPDATES_PER_GENERATOR_UPDATE}"
  --student_learning_rate "${STUDENT_LR}"
  --fake_score_learning_rate "${FAKE_SCORE_LR}"
  --student_beta1 "${STUDENT_BETA1}"
  --student_beta2 "${STUDENT_BETA2}"
  --fake_score_beta1 "${FAKE_SCORE_BETA1}"
  --fake_score_beta2 "${FAKE_SCORE_BETA2}"
  --fake_score_loss_type "${FAKE_SCORE_LOSS_TYPE}"
)
add_value dmd_lora_args --teacher_lora_checkpoint "${TEACHER_LORA_CHECKPOINT}"
add_value dmd_lora_args --student_init_lora_checkpoint "${STUDENT_INIT_LORA_CHECKPOINT}"
add_value dmd_lora_args --fake_score_init_lora_checkpoint "${FAKE_SCORE_INIT_LORA_CHECKPOINT}"

bsa_args=()
if [[ "${BSA_ENABLE}" == "1" ]]; then
  bsa_args+=(
    --bsa_enable
    --bsa_block_size "${BSA_BLOCK_SIZE}"
    --bsa_backend "${BSA_BACKEND}"
    --bsa_sparse_ratio "${BSA_SPARSE_RATIO}"
    --bsa_sdpa_chunk_size "${BSA_SDPA_CHUNK_SIZE}"
  )
  add_value bsa_args --bsa_sparse_ratio_start "${BSA_SPARSE_RATIO_START}"
  [[ "${BSA_SPARSE_RATIO_WARMUP_STEPS}" != "0" ]] && \
    bsa_args+=(--bsa_sparse_ratio_warmup_steps "${BSA_SPARSE_RATIO_WARMUP_STEPS}")
fi

train_args=(
  --task "dmd_lora"
  "${dataset_args[@]}"
  "${model_args[@]}"
  "${dmd_lora_args[@]}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_epochs "${NUM_EPOCHS}"
  --save_steps "${SAVE_STEPS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --output_path "${OUTPUT_PATH}"
  --extra_inputs "${EXTRA_INPUTS}"
  --use_gradient_checkpointing
  "${loader_args[@]}"
  "${bsa_args[@]}"
  --log_steps "${LOG_STEPS}"
)
add_flag train_args "${ENABLE_TENSORBOARD}" --enable_tensorboard

launch_train "${train_args[@]}"
