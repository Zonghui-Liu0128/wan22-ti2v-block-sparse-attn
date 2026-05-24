modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.2-TI2V-5B/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B/metadata.csv \
  --height 320 \
  --width 576 \
  --num_frames 17 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_block_sparse" \
  --training_log_file "./models/train/Wan2.2-TI2V-5B_block_sparse/training_log.jsonl" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing \
  --enable_block_sparse_attn \
  --block_sparse_block_size "2,8,8" \
  --block_sparse_sparsity 0.90 \
  --block_sparse_start_sparsity 0.85 \
  --block_sparse_end_sparsity 0.90 \
  --block_sparse_ramp_steps 100 \
  --block_sparse_q_chunk_blocks 2 \
  --block_sparse_dense_fallback_threshold 0
