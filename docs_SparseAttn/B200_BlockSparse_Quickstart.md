# B200 Block Sparse 直推与测试简明指南

适用: 内网 B200 机器, Wan2.2-TI2V-5B, block size `(2,8,8)`。

## 1. 环境

```bash
git clone <repo-url>
cd DiffSynth-Studio
pip install -e .
pip install imageio imageio[ffmpeg] transformers modelscope accelerate peft ftfy
```

确认 GPU:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

## 2. 逻辑测试

```bash
python -m pytest tests/test_block_sparse_3d.py tests/test_block_sparse_training_infra.py -q
```

预期: `6 passed`。

## 3. 三个输入 case 直推

输入图放在 `inputs/`, 输出写到 `outputs/`。脚本会从高到低尝试稀疏率, 第一个成功的稀疏率即该 case 的最高可用稀疏率。

```bash
python examples/wanvideo/model_inference/Wan2.2-TI2V-5B-BlockSparse-inputs.py \
  --inputs_dir inputs \
  --outputs_dir outputs \
  --height 320 \
  --width 576 \
  --num_frames 17 \
  --num_inference_steps 4 \
  --block_size 2,8,8 \
  --q_chunk_blocks 16 \
  --sparsity_candidates 0.99,0.975,0.95,0.925,0.90,0.875,0.85 \
  --debug_layer 0 \
  --debug_head 0 \
  --device cuda \
  --tiled
```

结果:

```text
outputs/wan22_ti2v_block_sparse_summary.json
outputs/<case>/video_sparsity_*.mp4
outputs/<case>/mask_sparsity_*/layer0_summary.json
outputs/<case>/mask_sparsity_*/layer0_mask_head0.txt
```

## 4. OOM 调参顺序

1. 降低 `--q_chunk_blocks`: `16 -> 8 -> 4 -> 2`。
2. 降低 `--height/--width` 或 `--num_frames`。
3. 降低最高候选稀疏率, 例如从 `0.99` 改为 `0.95` 开始。

## 5. 训练 Smoke

先跑 1 step:

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B/metadata.csv \
  --height 320 --width 576 --num_frames 17 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --max_train_steps 1 \
  --output_path ./models/train/Wan2.2-TI2V-5B_block_sparse_smoke \
  --training_log_file ./models/train/Wan2.2-TI2V-5B_block_sparse_smoke/training_log.jsonl \
  --trainable_models dit \
  --extra_inputs input_image \
  --use_gradient_checkpointing \
  --enable_block_sparse_attn \
  --block_sparse_block_size 2,8,8 \
  --block_sparse_start_sparsity 0.85 \
  --block_sparse_end_sparsity 0.90 \
  --block_sparse_ramp_steps 100 \
  --block_sparse_q_chunk_blocks 2
```

画训练曲线:

```bash
python docs_SparseAttn/plot_training_log.py \
  models/train/Wan2.2-TI2V-5B_block_sparse_smoke/training_log.jsonl
```
