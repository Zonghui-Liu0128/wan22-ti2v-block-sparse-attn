# BSA H100 Quickstart

这份文档用于在内网 H100 服务器上快速启动 Wan2.2-TI2V-5B 的 BSA LoRA 训练、查看训练日志和做 evaluation。当前方案不是全参数微调，而是 DiT LoRA 训练，BSA 只替换 self-attention 的计算路径。

## 改动概览

- `diffsynth/models/wan_bsa.py`：新增 Wan DiT self-attention 的 BSA 配置、恢复和 sparse ratio 更新工具。
- `examples/wanvideo/model_training/train.py`：新增 BSA 训练参数、sparse ratio warmup、cached train 支持、视频任务下的音频依赖懒加载。
- `diffsynth/diffusion/runner.py`：新增 fused AdamW、pin/persistent/prefetch dataloader、每步 metrics 写入和 BSA ratio 更新 hook。
- `diffsynth/diffusion/training_metrics.py`：新增 `training_metrics.csv/jsonl/html` 和可选 TensorBoard；480x832@81 记为 `8190 tokens/video`。
- `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-BSA.sh`：新增推荐训练脚本，先 cache，再用 cached data 训练 BSA LoRA。
- `examples/wanvideo/model_inference/run_bsa_test_ti2v.py`：新增 BSA 推理入口，并支持 `--lora-checkpoint` / `--lora-alpha` 做 BSA+LoRA evaluation。
- `examples/wanvideo/model_training/prepare_hq_vsr_smoke_dataset.py`：从 HQ-VSR zip 中抽样并转成 480x832@81 冒烟集。

## 数据和模型准备

训练数据目录建议保持：

```text
/data/bsa_toy_480x832x81/
  metadata.csv
  videos/
    xxx.mp4
```

`metadata.csv` 至少包含：

```csv
video,prompt
videos/xxx.mp4,A high quality video of a toy rotating horizontally 360 degrees with stable color and fine details.
```

确认模型文件在内网机器可被 DiffSynth 找到。默认脚本使用：

```text
Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors
Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth
Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth
```

如果内网模型目录不同，直接改 `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-BSA.sh` 里的 `MODEL_CONFIGS` 和 `DIT_MODEL_CONFIG`。

## 快速启动训练

推荐先在 `tmux` 里跑，避免 SSH 断开：

```bash
cd /path/to/DiffSynth-Studio
tmux new -s bsa_h100

conda activate diffsynth
python -m pip install -e .
python -m pip install peft ftfy tensorboard
```

配置路径：

```bash
export DATASET_BASE_PATH=/data/bsa_toy_480x832x81
export DATASET_METADATA_PATH=$DATASET_BASE_PATH/metadata.csv
export CACHE_PATH=/data/cache/bsa_ti2v_480x832x81_cache
export OUTPUT_PATH=/data/outputs/Wan2.2-TI2V-5B_bsa_lora

# 强烈建议从已有 dense LoRA 起训；没有就留空。
export LORA_CHECKPOINT=/data/outputs/Wan2.2-TI2V-5B_dense_lora/step-best.safetensors

mkdir -p "$OUTPUT_PATH"
bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-BSA.sh 2>&1 | tee "$OUTPUT_PATH/train.stdout.log"
```

默认关键训练配置：

- LoRA：`rank=32`，target `q,k,v,o,ffn.0,ffn.2`
- BSA：`block_size=2,2,3`，`backend=sdpa_chunked`，`chunk_size=64`
- sparse ratio：`0.90 -> 0.95`，warmup `500` step
- shape：`480x832@81`
- batch 等效：单进程每 step 1 个视频
- optimizer：fused AdamW
- dataloader：pin memory、persistent workers、prefetch factor 4

H100 上如果显存余量很大，可以先只把 `dataset_num_workers` 提到 8；再尝试把 `--bsa_sdpa_chunk_size` 从 64 提到 128。若 OOM，优先退回 64 或 32，不要先改分辨率和帧数。

## 查看日志和可视化

训练输出目录会生成：

```text
$OUTPUT_PATH/
  step-*.safetensors
  training_metrics.csv
  training_metrics.jsonl
  training_metrics.html
  tensorboard/
  train.stdout.log
```

命令行快速看：

```bash
tail -f "$OUTPUT_PATH/train.stdout.log"
tail -f "$OUTPUT_PATH/training_metrics.csv"
watch -n 5 nvidia-smi
```

HTML 曲线：

```bash
cd "$OUTPUT_PATH"
python -m http.server 7860
```

然后在浏览器打开：

```text
http://H100_SERVER_IP:7860/training_metrics.html
```

TensorBoard：

```bash
tensorboard --logdir "$OUTPUT_PATH/tensorboard" --host 0.0.0.0 --port 6006
```

CSV/HTML 中重点看：

- `loss`
- `tokens_per_hour` / `tokens_per_day`
- `videos_per_hour` / `videos_per_day`
- `bsa_sparse_ratio`
- `step_seconds`

480x832@81 的吞吐换算固定是 `8190 tokens/video`。

## Evaluation

每个 checkpoint 建议固定同一组 prompt、首帧、seed，对比 dense baseline 和 BSA 稀疏率 `0.90/0.93/0.95`。如果只有验证视频，先抽第一帧：

```bash
mkdir -p /data/eval/bsa
ffmpeg -y -i /data/valid/toy_rotate.mp4 -frames:v 1 /data/eval/bsa/input.png
```

指定要评估的 checkpoint。当前训练器默认保存 `step-*.safetensors`，不会自动生成 `step-best.safetensors`：

```bash
export EVAL_CKPT="$OUTPUT_PATH/step-1000.safetensors"
```

Dense LoRA baseline：

```bash
python examples/wanvideo/model_inference/run_bsa_test_ti2v.py \
  --image /data/eval/bsa/input.png \
  --prompt "A high quality video of a toy rotating horizontally 360 degrees with stable color and fine details." \
  --output /data/eval/bsa/step_best_dense.mp4 \
  --height 480 \
  --width 832 \
  --frames 81 \
  --steps 50 \
  --seed 1 \
  --lora-checkpoint "$EVAL_CKPT" \
  --lora-alpha 1.0 \
  --sparse-ratio 0
```

BSA target eval：

```bash
python examples/wanvideo/model_inference/run_bsa_test_ti2v.py \
  --image /data/eval/bsa/input.png \
  --prompt "A high quality video of a toy rotating horizontally 360 degrees with stable color and fine details." \
  --output /data/eval/bsa/step_best_bsa095.mp4 \
  --height 480 \
  --width 832 \
  --frames 81 \
  --steps 50 \
  --seed 1 \
  --lora-checkpoint "$EVAL_CKPT" \
  --lora-alpha 1.0 \
  --sparse-ratio 0.95 \
  --block-size 2,2,3 \
  --bsa-backend sdpa_chunked \
  --bsa-chunk-size 64 \
  --dump-attention-png /data/eval/bsa/step_best_bsa095_mask.png
```

建议 evaluation 表格至少记录：

```text
checkpoint | sparse_ratio | seed | steps | prompt_id | 是否完整 360 度 | 颜色是否漂移 | 细节是否丢失 | 是否崩坏
```

选择 checkpoint 时，不只看最后一步。优先选 `0.95` 下还能保持稳定水平旋转、颜色不明显漂移、细节不明显糊掉的 checkpoint。

## 快速自检

训练前：

```bash
python examples/wanvideo/model_training/train.py --help | grep -E "bsa_|training_metrics|tensorboard|optimizer_fused"
python examples/wanvideo/model_inference/run_bsa_test_ti2v.py --help | grep -E "lora|sparse|block"
```

训练后：

```bash
ls -lh "$OUTPUT_PATH"/training_metrics.*
python - <<'PY'
import csv, os
p = os.environ["OUTPUT_PATH"] + "/training_metrics.csv"
rows = list(csv.DictReader(open(p)))
print("steps", len(rows))
print("first_loss", rows[0]["loss"])
print("last_loss", rows[-1]["loss"])
print("last_tokens_per_hour", rows[-1]["tokens_per_hour"])
print("last_videos_per_day", rows[-1]["videos_per_day"])
print("last_bsa_sparse_ratio", rows[-1]["bsa_sparse_ratio"])
PY
```
