# Wan2.2-5B BSA LoRA 内网 H100 部署手册

这份手册面向内网 H100*4 环境，用于把原来的 Wan2.2-TI2V-5B full self-attention LoRA 训练切换到 BSA 训练，并保留内网训练效果对齐逻辑。

## 一句话结论

- 默认启动脚本仍走内网原始 `sft` 训练路径：每个 batch 现场跑 VAE/T5/DiT，首帧会被加噪并纳入 loss。
- BSA 只替换 Wan DiT 的 self-attention 计算路径，不改变 LoRA target、loss 语义、数据字段和本地模型加载方式。
- `USE_CACHE=1` 是可选优化：先 `sft:data_process` 缓存确定性的 VAE/T5 结果，再 `sft:train` 只跑 DiT+BSA+LoRA。只要缓存数据没有随机增强且模型/数据版本一致，训练目标和 direct `sft` 等价。

## 必须同步到内网的文件

把下面文件覆盖到内网 DiffSynth-Studio 对应路径：

```text
train_Wan2.2_5B_LoRA.sh
config_wan22_5B.yaml
examples/wanvideo/model_training/train.py
examples/wanvideo/model_inference/run_bsa_test_ti2v.py
diffsynth/models/wan_video_dit.py
diffsynth/models/wan_bsa.py
diffsynth/diffusion/parsers.py
diffsynth/diffusion/runner.py
diffsynth/diffusion/training_metrics.py
diffsynth/diffusion/training_module.py
diffsynth/diffusion/loss.py
diffsynth/diffusion/flow_match.py
tests/test_bsa_training_tools.py
tests/test_wan_bsa_unit.py
```

其中 `loss.py` 要重点确认：`FlowMatchSFTLoss` 不应把 `first_frame_latents` 覆盖到 noisy latents，也不应裁掉首帧 loss；这是对齐内网 full self-attn 训练效果的关键。

## `config_wan22_5B.yaml` 是什么

`config_wan22_5B.yaml` 是 Accelerate 的启动配置，不是模型配置。它决定 `accelerate launch` 如何创建分布式训练进程。

当前 H100*4 配置含义：

```text
distributed_type: DEEPSPEED  使用 DeepSpeed
num_processes: 4             本机 4 个 GPU 进程
mixed_precision: bf16        H100 上用 bf16
zero_stage: 2                ZeRO-2 分片优化器状态
gradient_accumulation_steps: 1
offload_*: none              不把参数/优化器 offload 到 CPU
```

内网是 4 张 H100 时直接使用仓库根目录的 `config_wan22_5B.yaml`。如果临时只用 1 张卡测试，不能用这份 4 进程配置，需要另写 `num_processes: 1` 且 `distributed_type: 'NO'` 的 smoke 配置。

## 启动训练

默认配置已经写进 `train_Wan2.2_5B_LoRA.sh`：

- 数据：`metadata_dataset_dolls_480p.csv`
- 分辨率/帧数：`832x480@81`
- 本地模型目录：`Wan2.2-TI2V-5B/`
- LoRA：`rank=32`，target `q,k,v,o,ffn.0,ffn.2`
- 初始 LoRA checkpoint：`models/step-66900.safetensors`
- BSA：`block_size=3,7,3`，`backend=sdpa_chunked`，`sparse_ratio=0.85`
- 日志：CSV/JSONL/HTML + TensorBoard

直接启动：

```bash
cd /path/to/DiffSynth-Studio
conda activate diffsynth
python -m pip install -e .

bash train_Wan2.2_5B_LoRA.sh 2>&1 | tee bsa_train.log
```

训练 90% 稀疏率：

```bash
BSA_SPARSE_RATIO=0.90 bash train_Wan2.2_5B_LoRA.sh
```

从 85% warmup 到 90%：

```bash
BSA_SPARSE_RATIO_START=0.85 \
BSA_SPARSE_RATIO=0.90 \
BSA_SPARSE_RATIO_WARMUP_STEPS=500 \
bash train_Wan2.2_5B_LoRA.sh
```

20 step 冒烟测试：

```bash
MAX_TRAIN_STEPS=20 SAVE_STEPS=20 OUTPUT_PATH=/path/to/smoke_out \
bash train_Wan2.2_5B_LoRA.sh
```

可选缓存训练：

```bash
USE_CACHE=1 CACHE_PATH=/path/to/cache OUTPUT_PATH=/path/to/bsa_lora \
bash train_Wan2.2_5B_LoRA.sh
```

缓存训练只建议在已经确认 direct `sft` 能跑通之后开启。它能省掉热路径里的 VAE/T5 计算，但正式效果对齐时先用 direct `sft` 更稳。

## BSA 是如何接入训练的

1. `WanTrainingModule` 加载模型后，在 LoRA 注入前调用 BSA 配置。
2. `configure_wan_bsa()` 遍历 `pipe.dit` / `pipe.dit2` 的 Wan blocks，把每个 `SelfAttention.attn` 替换成 `BlockSparseAttention`。
3. `BlockSparseAttention` 把 latent token 还原成 `(frames, height, width)` 3D 网格，按 `block_size=(3,7,3)` 分块。
4. 每个 query block 用 Q/K block mean 算相似度，按 `sparse_ratio` 丢掉最不相关的 key blocks。
5. `sdpa_chunked` 后端按 query block chunk 收集保留的 K/V tokens，再调用 PyTorch SDPA，避免构造完整 dense attention 矩阵。
6. LoRA 仍注入在 DiT 的 `q,k,v,o,ffn.0,ffn.2`。反向传播经过 BSA attention 路径，只更新 LoRA 参数。
7. `runner.py` 每个 step 调用 `model.on_train_step_start()`，可线性 warmup sparse ratio，并把当前 ratio 写入 metrics。

和内网 full self-attn 版本的关键差异：

```text
full self-attn: SelfAttention -> dense SDPA -> LoRA 更新
BSA LoRA:       SelfAttention -> BlockSparseAttention(sdpa_chunked) -> LoRA 更新
```

其他训练语义保持一致：

- 本地目录式模型加载保持一致。
- 默认 direct `sft` 热路径仍包含 VAE/T5。
- 首帧加噪并进入 loss。
- LoRA checkpoint 加载路径和内网脚本保持一致。

## 查看 loss 和吞吐曲线

训练输出目录会生成：

```text
training_metrics.csv
training_metrics.jsonl
training_metrics.html
tensorboard/
step-*.safetensors
```

快速看 CSV：

```bash
tail -f "$OUTPUT_PATH/training_metrics.csv"
```

HTML 曲线：

```bash
cd "$OUTPUT_PATH"
python -m http.server 7860
```

浏览器打开：

```text
http://H100_SERVER_IP:7860/training_metrics.html
```

TensorBoard：

```bash
tensorboard --logdir "$OUTPUT_PATH/tensorboard" --host 0.0.0.0 --port 6006
```

重点列：

```text
loss
step_seconds
tokens_per_hour
videos_per_day
bsa_sparse_ratio
```

Wan2.2-TI2V-5B 的吞吐口径：VAE spatial factor 16，DiT patch `(1,2,2)`。`832x480@81` 对应 latent frames `21`、patch grid `26x15`，即 `8190 tokens/video`。

## 推理和 BSA mask 可视化

从验证视频抽首帧：

```bash
ffmpeg -y -i /path/to/valid.mp4 -frames:v 1 /path/to/input.png
```

加载训练好的 LoRA，并导出 BSA mask：

```bash
python examples/wanvideo/model_inference/run_bsa_test_ti2v.py \
  --image /path/to/input.png \
  --prompt "A high quality video of a toy rotating horizontally 360 degrees with stable color and fine details." \
  --output /path/to/eval_bsa085.mp4 \
  --height 832 \
  --width 480 \
  --frames 81 \
  --steps 50 \
  --seed 1 \
  --model-paths /path/to/Wan2.2-TI2V-5B \
  --lora-checkpoint "$OUTPUT_PATH/step-XXXX.safetensors" \
  --lora-alpha 1.0 \
  --sparse-ratio 0.85 \
  --block-size 3,7,3 \
  --bsa-backend sdpa_chunked \
  --bsa-chunk-size 64 \
  --dump-attention-png /path/to/bsa_mask.png
```

mask 颜色含义：

```text
yellow = 保留的 key block
purple = 被丢弃的 key block
```

## 已保留的验证证据

本仓库保留了 AutoDL 20-step 阶段验证文件，路径：

```text
docs/bsa_h100_verification/
```

内容：

- `training_metrics.csv`：20 条 step 记录，最后 step 为 20。
- `training_metrics.html`：loss/吞吐曲线可视化。
- `training_metrics.jsonl`：逐步原始日志。
- `bsa_mask_step20.png` / `.txt`：BSA mask 可视化和元数据。
- `infer_step20_bsa.mp4`：加载 step-20 LoRA 后的 1-step BSA 推理 smoke。

验证环境是 AutoDL A800 80GB，因为该实例只有 1 张 GPU；它验证的是代码链路，不代表 H100*4 吞吐。验证结论：

```text
BSA enabled on 30 Wan blocks
block_size=(3,7,3)
sparse_ratio=0.85
backend=sdpa_chunked
metrics rows=20
last_step=20
checkpoint=step-20.safetensors
LoRA inference smoke passed
```

## 常见问题

`No such file or directory: models/step-66900.safetensors`

内网正式训练应确保该 checkpoint 存在。临时 smoke 不想加载 checkpoint 时：

```bash
LORA_CHECKPOINT= MAX_TRAIN_STEPS=20 bash train_Wan2.2_5B_LoRA.sh
```

`block_size=3,7,3` 为什么不用 flex？

`flex` 在 CUDA 上对 block token product 有限制；`3*7*3=63` 不适合走 flex。这里固定用 `sdpa_chunked`。

H100 上想跑 90% 稀疏率怎么办？

```bash
BSA_SPARSE_RATIO=0.90 BSA_BLOCK_SIZE=3,7,3 BSA_BACKEND=sdpa_chunked bash train_Wan2.2_5B_LoRA.sh
```

想进一步省显存怎么办？

优先开启缓存训练 `USE_CACHE=1`，再降低 `BSA_SDPA_CHUNK_SIZE=32`。不要先改分辨率、帧数或 loss 逻辑。
