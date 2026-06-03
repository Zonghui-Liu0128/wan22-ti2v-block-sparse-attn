# BSA H100 验证证据

本目录保留 AutoDL 阶段验证产物，用来证明 BSA LoRA 训练和推理链路已经跑通。

验证环境：

```text
AutoDL A800-SXM4-80GB
PyTorch 2.7.1+cu128
Wan2.2-TI2V-5B
1 GPU smoke config
```

训练 smoke：

```text
dataset=/root/autodl-tmp/hq_vsr_bsa_smoke
height=128
width=128
num_frames=17
max_train_steps=20
bsa_block_size=3,7,3
bsa_sparse_ratio=0.85
bsa_backend=sdpa_chunked
lora_rank=32
```

通过标准：

```text
BSA enabled on 30 Wan blocks
training_metrics.csv has 20 data rows
last step is 20
last bsa_sparse_ratio is 0.85
step-20.safetensors was saved remotely
LoRA checkpoint loaded in BSA inference smoke
BSA mask PNG and metadata were exported
```

文件说明：

```text
training_metrics.csv    每步 loss、吞吐、BSA sparse ratio
training_metrics.jsonl  原始逐步日志
training_metrics.html   loss/吞吐曲线页面
bsa_mask_step20.png     第一层 BSA block mask
bsa_mask_step20.txt     mask 元数据
infer_step20_bsa.mp4    加载 step-20 LoRA 后的 1-step BSA 推理 smoke
```

可视化：

![BSA mask](bsa_mask_step20.png)

视频：

[infer_step20_bsa.mp4](infer_step20_bsa.mp4)
