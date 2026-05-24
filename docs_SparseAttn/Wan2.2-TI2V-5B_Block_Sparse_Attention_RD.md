# Wan2.2-TI2V-5B Block Sparse Attention 算法研究与开发文档

> 状态: 研发与实现同步文档
> 目标: Wan2.2-TI2V-5B DiT self-attention
> 约束: 纯 PyTorch 实现, 不引入新的 sparse attention 库
> 日期: 2026-05-24

## 1. 目标

本文档定义 Wan2.2-TI2V-5B self-attention 的 Block Sparse Attention 研究、实现和验证计划。

实现需要支持:

- 用户指定稀疏率 `sparsity`。
- 用户指定 3D block size: `(block_t, block_h, block_w)`。
- 当 latent grid 的 `f/h/w` 不能被 block size 整除时, 先 padding 0 到可整除形状。
- padding token 不参与 block-level top-k 选择, 也不参与 token-level exact attention。
- 推理阶段稀疏直推。
- 训练恢复阶段启用稀疏 self-attention。
- 训练 log 记录 loss 和吞吐量, 并提供可视化脚本。

核心约束: 划分进同一个 block 的 token 必须是 Wan latent grid 中时空相邻的 token, 不能把 flatten 后一维序列机械地每 `block_tokens` 个切成一块。因此实现必须显式对齐 Wan 的 patchify 和 RoPE 顺序。

## 2. Wan Token 顺序分析

Wan self-attention 的 hidden states 形状应为:

```text
x: (B, S, D), S = f * h * w
```

Wan 的 3D RoPE 频率在 `WanModel.forward` 中按如下逻辑构造:

```python
freqs = torch.cat([
    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
], dim=-1).reshape(f * h * w, 1, -1)
```

这说明 RoPE 假设的 flatten 顺序是:

```text
token_id = ((t * h) + y) * w + x
t = token_id // (h * w)
y = (token_id % (h * w)) // w
x = token_id % w
```

因此稀疏 block 划分必须先执行:

```text
(B, S, D) -> (B, f, h, w, H, Dh)
```

再沿 `(t, y, x)` 三个坐标轴切连续小立方体。这样每个 block 内 token 才是时空相邻的。

开发时必须加入 shape 断言:

```text
assert S == f * h * w
```

并在 debug 模式下打印前若干 block 的坐标范围:

```text
block 0: t[0:block_t), h[0:block_h), w[0:block_w)
block 1: t[0:block_t), h[0:block_h), w[block_w:2*block_w)
...
```

说明: 当前开发前还需要在实际代码里确认 `patchify` 后进入 self-attn 的张量是否已经按 `(f,h,w)` flatten 成 `(B,S,D)`。如果现有实现路径在 `patchify` 后仍是 `(B,D,f,h,w)`, 则稀疏接入点必须放在 flatten 之后, 或先补齐现有 flatten 路径, 再做 sparse partition。

## 3. 算法定义

记:

```text
B  = batch size
S  = f * h * w
D  = hidden dimension
H  = attention heads
Dh = D / H
b  = block_t * block_h * block_w
N  = padded 3D blocks 数量
rho = sparsity ratio
K  = 每个 query block 保留的 KV block 数量
```

Block Sparse Attention 分两阶段:

1. Block routing:
   对每个 block 内有效 token 做 masked mean, 得到 block-level `Q_repr/K_repr`, 然后计算 draft score 并选择 top-k KV blocks。

2. Exact sparse attention:
   仅对选中的 block pair 执行 token-level scaled dot-product attention。最终 softmax 和 `V` 加权仍在 token 粒度上精确计算, draft score 只负责路由。

padding 规则:

- Q/K/V 的 padded token 值可以填 0。
- block mean 必须用 valid-token mask, padded token 不进入均值。
- top-k 前必须屏蔽无效 KV block。
- exact attention 的 logits 必须屏蔽 selected KV blocks 中的 padded token。
- padded query token 的输出置 0, 最终 reverse partition 后 crop 掉。

注意: 如果只是将单个样本 padding 到 `ceil(f/block_t), ceil(h/block_h), ceil(w/block_w)`, 通常不会产生整块全 padding 的 block, 因为每个边界 block 至少包含一部分真实 token。真正必须重点验证的是“边界 block 内混有 padded token 时, padded token 不参与 mean/top-k/exact attention”。保留 block-level valid mask 仍然有必要, 以支持后续异形 batch 或人为 mask 测试。

## 4. 学术论文风格伪代码

### Algorithm 1: Wan 3D Block Sparse Self-Attention

```text
Algorithm 1 Wan3DBlockSparseSelfAttention
Input:
    X in R^{B x S x D}
    Freqs in R^{S x 1 x D_rope}
    grid shape G = (f, h, w)
    block size R = (r_t, r_h, r_w)
    sparsity ratio rho in [0, 1)
    number of heads H
    query block chunk size C
Output:
    Y in R^{B x S x D}

1:  Q <- RMSNorm_Q(Linear_Q(X))
2:  K <- RMSNorm_K(Linear_K(X))
3:  V <- Linear_V(X)
4:  Q <- ApplyRoPE(Q, Freqs, H)
5:  K <- ApplyRoPE(K, Freqs, H)
6:  O <- BlockSparseAttention3D(Q, K, V, G, R, rho, H, C)
7:  Y <- Linear_O(O)
8:  return Y
```

### Algorithm 2: 3D Block Sparse Attention

```text
Algorithm 2 BlockSparseAttention3D
Input:
    Q, K, V in R^{B x S x D}
    grid shape G = (f, h, w)
    block size R = (r_t, r_h, r_w)
    sparsity ratio rho
    number of heads H
    query block chunk size C
Output:
    O in R^{B x S x D}

1:  D_h <- D / H
2:  Assert S = f * h * w
3:  Reshape Q, K, V from (B, S, D) to (B, f, h, w, H, D_h)

4:  f' <- ceil(f / r_t) * r_t
5:  h' <- ceil(h / r_h) * r_h
6:  w' <- ceil(w / r_w) * r_w

7:  Pad Q, K, V with zeros to shape (B, f', h', w', H, D_h)
8:  M_token <- False^{f' x h' x w'}
9:  M_token[0:f, 0:h, 0:w] <- True

10: Partition Q, K, V into adjacent 3D blocks:
        Q_blk, K_blk, V_blk in R^{B x H x N x b x D_h}
11: Partition M_token:
        M_blk in {0,1}^{N x b}
12: M_block <- any(M_blk over token dimension)

13: Q_repr <- MaskedMean(Q_blk, M_blk)       // R^{B x H x N x D_h}
14: K_repr <- MaskedMean(K_blk, M_blk)       // R^{B x H x N x D_h}

15: Score <- MatMul(Q_repr, Transpose(K_repr)) / sqrt(D_h)
        // Score in R^{B x H x N x N}
16: Score[:, :, :, not M_block] <- -infinity

17: N_valid <- sum(M_block)
18: K_keep <- max(1, round(N_valid * (1 - rho)))
19: K_keep <- min(K_keep, N_valid)

20: I <- TopKIndices(Score, K_keep, dimension = KV block dimension)
21: O_blk <- zeros_like(Q_blk)

22: for q0 in {0, C, 2C, ...} do
23:     q1 <- min(q0 + C, N)
24:     Q_c <- Q_blk[:, :, q0:q1, :, :]
25:     M_q <- M_blk[q0:q1, :]
26:     I_c <- I[:, :, q0:q1, :]

27:     K_c <- GatherBlocks(K_blk, I_c)
28:     V_c <- GatherBlocks(V_blk, I_c)
29:     M_kv <- GatherBlocks(M_blk, I_c)

30:     Flatten selected KV blocks:
            K_tok in R^{B x H x (q1-q0) x (K_keep*b) x D_h}
            V_tok in R^{B x H x (q1-q0) x (K_keep*b) x D_h}
            M_tok in {0,1}^{B x H x (q1-q0) x (K_keep*b)}

31:     Logits <- MatMul(Q_c, Transpose(K_tok)) / sqrt(D_h)
            // Logits in R^{B x H x (q1-q0) x b x (K_keep*b)}
32:     Logits[..., not M_tok] <- -infinity
33:     P <- Softmax(Logits, dimension = selected KV token dimension)
34:     O_c <- MatMul(P, V_tok)
            // O_c in R^{B x H x (q1-q0) x b x D_h}
35:     O_c[..., not M_q, :] <- 0
36:     O_blk[:, :, q0:q1, :, :] <- O_c
37: end for

38: Reverse 3D block partition from O_blk to O_pad in R^{B x f' x h' x w' x H x D_h}
39: O <- Crop O_pad to R^{B x f x h x w x H x D_h}
40: Reshape O to R^{B x S x D}
41: return O
```

### Algorithm 3: Masked Block Mean

```text
Algorithm 3 MaskedMean
Input:
    X_blk in R^{B x H x N x b x D_h}
    M_blk in {0,1}^{N x b}
Output:
    X_repr in R^{B x H x N x D_h}

1:  W <- Cast(M_blk, dtype = X_blk.dtype)
2:  Denom <- Sum(W over token dimension)
3:  Denom <- Clamp(Denom, min = 1)
4:  X_sum <- Sum(X_blk * W over token dimension)
5:  X_repr <- X_sum / Denom
6:  X_repr[:, :, not any(M_blk), :] <- 0
7:  return X_repr
```

## 5. 实现设计

### 5.1 文件规划

新增:

- `diffsynth/core/attention/block_sparse_3d.py`
  - 3D block partition / reverse partition。
  - padding 和 valid-token mask。
  - block-level top-k route 生成。
  - selected blocks 上的 token-level exact sparse attention。
  - block 坐标和 attn mask debug 输出。

修改:

- `diffsynth/models/wan_video_dit.py`
  - 将 `grid_shape=(f,h,w)` 传入 self-attn。
  - 给 `AttentionModule` 或 `SelfAttention` 添加 opt-in sparse config。
  - dense attention 保持默认行为。

- `diffsynth/pipelines/wan_video.py`
  - 增加 enable block sparse self-attention 的 helper, 同时处理 `dit` 和 `dit2`。

- `examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py`
  - 增加 sparse inference 示例或 sparse flags。

- `examples/wanvideo/model_training/train.py`
  - 增加 sparse attention CLI flags。

- `diffsynth/diffusion/runner.py` 和 `diffsynth/diffusion/logger.py`
  - 增加 loss 和吞吐量日志。

新增:

- `examples/wanvideo/model_training/special/block_sparse_training/Wan2.2-TI2V-5B.sh`
- `docs_SparseAttn/plot_training_log.py`

### 5.2 对外配置

建议 CLI 参数:

```text
--enable_block_sparse_attn
--block_sparse_sparsity 0.90
--block_sparse_block_size 2,8,8
--block_sparse_q_chunk_blocks 16
--block_sparse_dense_fallback_threshold 0
--block_sparse_debug
--block_sparse_debug_layer 0
--block_sparse_debug_head 0
--block_sparse_debug_output ./debug/block_sparse
```

说明:

- `sparsity=0.90` 表示每个 query block/head 保留约 10% 的有效 KV blocks。
- `dense_fallback_threshold=0` 表示禁用 dense fallback。设为正数时, 小 token 数可以回退到 dense attention。
- 分布式训练时 debug 文件仅由 main process 写出。

### 5.3 训练 Infra 和 OOM 控制方案

训练侧 infra 的目标是“先能稳定恢复训练, 再逐步提高稀疏率和分辨率”。已实现的优化点如下:

- **Chunked exact sparse attention**: token-level exact attention 按 query block chunk 计算, 由 `--block_sparse_q_chunk_blocks` 控制。训练默认建议 `2`, B200 显存充足时可尝试 `4/8` 提高吞吐。
- **Gradient checkpointing 兼容**: Wan 训练入口保留现有 gradient checkpointing, sparse attention 不引入新参数, 只替换 self-attn forward 的计算路径。
- **Cosine sparsity ramp**: 通过 `--block_sparse_start_sparsity`, `--block_sparse_end_sparsity`, `--block_sparse_ramp_steps` 从较低稀疏率逐步升到目标稀疏率, 避免恢复训练初期路由过硬导致 loss 波动。
- **Dense fallback 开关**: `--block_sparse_dense_fallback_threshold` 可让小 token 数样本回退 dense attention。默认 `0` 禁用, 因为 Wan2.2-TI2V-5B 的目标场景通常 token 数较大。
- **Debug 默认关闭**: `--block_sparse_debug` 默认关闭, 避免训练时保存大量 mask 文件或持有额外 CPU tensor。只在小规模 smoke test 中打开。
- **Step-level JSONL 日志**: 每 step 写 `training_log.jsonl`, 包含 `loss/lr/seconds_per_step/samples_per_second/frames_per_second/sparsity/block_size/mask_density/k_keep`, 便于发现 OOM 前的吞吐退化和稀疏率变化。
- **无依赖 SVG 可视化**: `docs_SparseAttn/plot_training_log.py` 不依赖 matplotlib, 在内网机器上可直接把 JSONL 画成 loss、吞吐和 sparsity 曲线。
- **`max_train_steps` smoke test**: 新增 `--max_train_steps`, 支持先跑 1-3 step 验证显存、日志和 checkpoint 逻辑, 再开始长训。

建议训练参数起点:

```text
--height 320
--width 576
--num_frames 17
--use_gradient_checkpointing
--enable_block_sparse_attn
--block_sparse_block_size "2,8,8"
--block_sparse_start_sparsity 0.85
--block_sparse_end_sparsity 0.90
--block_sparse_ramp_steps 100
--block_sparse_q_chunk_blocks 2
```

如果 B200 上显存余量明显充足, 优先提高 `--block_sparse_q_chunk_blocks` 来提升吞吐; 如果 OOM, 优先降低 `q_chunk_blocks`, 再降低分辨率或 `num_frames`。

## 6. 逻辑测试

逻辑测试不依赖完整 Wan 模型, 只验证 block sparse 核心函数。

### 6.1 Shape 和 Partition Round Trip

输入:

```text
B=1, f=3, h=5, w=7, H=2, Dh=4
block_size=(2,3,4)
```

预期:

- padded grid 为 `(4,6,8)`。
- block 数为 `(4/2)*(6/3)*(8/4)=8`。
- partition -> reverse partition -> crop 后恢复原 tensor。
- token 坐标映射满足:

```text
token_id = ((t*h)+y)*w+x
```

验收:

```text
max_abs(original - recovered) == 0
```

### 6.2 Padding Mask 不污染 Block Mean

输入:

```text
f=3, h=5, w=7, block_size=(2,3,4)
```

预期:

- `valid_token.sum() == f*h*w == 105`。
- padded token 数为 `4*6*8 - 105 = 87`。
- 所有 padded token 的 valid mask 为 `False`。
- 边界 block 内真实 token 和 padded token 混合时, masked mean 只统计真实 token。

验收:

```text
masked_mean_uses_only_real_tokens == True
padded_token_contribution_to_repr == 0
```

### 6.3 Top-k 数量测试

输入:

```text
N_valid = 20
sparsity = 0.90
```

预期:

```text
K_keep = max(1, round(20 * 0.10)) = 2
```

补充 case:

```text
N_valid=1, sparsity=0.95 -> K_keep=1
N_valid=10, sparsity=0.00 -> K_keep=10
N_valid=10, sparsity=0.99 -> K_keep=1
```

### 6.4 Dense 等价性测试

设置:

```text
sparsity = 0.0
```

预期:

- 所有有效 KV blocks 都被选中。
- sparse 输出应与 PyTorch dense SDPA 在有效 token 上数值接近。

容差:

```text
float32: atol <= 1e-5, rtol <= 1e-4
bfloat16: atol <= 5e-2, rtol <= 5e-2
```

### 6.5 Gradient 测试

使用小张量:

```text
B=1, f=2, h=4, w=4, H=2, Dh=8
block_size=(1,2,2)
sparsity=0.5
```

预期:

- forward 输出无 NaN/Inf。
- backward 成功。
- Q/K/V 的 selected path 有梯度。
- top-k index 本身不要求可微。

### 6.6 人工无效 Block 测试

标准 ceil padding 通常不会产生整块全 padding 的 block。为了验证 `M_block` 分支, 单测中需要手工构造一个 `M_blk` 全 False 的 KV block。

预期:

- 即使该无效 KV block 的 raw score 被人为设为最大, top-k 结果也不能选中它。
- `invalid_kv_selected == 0`。

## 7. Attn Mask 小范围测试

这组测试必须直接检查 block-level attn mask 和 block 坐标。

### 7.1 坐标布局测试

使用:

```text
f=2, h=4, w=4
block_size=(1,2,2)
```

预期 block 坐标:

```text
block 0: t[0:1), h[0:2), w[0:2)
block 1: t[0:1), h[0:2), w[2:4)
block 2: t[0:1), h[2:4), w[0:2)
block 3: t[0:1), h[2:4), w[2:4)
block 4: t[1:2), h[0:2), w[0:2)
block 5: t[1:2), h[0:2), w[2:4)
block 6: t[1:2), h[2:4), w[0:2)
block 7: t[1:2), h[2:4), w[2:4)
```

验收:

- 打印出来的 block range 与上表一致。
- 每个 block 内 token 在 `(t,y,x)` 上相邻。
- block 顺序与 Wan RoPE flatten 顺序一致。

### 7.2 Padding Exclusion 测试

使用:

```text
f=3, h=5, w=5
block_size=(2,3,3)
padded=(4,6,6)
```

预期:

- 8 个 block 都至少包含一部分真实 token。
- 边界 block 内存在 padded token。
- padded token 不参与 masked mean。
- selected KV block 内的 padded token 在 exact attention logits 中被 mask 为 `-inf`。

需要打印:

```text
grid=(3,5,5), padded=(4,6,6), block_size=(2,3,3)
num_blocks=8
num_valid_tokens=75
num_padded_tokens=69
boundary_blocks_with_padding=<nonzero>
mask_density=<value>
padded_token_logits_masked=True
```

### 7.3 Deterministic Top-k 测试

构造 synthetic block representatives:

```text
Q_repr[query_block_i] = one_hot(i)
K_repr[key_block_j]   = one_hot(j)
```

预期:

- 当 `K_keep=1` 时, query block `i` 选择 key block `i`。
- 如果手工将 key block `i` 标为无效, 则 query block `i` 必须选择下一个有效高分 block。

### 7.4 Mask 文件保存和可视化

对指定 layer/head 保存:

```text
block_mask_layer{L}_head{H}.pt
block_mask_layer{L}_head{H}.txt
selected_blocks_layer{L}_head{H}.txt
```

验收:

- heatmap 尺寸为 `N_query_blocks x N_kv_blocks`。
- 无效 KV block 列没有被选中。
- 前若干 query block 的 selected KV 坐标可读, 能人工确认时空邻近关系。

## 8. 稀疏直推测试

### 8.1 小规模 Smoke Inference

使用较小配置跑 Wan sparse inference:

```text
height=256 or 320
width=448 or 512
num_frames=17
num_inference_steps=2 to 4
seed=0
sparsity=0.90
block_size=(2,8,8)
```

预期:

- pipeline 完成。
- model output latents 无 NaN/Inf。
- 输出视频文件生成。
- 指定 layer/head 的 debug mask summary 生成。

### 8.2 Dense vs Sparse 性能对比

同 prompt / seed 分别运行:

```text
dense baseline
sparse sparsity=0.50
sparse sparsity=0.90
```

记录:

- peak CUDA memory。
- 每个 denoising step 的 wall-clock time。
- 每层 selected KV block 数。
- 有效 block mask density。

预期:

- 大 token 数下 sparse 显存降低。
- 小 grid 下 sparse 可能因为 gather/top-k overhead 慢于 dense, 这是可接受结果, 用于决定 dense fallback threshold。

### 8.3 非整除 Grid Inference

选择能产生非整除 latent grid 的尺寸, 使 padding path 被实际触发。

预期:

- padding path 被触发。
- `padded_token_logits_masked=True`。
- final output crop 回原始 latent grid。
- 视频生成完成。

### 8.4 可复现性测试

相同 seed 和 sparse config 跑两次。

预期:

- deterministic 设置下 latents 一致或数值近似一致。
- debug masks 一致。

## 9. 训练测试

### 9.1 单步训练 Smoke Test

基于现有 Wan2.2-TI2V-5B 训练示例, 使用 tiny dataset 或 cached sample:

```text
height=256 or 320
width=448 or 512
num_frames=17
max_train_steps=1 or one epoch with one repeated item
sparsity=0.90
block_size=(2,8,8)
```

预期:

- forward/backward 成功。
- optimizer step 成功。
- loss finite。
- checkpoint save 仍然可用。

### 9.2 Loss 和吞吐量日志测试

训练 logger 每 step 写一行 JSONL 或 CSV:

```json
{
  "step": 1,
  "epoch": 0,
  "loss": 0.1234,
  "lr": 1e-5,
  "seconds_per_step": 3.21,
  "samples_per_second": 0.31,
  "frames_per_second": 5.27,
  "sparsity": 0.9,
  "block_size": "2,8,8",
  "mask_density": 0.1
}
```

预期:

- accelerate 分布式下仅 main process 写 log。
- loss 字段存在且 finite。
- 吞吐量字段为正数。
- sparse config 字段存在。

### 9.3 训练 Log 可视化测试

输入:

```text
training_log.jsonl
```

输出:

```text
loss_curve.png or loss_curve.svg
throughput_curve.png or throughput_curve.svg
```

预期:

- loss 曲线可渲染。
- `seconds_per_step` 和 `frames_per_second` 曲线可渲染。
- 可选字段缺失时脚本能给出清晰提示, 不静默失败。

### 9.4 Sparse Recovery 测试

短流程:

1. dense 训练若干 step 并保存 checkpoint。
2. 从 checkpoint resume, 同时启用 sparse attention。
3. 再训练若干 step。

预期:

- checkpoint 加载无 unexpected keys。
- sparse attention 不引入新 trainable parameters。
- resume 后 loss 保持 finite。
- 训练吞吐量 log 继续写入, step 计数清晰。

### 9.5 稀疏率 Schedule 可选测试

如果后续加入 ramp schedule:

```text
step 0: sparsity=0.50
step N: sparsity=0.90
```

预期:

- log 中的 sparsity 与 schedule 一致。
- `K_keep` 随 sparsity 增大而降低。
- schedule 变化过程中 padded token 仍不影响 top-k 和 exact attention。

## 10. Debug 和验收输出

启用 `--block_sparse_debug` 时输出:

```text
debug/block_sparse/
  block_layout.txt
  mask_summary.json
  mask_layer{L}_head{H}.pt
  mask_layer{L}_head{H}.txt
  selected_blocks_layer{L}_head{H}.txt
```

`mask_summary.json` 至少包含:

```json
{
  "grid": [3, 5, 5],
  "padded_grid": [4, 6, 6],
  "block_size": [2, 3, 3],
  "num_blocks": 8,
  "num_valid_tokens": 75,
  "num_padded_tokens": 69,
  "boundary_blocks_with_padding": 7,
  "sparsity": 0.9,
  "k_keep": 1,
  "mask_density": 0.125,
  "invalid_kv_selected": 0,
  "padded_token_logits_masked": true
}
```

## 11. 验收标准

实现只有在以下项目全部通过后才算完成:

- divisible 和 non-divisible grid 的 partition round trip 测试通过。
- padding mask 测试证明 padded token 不进入 block mean。
- 人工无效 block 测试证明无效 KV block 不会进入 top-k。
- `sparsity=0.0` 时 dense-equivalence 测试通过。
- gradient smoke test 通过。
- attn mask 小范围坐标打印与 Wan RoPE 顺序一致。
- Wan2.2-TI2V-5B sparse inference 完成并生成 debug mask。
- 非整除 grid inference 完成, 且 `padded_token_logits_masked=True`。
- sparse training 单步 smoke test 完成。
- 训练 log 包含 finite loss 和正吞吐量。
- 可视化脚本能从训练 log 渲染 loss 和 throughput 曲线。

## 12. 已知风险

- 纯 PyTorch gather sparse attention 在小分辨率下可能慢于 dense attention。
- top-k routing 不可微。为了效率这可以接受, 但训练恢复建议先从较低稀疏率或较大 `K_keep` 开始。
- `K_keep` 太小可能影响质量, 后续可能需要加入 self-block 保留或 minimum keep count。
- 如果未来 Wan 变体改变 patchify flatten 顺序, block 坐标映射必须重新验证。
- Unified sequence parallel 会重写 self-attn forward, 初版 sparse attention 应先视为与 USP 不兼容, 除非单独适配。

## 13. 开发顺序

1. 实现 3D block partition, reverse partition 和 valid masks。
2. 增加 partition, padding, top-k, dense equivalence, gradient 逻辑测试。
3. 增加 debug mask 保存和小范围 mask 测试。
4. 以 opt-in config 接入 Wan self-attention。
5. 增加 inference enablement 并运行 smoke inference。
6. 增加 training CLI flags。
7. 增加训练 loss 和吞吐量日志。
8. 增加训练 log 可视化脚本。
9. 运行 sparse training smoke test 和 sparse recovery test。
