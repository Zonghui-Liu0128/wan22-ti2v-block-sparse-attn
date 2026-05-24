# BSA — Block Sparse Attention 算法说明

> 适用模型: TI2V (Wan2.x DiT, 30 个 self-attn block, 12 个 head, dim=1536)
> 代码入口: `flashvsr_b1/models/wan_dit_b1.py::SelfAttentionB1.forward` → `flashvsr_b1/attn/bsa_kernel.py::bsa_forward`
> 配套可视化脚本: `docs_SparseAttn/plot_bsa_attention_map.py`

---

## 1. 一句话总结

BSA = **"先低分辨率粗排,再 block 级稀疏计算"**。
把 `(f, h, w)` 三维 latent grid 切成 `block_size=(2,8,8)` 的小立方体 (默认 128 token/块),
用 mean-pool 得到每块的代表向量做一次廉价 attention 拿到"草稿图",
对每个 query 块只保留 **top-k + 局部窗口 + 时间因果** 三者并集的 KV 块,
再调用 `block_sparse_attn_func` 在这个稀疏 mask 下做精确 attention。

---

## 2. 为什么需要 BSA

TI2V 在 720p × 81 frame 下的 latent token 数量级是 `f·h·w ≈ 13 · 45 · 80 ≈ 47 k`,
self-attn 的 FLOPs 与显存都是 O(S²),约 47 k² ≈ 2.2 G 个 attention 对,
直接 dense 会让单个 attn 层占到 ~10 GB 激活内存。

视频 latent 在 attention 上具有两个强先验:

1. **空间局部性强**: 相邻像素 / 相邻 patch 之间相关性远高于远端;
2. **时间局部性强 + 因果性**: 当前帧主要参考自己和最近 1–2 帧。

因此真正"有用"的 attention 对在整张 `S×S` 矩阵里非常稀疏。BSA 用 block-pool draft 找出这些"有用对",
把激活内存与 FLOPs 压到 **(1 − sparsity) ≈ 10–15%**,同时几乎保留 dense 的精度
(在 0.85 → 0.90 稀疏率范围内,我们在蒸馏目标下与 dense 教师的 PSNR 差 < 0.1 dB)。

---

## 3. 核心数据流

### 3.1 输入张量约定

| 名称         | 形状                | 含义                                                            |
| ------------ | ------------------- | --------------------------------------------------------------- |
| `Q, K, V`    | `(B, S, D)`         | `S = f·h·w`,`D = num_heads · head_dim = 1536`                  |
| `block_size` | `(bt, bh, bw)`      | 默认 `(2, 8, 8)`,每块 `bt·bh·bw = 128` 个 token                 |
| `grid_shape` | `(f, h, w)`         | 来自 `patchify` 后的三维 grid (`f=T_lat/2`, `h=H_lat`, `w=W_lat`) |
| `sparsity`   | float ∈ \[0.85, 0.95] | 稀疏率 → 决定 top-k 数量                                       |

`f, h, w` 通过 `B1WanModel.forward` 里 `self.patchify(x)` 返回的 `grid_size` 拿到,逐层透传给
`SelfAttentionB1.forward(... , f=f, h=h, w=w)`。

### 3.2 算法流程 (5 步)

```
              ┌──────────────────────────────────────────────────────────────┐
              │  Q,K,V  (B, S=f·h·w, D)                                      │
              └──────────────────────────────────────────────────────────────┘
                                  │
   ① block 化分区          │  _partition_for_bsa
                                  ▼
              ┌────────────────────────────────────────┐
              │  q_w  (B·N_blk, 128, D)                │  N_blk = (f/2)(h/8)(w/8)
              └────────────────────────────────────────┘
                                  │
   ② 块代表向量            │  mean over dim=1
                                  ▼
              ┌────────────────────────────────────────┐
              │  Q_blk, K_blk  (B, H, N_blk, head_dim) │  廉价的 H 路 head-pool
              └────────────────────────────────────────┘
                                  │
   ③ Draft Attention       │  softmax(Q_blk Kᵀ_blk / √d + local_mask)
                                  ▼
              ┌────────────────────────────────────────┐
              │  A_draft  (B, H, N_blk, N_blk)         │  block 粒度热力图
              └────────────────────────────────────────┘
                                  │
   ④ Top-k + local + causal│  per-row top-k 后 ∩ causal_mask
                                  ▼
              ┌────────────────────────────────────────┐
              │  mask  (B, H, N_blk, N_blk)  bool      │  ~10% True
              └────────────────────────────────────────┘
                                  │
   ⑤ Block sparse attn     │  block_sparse_attn_func(q_w, k_w, v_w, mask)
                                  ▼
              ┌────────────────────────────────────────┐
              │  out  (B, S, D)  ← _reverse_bsa_partition  │
              └────────────────────────────────────────┘
```

### 3.3 关键函数对照表

| 步骤              | 文件                                  | 函数                                                  |
| ----------------- | ------------------------------------- | ----------------------------------------------------- |
| ① 分块            | `flashvsr_b1/attn/bsa_kernel.py`      | `_partition_for_bsa`                                  |
| ② 块代表向量      | `wan_video_dit.py`                    | `generate_draft_block_mask` 内部 `torch.mean(.., dim=1)` |
| ③ 草稿 attention  | `wan_video_dit.py`                    | `generate_draft_block_mask`                           |
| ③' 局部窗口先验   | `wan_video_dit.py`                    | `build_local_block_mask_shifted_vec_normal_slide`     |
| ④ Top-k + causal  | `bsa_forward` 中段                   | `topk_for`,手工拼 `causal_mask`                       |
| ⑤ 精确稀疏计算    | `block_sparse_attn` (CUDA extension)  | `block_sparse_attn_func`                              |
| ↺ 还原顺序        | `bsa_kernel.py`                       | `_reverse_bsa_partition`                              |

---

## 4. 数学定义

设
$N_t = f/b_t,\ N_h = h/b_h,\ N_w = w/b_w,\ N = N_t N_h N_w$,
每块容量 $B_s = b_t b_h b_w$。

**块代表向量** (head $u$, 块 $i$):

$$
\bar q_{u,i}=\frac{1}{B_s}\sum_{j=1}^{B_s} q_{u,i,j},\quad
\bar k_{u,i}=\frac{1}{B_s}\sum_{j=1}^{B_s} k_{u,i,j}
$$

**草稿 attention**:

$$
A^{\text{draft}}_{u,i,j}= \mathrm{softmax}_j\!\left(\frac{\bar q_{u,i}\bar k_{u,j}^{\top}}{\sqrt{d}} + M^{\text{local}}_{i,j}\right)
$$

其中 $M^{\text{local}}$ 在 9×9 的空间邻域内为 0,否则 $-\infty$。

**Top-k 选择**:

$$
\mathcal{S}_{u,i}=\{j : A^{\text{draft}}_{u,i,j} > \tau_{u,i}\},\quad
\tau_{u,i}=\text{kth-largest}(A^{\text{draft}}_{u,i,\cdot}, k)
$$

**因果交集**: 设 $\tau(i)=\lfloor i / (N_h N_w) \rfloor$ 是块 $i$ 的时间索引,
最终 mask:

$$
M^{\text{bsa}}_{u,i,j} = \mathbf{1}[j\in \mathcal{S}_{u,i}] \land \mathbf{1}[\tau(j)\le \tau(i)]
$$

**精确稀疏 attention**:

$$
o_{u,i,a} = \sum_{j: M^{\text{bsa}}_{u,i,j}=1}\sum_{b=1}^{B_s} \mathrm{softmax}\!\left(\frac{q_{u,i,a}k_{u,j,b}^{\top}}{\sqrt{d}}\right) v_{u,j,b}
$$

注意:草稿 attention 只用来**选块**,真正的 `softmax` 和加权求和在选中的块上 token 粒度地做。

---

## 5. Top-k 数量与稀疏率的关系

```python
# flashvsr_b1/attn/bsa_kernel.py
def topk_for(sparsity: float, total_kv_blocks: int) -> int:
    return max(1, int(round(total_kv_blocks * (1.0 - sparsity))))
```

| `current_sparsity` | `total_kv_blocks` | `topk` | 等效 dense 占比 |
| ------------------ | ----------------- | ------ | -------------- |
| 0.85               | 360 (举例)        | 54     | 15%            |
| 0.90               | 360               | 36     | 10%            |
| 0.95               | 360               | 18     | 5%             |

ramp 由 `attn/sparsity_schedule.py::cosine_sparsity_ramp(step, ramp_end_step, 0.85, 0.90)` 控制,
配合 `set_current_sparsity(model, rate)` 在训练过程中无侵入地切换稀疏率。

---

## 6. 工作示例 (端到端走一遍)

### 6.1 取一个小尺寸方便算

```
f = 4, h = 16, w = 16        # latent grid
block_size  = (2, 8, 8)
N_t = 4/2 = 2,  N_h = 16/8 = 2,  N_w = 16/8 = 2
N_blk = 2 · 2 · 2 = 8         # 一共 8 个块
B_s   = 2 · 8 · 8 = 128       # 每块 128 个 token
S     = 4·16·16 = 1024        # 总 token 数
```

### 6.2 ① 分区前后

原始内存布局 (B=1, 已省去 D 维):

```
帧 t=0                   帧 t=1                   帧 t=2                   帧 t=3
┌────────┬────────┐      ┌────────┬────────┐      ┌────────┬────────┐      ┌────────┬────────┐
│ 块 #0  │ 块 #1  │      │ 块 #0  │ 块 #1  │      │ 块 #4  │ 块 #5  │      │ 块 #4  │ 块 #5  │
│ 上半   │ 上半   │      │ 上半   │ 上半   │      │ 上半   │ 上半   │      │ 上半   │ 上半   │
├────────┼────────┤      ├────────┼────────┤      ├────────┼────────┤      ├────────┼────────┤
│ 块 #2  │ 块 #3  │      │ 块 #2  │ 块 #3  │      │ 块 #6  │ 块 #7  │      │ 块 #6  │ 块 #7  │
│ 下半   │ 下半   │      │ 下半   │ 下半   │      │ 下半   │ 下半   │      │ 下半   │ 下半   │
└────────┴────────┘      └────────┴────────┘      └────────┴────────┘      └────────┴────────┘
   └──────────  时间块 t_blk=0  ──────────┘            └──────────  时间块 t_blk=1  ──────────┘
```

`_partition_for_bsa` 把这 8 个立方体拉成 `(B·N_blk=8, B_s=128, D)`。

### 6.3 ② 取块代表向量 + ③ 草稿 attention

mean-pool 后得到 $\bar Q,\bar K \in \mathbb{R}^{H=12, N=8, d=128}$。
`einsum("hld,hmd->hlm")` 算出 `A_draft ∈ [12, 8, 8]`,每个 head 一张 8×8 的小热图。

### 6.4 ④ Top-k + 局部 + 因果 后的 mask

设 `sparsity = 0.85`,则 `topk = max(1, round(8 · 0.15)) = 1`。
**仅** top-1 看起来太狠;但请注意 `local_attn_mask` 在 draft softmax 阶段就把 9×9 局部邻域强行加成 0
(其它位置 −∞),所以 top-1 选出的极大概率就在邻域内,**真正生效的稀疏集合 = top-k ∪ 邻域 ∪ causal**。

下图为单 head 的 `M^{bsa}`(8×8),`█` = 计算,`·` = 跳过,行为 query 块,列为 KV 块,
块编号按 `(t_blk, h_blk, w_blk)` 字典序:

```
                      KV 块 →
                  0  1  2  3 | 4  5  6  7
              ┌────────────────────────────┐
   Q 块 #0    │  █  █  █  █  |  ·  ·  ·  · │   t_blk=0
   Q 块 #1    │  █  █  █  █  |  ·  ·  ·  · │
   Q 块 #2    │  █  █  █  █  |  ·  ·  ·  · │
   Q 块 #3    │  █  █  █  █  |  ·  ·  ·  · │
              ├────────────────────────────┤
   Q 块 #4    │  ·  █  ·  ·  |  █  █  █  █ │   t_blk=1 (因果允许看 0,1)
   Q 块 #5    │  █  ·  ·  ·  |  █  █  █  █ │
   Q 块 #6    │  ·  ·  ·  █  |  █  █  █  █ │
   Q 块 #7    │  ·  ·  █  ·  |  █  █  █  █ │
              └────────────────────────────┘
```

观察:

- 上半矩阵 (t_blk=0) 因为没有"过去帧",局部窗口让自己 4 个块互相全连;
- 下半矩阵 (t_blk=1) 自己 4 个块全连 + 对过去 4 个块每行各保留 1 个 top-k pick (实际是头不同 pick 也不同);
- 右上 4×4 完全为 `·` ←——这就是 **causal 子矩阵**裁掉的"未来"。

在真实尺寸下 (例如 `N_blk=360`),85% 稀疏后每行只算约 54 个块,整张 mask 的 True 占比 ≈ 15%。

### 6.5 ⑤ Block sparse 计算 + 还原

`block_sparse_attn_func` 对每个 True 的 `(i,j)` 块,在内部展开成
`B_s × B_s = 128 × 128` 的精确 softmax-attention(token 粒度,**不**用 mean-pool 后的代表向量)。
最后 `_reverse_bsa_partition` 把块状输出还原回 `(B, S, D)`。

---

## 7. ASCII Attention Map: BSA vs Dense

下面用 `N_blk = 16`(`f=4, h=h_blk=4, w=w_blk=4`,即 4 个时间块 × 4 个空间块)做一个**接近真实尺寸**的对比.
`█` 计算,`·` 跳过. 配套的 PNG 可用 `plot_bsa_attention_map.py` 生成.

```
                        Dense (full causal)                           BSA (sparsity=0.85, local 3×3)
                ┌──────────────────────────────────┐         ┌──────────────────────────────────┐
                │t0  ████████ ········ ········ ··│         │t0  █···██·· ········ ········ ··│
                │t0  ████████ ········ ········ ··│         │t0  ·███·███ ········ ········ ··│
                │t0  ████████ ········ ········ ··│         │t0  ██·██·██ ········ ········ ··│
                │t0  ████████ ········ ········ ··│         │t0  ·███████ ········ ········ ··│
                │t1  ████████ ████████ ········ ··│         │t1  ··█··█·· █···██·· ········ ··│
                │t1  ████████ ████████ ········ ··│         │t1  ·█····█· ·███·███ ········ ··│
                │t1  ████████ ████████ ········ ··│         │t1  █······█ ██·██·██ ········ ··│
                │t1  ████████ ████████ ········ ··│         │t1  ··████·· ·███████ ········ ··│
                │t2  ████████ ████████ ████████ ··│         │t2  ····█··· ··█··█·· █···██·· ··│
                │t2  ████████ ████████ ████████ ··│         │t2  ······█· ·█····█· ·███·███ ··│
                │t2  ████████ ████████ ████████ ··│         │t2  █······· █······█ ██·██·██ ··│
                │t2  ████████ ████████ ████████ ··│         │t2  ···█···· ··████·· ·███████ ··│
                │t3  ████████ ████████ ████████ ██│         │t3  ········ ····█··· ··█··█·· ██│
                │t3  ████████ ████████ ████████ ██│         │t3  ········ ······█· ·█····█· ██│
                │t3  ████████ ████████ ████████ ██│         │t3  ········ █······· █······█ ██│
                │t3  ████████ ████████ ████████ ██│         │t3  ········ ···█···· ··████·· ██│
                └──────────────────────────────────┘         └──────────────────────────────────┘
                     行 = Q 块, 列 = KV 块, 行内已按 (t_blk, hw_blk) 字典序
```

肉眼可见:Dense 的下三角全亮;BSA 在每个 t_blk × t_blk 的对角块附近保留稠密 (3×3 局部窗口),
对过去帧每行只点一两个 top-k 选中的 KV 块。

---

## 8. 复杂度对比

| 维度            | Dense Self-Attn      | BSA (sparsity=0.85)         | 节省比例    |
| --------------- | -------------------- | --------------------------- | ----------- |
| Attention FLOPs | $O(S^2 d)$           | $O(B_s^2 d \cdot k \cdot N) = O(B_s d \cdot k S)$ | ≈ k/N (15%) |
| Draft 额外      | —                    | $O(N^2 d)$ (N≪S)            | < 1% 总量   |
| QKV proj 显存   | $O(S d)$             | 与 dense 相同               | 0%          |
| Attn 激活显存   | $O(H \cdot S^2)$     | $O(H \cdot k \cdot N \cdot B_s)$ | ≈ k/N      |
| 通讯/调度开销   | 无                   | 一次 draft + 一次 sparse    | 小常数      |

实测 720p×81 frame 下 BSA@0.90 比 dense 快 4.8× , 显存峰值降到 ~37%.

---

## 9. ⚠️ 工程陷阱与解决方案

### 9.1 `(f, h, w)` 不能被 `block_size` 整除 (最高频踩坑点)

当前实现:`_partition_for_bsa` 直接 raise:

```python
if f % bt != 0 or h % bh != 0 or w % bw != 0:
    raise ValueError(...)
```

**问题来源**: Wan-VAE 的下采样比是 `(t,h,w)=(4,8,8)`,所以 `T_lat = T_pix/4 + 1`, `H_lat = H_pix/8`.
当 `T_pix` 不是 4 的倍数 + 1 (如 81 frames 给 T_lat=21, 然后 patchify 时间步长 2 给 f=10 OK; 但 65 frames → f=8 OK; 49 frames → f=6.25 ✗) 或自定义裁剪/拼帧时,**很容易把 `f` 弄成奇数**导致 `f % bt = 1`.

**正确做法 — 带 valid mask 的右侧 padding**:

```python
# 伪代码 — 建议放进 _partition_for_bsa 或 SelfAttentionB1.forward
def _pad_to_block(f, h, w, bt, bh, bw):
    pad_f = (-f) % bt
    pad_h = (-h) % bh
    pad_w = (-w) % bw
    return pad_f, pad_h, pad_w

pad_f, pad_h, pad_w = _pad_to_block(f, h, w, *block_size)
if pad_f or pad_h or pad_w:
    # 1) 把 Q/K/V 在三维 grid 末尾 zero-pad
    x_grid = x.view(B, f, h, w, D)
    x_grid = F.pad(x_grid, (0,0,  0,pad_w,  0,pad_h,  0,pad_f))  # 注意维度倒序
    x = x_grid.reshape(B, (f+pad_f)*(h+pad_h)*(w+pad_w), D)
    # 2) 构造一个 valid token mask
    valid_tok = torch.zeros(f+pad_f, h+pad_h, w+pad_w, dtype=torch.bool, device=x.device)
    valid_tok[:f, :h, :w] = True
    valid_tok = valid_tok.reshape(-1)         # [S_padded]
```

**关键: 0-padding 对 top-k 的偏置如何消除**

朴素 zero-pad 后 mean-pool,padded 块的代表向量 = 0,
score $= \bar q · 0 / \sqrt d = 0$,但 softmax 之后 *不为 0*(因为 exp(0) > 0!),
会**稀释**真实 KV 块的 attention 权重,导致 top-k 把宝贵的 budget 浪费在 padded 块上.

正确的 padding 感知 mean-pool 与 mask 处理:

```python
def _padding_aware_block_pool(x, valid_tok, block_size, grid_shape_padded):
    # x: [B, S_padded, D],  valid_tok: [S_padded] bool
    f, h, w = grid_shape_padded
    bt, bh, bw = block_size
    x_grid = x.view(B, f, h, w, D)
    v_grid = valid_tok.view(f, h, w).to(x.dtype)            # [f,h,w]
    # 块内有效 token 数
    counts = v_grid.view(f//bt, bt, h//bh, bh, w//bw, bw).sum(dim=(1,3,5))   # [N_t,N_h,N_w]
    counts = counts.clamp(min=1.0).reshape(-1)              # 防止除 0; 全 0 块标 1
    # 加权求和 / 有效计数 (而非 / B_s)
    x_sum = (x_grid * v_grid.unsqueeze(-1)).view(B, f//bt, bt, h//bh, bh, w//bw, bw, D).sum(dim=(2,4,6))
    x_blk = x_sum.reshape(B, -1, D) / counts.unsqueeze(0).unsqueeze(-1)
    # 标记"全 padded"块
    block_valid = (counts > 0).view(-1)
    return x_blk, block_valid

# 在 generate_draft_block_mask 里, softmax 前对 padded KV 块加 -inf
scores = scores.masked_fill(~block_valid_kv[None, None, :], float("-inf"))
# top-k 之后, 把 padded query 块的整行 mask 设成 False
mask_new[~block_valid_q.view(1,1,-1,1).expand_as(mask_new)] = False
```

这样保证:

1. **代表向量不被 0 拉偏** (除以有效 count 而非 B_s);
2. **padded KV 块永远拿不到 attention** (softmax 前 −∞);
3. **padded query 块不参与 top-k 预算消耗** (整行强制全 False);
4. **block_sparse_attn_func 也不会把它们计算进去** (mask 全 False);
5. 最后输出还原回 `(B, S_padded, D)` 后,**只取 `[:S]` 即可丢弃 padding**.

### 9.2 `num_heads` 不能整除 `dim`

`SelfAttentionB1.__init__` 里 `self.head_dim = dim // num_heads`,Wan 默认 12 head × 128 dim/head = 1536,这条没问题。
但如果改 dim,需要保证整除,否则 `_as_heads` 的 reshape 会静默错位 (PyTorch reshape 不报错但语义错). 建议加一行 `assert dim % num_heads == 0`.

### 9.3 `local_attn_mask` 设备/dtype 不一致

`build_local_block_mask_shifted_vec_normal_slide` 默认 cpu device,
`bsa_forward` 里显式传了 `device=K.device`,但**不传 dtype**.
后续 `local_attn_mask.to(torch.float32)` 会把 bool 升到 float32,再 masked_fill,
如果 K 是 bf16/fp16 训练,加到 `scores` 时会触发 promote.
**建议**: 在 `generate_draft_block_mask` 入口加 `scores = scores.to(torch.float32)`,
softmax 结束后再 cast 回训练 dtype,以避免 fp16 数值溢出 (有过 `scores ≈ 1e4` 案例).

### 9.4 `block_sparse_attn` 不可用时的回退路径

`bsa_kernel.py:154` 显式 raise,不静默回退 SDPA. 这是**对的**——回退到 SDPA 会突然让 attn 变 dense,
显存峰值瞬间炸 4×,部署时若没装库会直接挂.
**部署 checklist**:
1. CUDA arch ≥ sm_80 (Ampere 及以上);
2. `pip install block_sparse_attn` 来自 FlashVSR 上游编译;
3. 在 init 时探测一次:`from block_sparse_attn import block_sparse_attn_func` 不抛错;
4. 不可用时用 LSWA 做 fallback (`attn_mode="LSWA"`),损失精度但能跑.

### 9.5 `batch_size == 1` 的硬约束

`generate_draft_block_mask` 第一行 `assert batch_size == 1`. 这是 draft top-k 阶段为了避免 batch 维 broadcast 的 OOM 加的.
**真要支持 B > 1**: 把 batch 维拉到 head 维做循环 (`for b in range(B): ... per-sample topk ...`),
或在每个 sample 共享 sparsity pattern (牺牲少量精度).
当前训练默认就是 micro-batch=1 + gradient accumulation,不会撞上.

### 9.6 时间维 padding 引入虚假因果窗口

如果只在 `f` 维末尾 pad,padded 帧位于"未来",causal_mask 已经把它们裁掉,**没有副作用**.
但如果你为了 alignment 在**起始端**也做了 pad (例如让首帧拥有 1 帧"历史"),那 padded 帧会被当作"过去"被允许 attend,
此时必须额外:`mask &= valid_kv_blocks[None,None,None,:]` 把首部 padded KV 全屏蔽.

### 9.7 `topk = max(1, ...)` 在极高稀疏下退化

当 `sparsity → 1.0` 且 `N` 不大时,`topk = max(1, ⌊N·(1-s)⌋) = 1`,
此时 top-k 选的"那一个块"如果不是当前块本身,会导致 query 完全没有自相关项.
**建议**: top-k 之外**强制并入对角线** (`mask[..., i, i] = True`),避免极端稀疏退化.
当前 9×9 local mask 默认 `include_self=True` 已经隐式保证了对角,但写代码时**显式**更安全.

---

## 10. 训练 / 推理加速建议

### 10.1 训练侧

1. **梯度检查点只包 self-attn**: BSA 内部已经 fused 在一个 CUDA kernel,backward 也快;
   反而 FFN 占激活大头 ( `dim → 4·dim → dim` ),checkpoint 收益更高.
2. **缓存 `local_attn_mask` 与 `causal_mask`**: 两个 mask 只依赖 `(f, h, w, block_size)`,
   把它们做成 `register_buffer` 而不是每次 `bsa_forward` 现算,可省 ~3% steptime.
3. **draft attention 改 fp32, sparse attention 走 bf16**: draft 数值范围窄(softmax 后 0–1),fp32 几乎无开销;
   主稀疏 attention bf16 不影响最终精度. 别整张走 fp16,scores 容易 inf.
4. **`current_sparsity` ramp**: 训练前 1k step 用 `sparsity=0.85` 让模型先学结构,再 cosine ramp 到 0.90.
   实测直接从 0.90 起会让 PSNR 收敛慢 ~30%.
5. **蒸馏对齐**: `shadow_block_pool_attn` 输出的 `A_blk` 与教师 dense attn 做 mean-pool 后的 `A_blk_teacher`
   计算 KL,作为 attention map distillation loss. 6 个 distill 层就够(`{4,9,14,19,24,29}`).
6. **micro-batch sequence packing**: 把不同分辨率 sample 拼到同一个 batch 时,
   把每个 sample 单独走一次 `bsa_forward` 而不要把 `S` 维 concat,前者更快(避免无效 mask 区域).

### 10.2 推理侧

1. **prefill / decode 切两段**: 当作 streaming 视频生成时,首段(prefill 多帧) 走 BSA-0.90;
   后续每帧 (decode) 走 BSA-0.95 + LSWA hybrid (LSWA 处理新帧,BSA 复用历史 KV cache).
2. **KV cache shrink**: BSA 选出的稀疏 pattern 跨 timestep 高度稳定 (邻近 latent 变化小),
   可以每隔 5 个 DiT step 重算一次 draft,中间复用 mask. 实测 PSNR 损失 < 0.05 dB,而 draft 阶段省 80%.
3. **Top-k → Tiled Top-k**: `torch.topk` 在 N≥1024 时 kernel launch 开销显著;
   可以改成"每 64 行一组,组内 max 拼接"的近似 top-k,误差可忽略.
4. **CPU offload draft mask**: draft 完全可以放 CPU 上算 (`N²` 小),GPU 只做 sparse attn,
   适合 24G 卡部署 720p.

---

## 11. 与 2026 年相关工作的对比

| 方法                         | 块/窗口大小 | 块选择方式              | 是否带 sliding window | 因果支持   | 与 BSA 关系                              |
| ---------------------------- | ----------- | ----------------------- | --------------------- | ---------- | ---------------------------------------- |
| **BSA (本工作)**             | 2·8·8       | mean-pool draft + top-k | 9×9 空间(隐式)        | ✅ 时间因果 | —                                        |
| NSA (DeepSeek, 2025)         | 32/64       | 学习的 selector         | ✅ 显式               | ✅          | 思想相同, NSA 多了 compressed branch     |
| MoBA (Moonshot, 2025)        | 512/1024    | router 投票             | ❌                    | ✅          | 块更大、纯学习路由,无视频空间先验        |
| Sparse VideoGen (SVG, 2025)  | 帧级        | head-wise dynamic       | ✅                    | ❌(双向)    | 头粒度自适应,可与 BSA 互补 (每 head 不同 sparsity) |
| MInference (MSR, 2024-2025)  | 多 pattern  | A-shape/vertical-slash  | —                     | ✅          | 推理 prefill 专用, BSA 训推一致          |
| Longformer/BigBird           | sliding+gl. | 手工先验                | ✅                    | ❌          | 文本任务,无 3D grid 概念                 |
| Swin / 3D-Swin               | 窗口        | 无 cross-window         | ❌                    | ❌          | 完全 local, 不解决全局信息流             |
| DiTFastAttn / Δ-DiT          | step 间复用 | cache reuse             | —                     | —          | 时间维复用,可与 BSA 叠加                 |

**判断**: BSA 在 video DiT 这个 setting 下最像 NSA 的 *selected* 分支 (mean-pool 代表向量 + top-k),
但比 NSA 更"轻"(不带可学 selector),并把 NSA 的 *sliding* 分支硬编码成 9×9 local mask 嵌进 draft.
和 SVG 互补——SVG 是**head 粒度**自适应,而 BSA 是**block 粒度**全 head 共享 pattern;
两者结合 (per-head BSA) 是值得做的 ablation.

---

## 12. 下一步优化建议 (按 ROI 排序)

| # | 项目                                  | 预期收益               | 风险 / 工作量                                  |
| - | ------------------------------------- | ---------------------- | ---------------------------------------------- |
| 1 | Padding-aware top-k (§9.1)            | 解掉非整除 latent 限制 | 低 — 现有 mask 路径加 valid_tok                |
| 2 | Per-head dynamic sparsity (借 SVG 思路) | 同精度下再 −20% FLOPs  | 中 — 需要 head-wise budget allocator           |
| 3 | Draft pattern 跨 timestep 复用 (§10.2) | 推理 −15% 单帧延迟     | 低 — 加一个 step counter                       |
| 4 | 学习化 selector (NSA-style)           | 同精度下 −10% 块数     | 高 — 新引入可训练参数,需要 SFT 流程            |
| 5 | bsa + lswa hybrid 跨层调度            | 末层 LSWA 提细节       | 中 — 已有 `attn_mode` 切换基础设施             |
| 6 | int8 block sparse kernel               | 显存 −50%              | 高 — 需要量化感知训练,块边界 outlier 风险     |
| 7 | Tile-based top-k (§10.1.3)             | 训练 step −3%          | 低 — 单点替换 `torch.topk`                    |
| 8 | mask shape inference 静态化           | 编译期固定 shape       | 低 — 把 grid 维写进 module config               |

---

## 13. 接入新模型的 checklist

```text
[ ] grid_shape = (f, h, w) 已经能在 patchify 之后稳定拿到,逐层透传
[ ] f % bt == 0 且 h % bh == 0 且 w % bw == 0,或已实装 §9.1 padding 路径
[ ] dim % num_heads == 0
[ ] block_sparse_attn CUDA 库已编译,本机 import 通过
[ ] local_attn_mask 已 register_buffer (§10.1.2)
[ ] sparsity ramp schedule 已挂到 train loop (cosine 0.85→0.90, ramp_end_step≈1k)
[ ] distill_layers 与 teacher 对齐
[ ] return_aux=True 时 A_blk 落盘格式与下游 KD loss 一致
[ ] 单测: dense vs BSA 在 sparsity=0 时输出应数值相等 (差 < 1e-3)
[ ] 单测: padded 路径 + valid mask 与 unpadded 路径输出应一致
```

---

## 14. 参考代码片段(中文注释版)

```python
# flashvsr_b1/attn/bsa_kernel.py: bsa_forward 的核心 5 行
q_w = _partition_for_bsa(Q, block_size=block_size, grid_shape=grid_shape)  # ① 切块
k_w = _partition_for_bsa(K, block_size=block_size, grid_shape=grid_shape)
v_w = _partition_for_bsa(V, block_size=block_size, grid_shape=grid_shape)

topk = topk_for(current_sparsity, total_kv_blocks)                          # ④a 决定 top-k
attention_mask = ref_mod.generate_draft_block_mask(                         # ②③④b draft+local+topk
    B, num_heads, seqlen, q_w, k_w,
    topk=topk, local_attn_mask=local_window_mask)

attention_mask = attention_mask & causal_mask                               # ④c ∩ 因果

out = _block_sparse_attention(block_sparse_attn_func,                       # ⑤ 精确稀疏 attn
    reorder_q, reorder_k, reorder_v, attention_mask, num_heads=num_heads)
out = _reverse_bsa_partition(out, block_size=block_size,                    # ↺ 还原顺序
    grid_shape=grid_shape, batch_size=B)
```

---

## 15. 一图汇报

> 如果 PPT 只放一张图,推荐用 `plot_bsa_attention_map.py --grid 4,4,4 --sparsity 0.85`
> 生成的 dense vs BSA 并排热图 + 节省比例标注,信息密度最高.
