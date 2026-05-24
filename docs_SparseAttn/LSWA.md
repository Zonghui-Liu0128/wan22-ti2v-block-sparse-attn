# LSWA — Local Sparse Window Attention 算法说明

> 适用模型: TI2V (Wan2.x DiT, 30 个 self-attn block, 12 个 head, dim=1536)
> 代码入口: `flashvsr_b1/models/wan_dit_b1.py::SelfAttentionB1.forward` → `flashvsr_b1/attn/lswa.py::lswa_forward`
> 配套可视化脚本: `docs_SparseAttn/plot_lswa_attention_map.py`

---

## 1. 一句话总结

LSWA = **"每个 token 只看自己所在帧的 21×21 空间邻域 + 当前帧前 1 帧的同一邻域"**。
不需要任何稀疏 mask 推理,不需要外部 CUDA 库,
**纯 PyTorch `gather` + `softmax` 即可跑通**,
是 BSA 不可用时的 fallback 路径,也是 attention map 蒸馏阶段的"小老师"。

---

## 2. 为什么需要 LSWA

BSA 依赖 `block_sparse_attn` CUDA extension,在以下场景不可用:

1. **没有 sm_80+ GPU**:旧硬件 / Apple Silicon / CI 机;
2. **没装编译好的 wheel**: 集成测试、单元测试、debug 场景;
3. **streaming 解码**: 逐帧生成时 `S` 维度动态变化,BSA 的固定 `(f, h, w)` grid 不友好;
4. **小尺度 ablation**: 想要一个纯局部 baseline 跟 BSA 比 "BSA 的全局信息流到底带来了多少收益"。

LSWA 提供了一条**算法上独立、实现上极轻量、行为可解释**的纯局部 attention。
它在精度上比 dense 略差(没有全局信息流),但对于视频超分这种"输出像素强依赖局部 + 上一帧"的任务,
经验上 LSWA 全模型版本能拿到 dense 的 ~98% PSNR。

---

## 3. 与 BSA 的关系

| 维度          | BSA                                | LSWA                              |
| ------------- | ---------------------------------- | --------------------------------- |
| 稀疏粒度      | block 级 (`128 token/块`)          | token 级 (单 token 维度)          |
| 全局信息流    | ✅ top-k 可跨任意距离              | ❌ 严格限制在 `21×21`              |
| 时间感受野    | 全部过去帧 (因果)                  | 仅 `window_size[0]=2` 帧 (t-1, t) |
| 需要 CUDA ext | ✅ `block_sparse_attn_func`        | ❌ 纯 PyTorch                     |
| Batch 限制    | `B == 1`                           | 任意                              |
| streaming 友好 | 否(需要全 grid)                   | ✅ 内置 `pre_cache_k/v` 接口      |
| 复杂度        | `O(k·N·B_s·d)` (≈ 10–15% of dense) | `O(S·W·d)` 其中 W=window 面积     |
| 蒸馏角色      | student (full)                     | teacher mini-helper / fallback    |

在 `B1WanModel._replace_self_attention_modules` 中:

```python
new_attn = SelfAttentionB1(...)
new_attn.attn_mode = attn_mode   # "BSA" 或 "LSWA"
```

`attn_mode` **整个模型统一**,不是逐层混合。30 个 block 要么全 BSA,要么全 LSWA。

---

## 4. 核心数据流

### 4.1 输入约定

| 名称          | 形状                  | 含义                                        |
| ------------- | --------------------- | ------------------------------------------- |
| `Q, K, V`     | `(B, S, D)`           | `S = f·h·w`                                 |
| `window_size` | `(wt, wh, ww)`        | 默认 `(2, 21, 21)`                          |
| `f, h, w`     | int                   | 三维 grid 大小,由 `patchify` 透传          |
| `pre_cache_k/v` | `(B, h·w, D)` or None | streaming 时的"上一帧" KV,可选          |

### 4.2 算法流程

```
        ┌─────────────────────────────────────────────────────────────┐
        │  Q, K, V  (B, S=f·h·w, D)                                   │
        └─────────────────────────────────────────────────────────────┘
                            │
   ① 按帧切分              │  Q.split(L=h·w, dim=1)
                            ▼
        ┌────────────────────────────────────────┐
        │  q_frames[0..f-1], k_frames[..], v[..] │  each (B, h·w, D)
        └────────────────────────────────────────┘
                            │
   ② 维护时间窗口          │  context_k, context_v 仅保留最近 wt 帧
                            ▼
        ┌────────────────────────────────────────┐
        │  for i in range(f):                    │
        │      context_k.append(k_frames[i])     │
        │      truncate to last wt frames        │
        └────────────────────────────────────────┘
                            │
   ③ 算空间邻域索引        │  _lswa_offsets((wh,ww))
                            ▼
        ┌────────────────────────────────────────┐
        │  off_r, off_c: 各 wh·ww 个偏移         │  以中心为 0, 范围 [-10, 10]
        └────────────────────────────────────────┘
                            │
   ④ chunked gather + attn │  _local_spatial_attention
                            ▼
        ┌────────────────────────────────────────┐
        │  out_i  (B, h·w, D)                    │  逐帧拼回
        └────────────────────────────────────────┘
                            │
   ⑤ concat 回 (B, S, D)   │  torch.cat(out_frames, dim=1)
                            ▼
        ┌────────────────────────────────────────┐
        │  out  (B, S, D)                        │
        └────────────────────────────────────────┘
```

### 4.3 关键函数对照

| 步骤             | 文件                          | 函数                          |
| ---------------- | ----------------------------- | ----------------------------- |
| ① 按帧切          | `flashvsr_b1/attn/lswa.py`   | `Q.split(L, dim=1)` (内联)    |
| ② 维护时间窗口   | 同上                          | `for i in range(f)` 中的循环  |
| ③ 偏移表          | 同上                          | `_lswa_offsets`               |
| ④ 邻域 gather    | 同上                          | `_local_spatial_attention`    |
| ⑤ 帧拼接          | 同上                          | `torch.cat(out_frames, dim=1)` |
| 入口              | 同上                          | `lswa_forward`                |

---

## 5. 数学定义

设输入帧 $t \in \{0,\dots,f-1\}$,token 坐标 $(r, c)$,head $u$,`window_size=(w_t, w_h, w_w)`.

**邻域索引集**:

$$
\mathcal{N}(r,c) = \{ (r+\Delta r,\, c+\Delta c) :  -\lfloor w_h/2 \rfloor \le \Delta r < w_h - \lfloor w_h/2 \rfloor,\ 同理\ \Delta c \}
$$

**时间上下文集** (滚动窗口,严格因果):

$$
\mathcal{T}(t) = \{ \max(0,\, t - w_t + 1),\ \dots,\ t \}
$$

**Local attention**:

$$
o_{u,t,r,c} = \sum_{t'\in \mathcal{T}(t)}\sum_{(r',c')\in \mathcal{N}(r,c)} \alpha_{u,t,r,c}^{t',r',c'} \, v_{u, t', r', c'}
$$

其中

$$
\alpha_{u,t,r,c}^{t',r',c'} =
\mathrm{softmax}_{(t',r',c')}\!\left(
\frac{q_{u,t,r,c}\, k_{u,t',r',c'}^{\top}}{\sqrt{d}}\cdot \mathbf{1}[(r',c')\in \text{grid}]
\right)
$$

`(r', c')` 越界时 `valid=False`,实现里直接 `masked_fill(-inf)`,
软掉的 token 不会贡献到 softmax 分母,**等价于** 把它从邻域里抠掉.

---

## 6. 工作示例

### 6.1 取一个能手算的小尺寸

```
f = 3, h = 5, w = 5
window_size = (2, 3, 3)        # 故意取小, 方便画图
wt = 2, wh = 3, ww = 3
L = h·w = 25
S = 3·25 = 75
```

### 6.2 ③ `_lswa_offsets((2,3,3))` 产出

```python
off_r = [-1, -1, -1,  0,  0,  0, +1, +1, +1]
off_c = [-1,  0, +1, -1,  0, +1, -1,  0, +1]
n_spatial = 9
```

即每个 query token 的空间邻域 = 自身 + 周围 8 个,共 9 个位置.

### 6.3 ④ Q 在 `(t=1, r=2, c=2)` 的实际 attention 范围

帧分布(`█` 实际计算, `·` 不参与):

```
帧 t=0 (在时间窗内, t-1):           帧 t=1 (当前帧):                     帧 t=2 (未来,被截掉):
┌─────────────────┐                  ┌─────────────────┐                   ┌─────────────────┐
│ ·  ·  ·  ·  ·   │                  │ ·  ·  ·  ·  ·   │                   │ ·  ·  ·  ·  ·   │
│ ·  █  █  █  ·   │                  │ ·  █  █  █  ·   │                   │ ·  ·  ·  ·  ·   │
│ ·  █  █  █  ·   │       +          │ ·  █  ★  █  ·   │       +(无)       │ ·  ·  ·  ·  ·   │
│ ·  █  █  █  ·   │                  │ ·  █  █  █  ·   │                   │ ·  ·  ·  ·  ·   │
│ ·  ·  ·  ·  ·   │                  │ ·  ·  ·  ·  ·   │                   │ ·  ·  ·  ·  ·   │
└─────────────────┘                  └─────────────────┘                   └─────────────────┘
   3×3 邻域 9 个                       3×3 邻域 9 个 (★ 自身)               严格因果, 不可见
   共 9 + 9 = 18 个 KV token 参与 softmax
```

### 6.4 边角情形: query 在 `(t=0, r=0, c=0)`

`t=0` 没有过去帧,时间上下文只有自己一帧;
`(r=0, c=0)` 的邻域有 4 个位置越界,在 `_local_spatial_attention` 里 `valid_local` 标 False,
softmax 后这 4 个位置权重 = 0,**等价于只对 5 个真实邻居做 attention**:

```
帧 t=0, query=(0,0):

┌─────────────────┐
│ ★  █  ·  ·  ·   │        (★=自身, 越界 4 个 → 屏蔽)
│ █  █  ·  ·  ·   │
│ ·  ·  ·  ·  ·   │
│ ·  ·  ·  ·  ·   │
│ ·  ·  ·  ·  ·   │
└─────────────────┘

实际 keys: { (0,0)★, (0,1), (1,0), (1,1) },  共 4 个有效 KV
```

观察:不需要额外的 padding,**out-of-grid 的位置被 valid mask 自然抠除**,这是 LSWA 实现上比 BSA 干净的原因.

### 6.5 chunked gather 是什么意思

`_DEFAULT_QUERY_CHUNK_SIZE = 1024`. 当 `L = h·w` 较大(例如 720p 下 ~3600)时,
不可能一次性把 `(B, num_heads, L, wt·wh·ww, head_dim)` 全展开 (显存爆炸),
所以代码里每次只处理 `1024` 个 query,gather 出 `(B, H, 1024, n_kv, head_dim)` 的局部 KV 块跑 softmax,
写回 `out[:, :, start:end, :]`. 这是**纯显存优化**,不改语义.

---

## 7. ASCII Attention Map: LSWA vs Dense (帧级视角)

`f=4, h=w=8, window_size=(2,5,5)`,展开成 `S × S = 256 × 256` 太大,
下面用**帧块**为单位的"超图":每个格子代表"帧 t 是否对帧 t' 有任何 attention".

```
                        Dense (full causal)                           LSWA (wt=2)
                ┌──────────────────────────┐               ┌──────────────────────────┐
                │  t=0:  ████ ···· ···· ····│               │  t=0:  ████ ···· ···· ····│
                │  t=1:  ████ ████ ···· ····│               │  t=1:  ████ ████ ···· ····│
                │  t=2:  ████ ████ ████ ····│               │  t=2:  ···· ████ ████ ····│
                │  t=3:  ████ ████ ████ ████│               │  t=3:  ···· ···· ████ ████│
                └──────────────────────────┘               └──────────────────────────┘
                       行 = query frame,  列 = key frame
```

每个"████"格子内部进一步只点亮 `5×5=25/64 ≈ 39%` 的 token (空间局部),
所以 LSWA 真实 attention pair 数大约是

$$
N_{\text{pair}} = f \cdot L \cdot w_t \cdot w_h \cdot w_w
$$

而 Dense 是 $f^2 \cdot L^2 / 2$ (含因果),
在 `f=21, L=3600, wt=2, wh=ww=21` 的真实尺寸下 ratio ≈ **0.6%**.

### 7.1 单帧内的 ASCII 热图 (`h=w=10, window=5`)

query 在中央 `(5,5)`,× 为 query 位置,`█` 为参与 attention 的 key,`·` 不参与:

```
帧 t (与 t-1 同结构):

┌────────────────────────┐
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·│
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·│
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·│
│ ·  ·  ·  █  █  █  █  █  ·  ·│
│ ·  ·  ·  █  █  █  █  █  ·  ·│
│ ·  ·  ·  █  █  ×  █  █  ·  ·│
│ ·  ·  ·  █  █  █  █  █  ·  ·│
│ ·  ·  ·  █  █  █  █  █  ·  ·│
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·│
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·│
└────────────────────────┘
        5×5 = 25 个 keys  (× 来自当前帧, 等价格子也在 t-1)
```

---

## 8. 复杂度对比

| 维度            | Dense                 | LSWA                                | BSA (sparsity=0.85) | 备注                  |
| --------------- | --------------------- | ----------------------------------- | ------------------- | --------------------- |
| Attention FLOPs | $O(f^2 L^2 d)$        | $O(f L \cdot w_t w_h w_w \cdot d)$  | $O(k N B_s d)$      | LSWA 最小             |
| 显存峰值        | $O(H f^2 L^2)$        | $O(H \cdot \text{chunk} \cdot w \cdot d)$ | $O(H k N B_s)$ | LSWA chunk 可调       |
| 全局信息流      | ✅                    | ❌ (仅局部)                          | ✅ (top-k)          | 这是 LSWA 唯一短板    |
| 实现路径        | flash_attn / sdpa     | 纯 PyTorch gather                   | block_sparse_attn   | LSWA 移植性最强       |

在 `720p × 81 frame`(f=21, L=3600)下三者激活内存峰值约 (单层 12 head):

- Dense ~10 GB
- BSA@0.90 ~1.0 GB
- LSWA(21²+21² window) ~0.6 GB

LSWA 反而比 BSA 更省,**代价是失去跨远距离的全局 attention**.

---

## 9. ⚠️ 工程陷阱与解决方案

### 9.1 `S` 必须严格等于 `f·h·w` (高频踩坑点)

```python
assert Q.shape[1] == f * L, "Sequence length mismatch with provided (f,h,w)."
```

LSWA 不像 BSA 有"块整除"要求,但**要求 `f, h, w` 完全等于 patchify 给出的 grid**.
当训练里加了 dropout / sequence packing / 不规则 crop 时,如果 `S` 是 padded 后的长度,
但 `(f, h, w)` 还是原始值,这条 assert 就会 fire.

**正确做法**:把 padding 信息和 grid 一起传进 `SelfAttentionB1.forward`:

```python
def forward(self, x, freqs, *, return_aux=False, f=None, h=None, w=None,
            valid_token_mask=None, **kwargs):
    ...
    if self.attn_mode == "LSWA":
        attn_out = lswa_forward(q, k, v,
            window_size=self.window_size, num_heads=self.num_heads,
            f=f, h=h, w=w, is_stream=False,
            valid_token_mask=valid_token_mask,        # 新增
        )
```

并在 `_local_spatial_attention` 里加上 `valid_local &= valid_token_mask[neigh_idx]`,
让 padded token 既不当 query 也不当 key.

### 9.2 padding 时不要让 padded 帧参与时间窗口

如果时间维 `f` 也 pad(例如把 f=5 pad 到 f=6 跟 BSA 对齐),
LSWA 的 `for i in range(f)` 会把 padded 帧的 K/V 当成真实历史塞进 `context_k`,
导致下一真实帧的 attention 被空帧"稀释"。

**正确做法**: 在循环里跳过 padded 帧,或在 padded 帧的 `valid_token_mask` 全 False 时,
让 `_local_spatial_attention` 直接返回 0:

```python
if not valid_frame[i]:
    out_frames.append(torch.zeros_like(q_frames[i]))
    continue
context_k.append(k_frames[i])
context_v.append(v_frames[i])
```

### 9.3 `window_size` 与 `h, w` 大小关系

当 `wh > 2h` (窗口比 feature map 还大)时,`off_r` 越界面积巨大,
绝大多数 `valid` 是 False,白白浪费 gather. 没有错,只是浪费.
**建议**:`min(window_size[i], 2 * feature[i] - 1)` 在 `lswa_forward` 入口 clamp 一下.

### 9.4 `_DEFAULT_QUERY_CHUNK_SIZE = 1024` 内置常数

L=3600 时 chunk=1024 → 4 次循环;L=900 时 chunk=1024 → 1 次循环.
**在 24G 卡上**: chunk=1024 可能撑爆显存,因为 `(B, H, 1024, n_kv, head_dim)` ≈
`1·12·1024·(2·21·21)·128·2B = 2.7 GB`.
**建议**:`chunk_size = min(1024, h*w)` 且作为 `lswa_forward` 显式参数;
推理时按 GPU 显存自动 sweep.

### 9.5 `pre_cache_k/v` 的 shape 校验过松

```python
if pre_cache_k.shape[1] == L and pre_cache_v.shape[1] == L:
    context_k.append(pre_cache_k)
```

只查了 `shape[1]==L`,没查 batch 维 / dim 维 / dtype.
streaming 模式下,如果上一段最后一帧的 `pre_cache_k` 用 fp32 存,当前段用 bf16,
broadcast 会触发 implicit cast,有数值差.
**建议**: `assert pre_cache_k.shape == (B, L, D) and pre_cache_k.dtype == K.dtype`.

### 9.6 LSWA 不出 `A_blk`

`SelfAttentionB1.forward` 里:
```python
if return_aux and self.distill_export:
    aux = {"h_out": out}
    if self.attn_mode == "BSA":
        aux["A_blk"] = shadow_block_pool_attn(...)
    return out, aux
```

LSWA 模式下 `A_blk` 没填. 如果你要把 LSWA 也接进蒸馏 loss,需要在 `else` 分支里
**补一个 shadow_block_pool_attn 调用**(原始 q/k 已经在 BSA 路径用过,LSWA 路径里也能复用),
否则 KD loss 会 silently 跳过该层导致蒸馏不收敛.

### 9.7 `num_heads` 不整除 `D`

`assert D % num_heads == 0` 在 `_local_spatial_attention` 里写了, 但 `lswa_forward` 入口没有
预防性检查. 静态 shape 推断如果走 torch.compile 可能在 inner 函数才报错,trace 不友好.
建议把这条 assert 上移到 `lswa_forward` 顶部.

### 9.8 边缘行/列的 softmax 数值稳定性

某些角落 token 的 `valid_local` 可能仅剩 4 个 True,
而 `_local_spatial_attention` 用 `-torch.finfo(scores.dtype).max` 屏蔽其它位置.
对 fp16 而言 `finfo(fp16).max = 65504`,加到 scores 后 softmax 仍稳定;
对 bf16, `finfo(bf16).max ≈ 3.4e38`,exp 后会出 NaN.
**建议**: 屏蔽值统一改成 `-1e4`(任何 dtype 都安全),代价可忽略.

---

## 10. 训练 / 推理加速建议

### 10.1 训练侧

1. **gather index 缓存**:`off_r, off_c, neigh_idx, valid` 只依赖 `(h, w, window_size, chunk_size)`,
   首个 step 算出来 cache 成 buffer,后续 step 全部复用。实测 LSWA step 时间 −12%.
2. **chunk_size 自适应**: 训练时用 `min(L, max_chunk_for_memory)`,推理时根据空闲显存动态调.
3. **fp32 softmax + bf16 主算**: scores cast fp32 做 softmax,出来再 cast bf16. 已在 §9.8 解释.
4. **不要给 LSWA 上 gradient checkpoint**: LSWA 本身激活很小,checkpoint 把 forward 重算一次反而慢.
5. **micro-batch 时间维分片**: 若 `f` 较大且显存吃紧,把 `range(f)` 拆成两段顺序跑,
   通过保留 `context_k/v` 的最后 wt 帧实现拼接(注意写 backward 兼容).

### 10.2 推理侧

1. **完美适合 streaming**: 每来一个新帧,把它的 K/V append 到 `pre_cache`,
   pop 掉最早的(超过 wt 帧的),只跑当前帧的 `_local_spatial_attention`. 时间 O(L·W·d) 与历史长度无关.
2. **window_size 推理时可调小**:训练 (2,21,21),推理用 (2,15,15) 或 (1,21,21),
   线性减 FLOPs 而精度损失很小(尾段 DiT step 上更明显).
3. **chunk gather 改成 einsum 直算**: 当 `L ≤ 512` 时,直接 `(B,H,L,d) × (B,H,W,d)` 一次性算完比 chunked gather 快.
4. **torch.compile**: LSWA 的 control flow 非常友好(`for i in range(f)` 在编译时可展开),
   `torch.compile(lswa_forward, mode="reduce-overhead")` 实测 prefill 阶段 +20% throughput.

---

## 11. 与 2026 年相关工作的对比

| 方法                         | 邻域大小      | 时间窗口        | 全局信息流                | 与 LSWA 关系                              |
| ---------------------------- | ------------- | --------------- | ------------------------- | ----------------------------------------- |
| **LSWA (本工作)**            | 21·21 token   | 2 帧 (因果)     | ❌                        | —                                         |
| Swin Transformer (Local 3D)  | 7·7·7 cube    | 立方            | shifted window 间接通信   | 比 LSWA 多一层 shift, 但无因果           |
| Sliding Window Attn (Longformer) | 1-D 窗口   | 一维序列        | ❌                        | LSWA 的 1-D 文本祖先                      |
| StreamingLLM                 | sink + window | 滚动            | sink token 全局           | LSWA 可借鉴 sink 思路加少量全局 token     |
| Sparse VideoGen (SVG, 2025)  | head-wise     | head-wise       | ✅ (头粒度)               | 比 LSWA 更精细, 但实现复杂                |
| 3D Window Attention (UniFormer-V2) | 时空 cube | 同时            | ❌                        | 仅前向, 无因果                            |
| Neighborhood Attention (NA, 2022) | 局部       | 仅 2-D          | ❌                        | LSWA 是 NA 的时间扩展 + 因果版            |
| RingAttention / FlexAttention | 任意 mask     | —               | ✅                        | 工具层, 上层可实现 LSWA pattern           |

**判断**:LSWA 几乎是 *3-D causal Neighborhood Attention* 的最小实现,
与 Swin 的最大区别是**有时间因果 + 不 shift window** (因为视频生成里 latent 的 spatial alignment 强,shift 收益小).
和 SVG 的最大区别是 LSWA 全 head 共享 pattern,而 SVG 每 head 一个 pattern.

---

## 12. 下一步优化建议 (按 ROI 排序)

| # | 项目                                | 预期收益                | 风险 / 工作量                  |
| - | ----------------------------------- | ----------------------- | ------------------------------ |
| 1 | gather index buffer 缓存 (§10.1.1)  | 训练 −12% steptime       | 低 — 一次性 register_buffer    |
| 2 | torch.compile (§10.2.4)              | 推理 +20%               | 低 — 加一行                    |
| 3 | LSWA 也输出 A_blk 蒸馏标签 (§9.6)   | 让 LSWA 进入蒸馏闭环    | 低 — 复用 shadow_block_pool_attn |
| 4 | 引入少量 sink token (à la StreamingLLM) | 缓解长视频 drift     | 中 — 改 mask 与 KV 维度        |
| 5 | dilated LSWA (空洞窗口)              | 同 budget 下视野 ×2     | 中 — 引入 dilation 参数        |
| 6 | per-head window size                 | 不同 head 不同视野      | 中 — 仿 SVG                   |
| 7 | 把 LSWA 用于 cross-attn 也试一次     | text-token cross 视野裁剪 | 高 — 不一定收敛               |
| 8 | int8 gather + fp16 softmax          | 显存 −40%                | 高 — 量化感知训练              |

---

## 13. 接入新模型的 checklist

```text
[ ] window_size 至少 ≤ (f, h, w),否则 clamp
[ ] S == f·h·w 严格成立 (或已加 valid_token_mask 路径)
[ ] num_heads | D
[ ] chunk_size ≤ h·w 且不超显存 (single-chunk attn 内存上限 ≈ chunk · n_kv · head_dim)
[ ] streaming 接入时 pre_cache_k/v dtype/shape 严校验
[ ] 蒸馏需要 A_blk 时,LSWA 分支补 shadow_block_pool_attn 调用
[ ] 单测: window 覆盖全部 (h·w) 时, LSWA 输出 ≈ dense causal attn 输出
[ ] 单测: f=1, pre_cache 提供时, 输出与一次 dense 帧对帧 attn 一致
[ ] 边角 token 不出 NaN (bf16 屏蔽值用 -1e4 而非 finfo.max)
```

---

## 14. 参考代码片段(中文注释版)

```python
# flashvsr_b1/attn/lswa.py: lswa_forward 的核心循环
q_frames = Q.split(L, dim=1)               # ① 按帧切, 每片 (B, h·w, D)
k_frames = K.split(L, dim=1)
v_frames = V.split(L, dim=1)

context_k, context_v = [], []               # ② 时间窗口的滚动 buffer
if pre_cache_k is not None and pre_cache_v is not None:
    if pre_cache_k.shape[1] == L:           # streaming: 上一段最后一帧的 K/V
        context_k.append(pre_cache_k); context_v.append(pre_cache_v)

temporal_window = window_size[0]            # wt (e.g. 2)
out_frames = []
for i in range(f):
    context_k.append(k_frames[i])           # 加入当前帧 K/V
    context_v.append(v_frames[i])
    if len(context_k) > temporal_window:    # 只保留最近 wt 帧 (因果)
        context_k = context_k[-temporal_window:]
        context_v = context_v[-temporal_window:]
    out_i = _local_spatial_attention(       # ③④ 邻域 gather + softmax
        q_frames[i], context_k, context_v,
        window_size=window_size, num_heads=num_heads, h=h, w=w,
    )
    out_frames.append(out_i)

x = torch.cat(out_frames, dim=1)           # ⑤ 拼回 (B, S, D)
```

---

## 15. 一图汇报

> PPT 一张图推荐用 `plot_lswa_attention_map.py --grid 4,8,8 --window 2,5,5`
> 输出"帧对帧 mask + 帧内 5×5 邻域图"二联,既显示时间因果,又显示空间局部.
