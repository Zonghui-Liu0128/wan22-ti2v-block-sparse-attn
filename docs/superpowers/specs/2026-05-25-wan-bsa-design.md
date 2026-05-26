# Wan2.2-5B SelfAttention — Block Sparse Attention 设计

- **日期**: 2026-05-25
- **作者**: 与 Claude 协同设计
- **范围**: 单文件改动 `diffsynth/models/wan_video_dit.py` + 1 个新增测试脚本 `tests/test_wan_bsa_visualization.py`
- **算法基准**: `template_BSA.py` 中的 `JoinAttention.split_and_squeeze_3D` + `JoinAttention.get_sparse_mask`(下文统称"模板")

---

## 1. 目标

1. 在 `wan_video_dit.py::SelfAttention` 中加入 Block Sparse Attention(下称 BSA),算法逻辑与模板严格一致。
2. 通过 token 重排实现**真稀疏计算**(不只是稠密 mask),计算/内存按 `sparse_ratio` 实际下降。
3. 支持 `latent_t / latent_h / latent_w` 不能被对应 `block_t / block_h / block_w` 整除的情况,**不损失块级 similarity 与 top-K 选择精度**。
4. 改动只触及 `wan_video_dit.py` 与一个新测试脚本,代码逻辑清晰;`bsa_enable=False`(默认)时所有 forward 与现状逐字节一致。
5. 提供可视化测试与程序化自检,验证实现正确性。

非目标:
- 不重写 `AttentionModule` / `CrossAttention` / `patchify` / `unpatchify` / `Head` 等无关模块。
- 不引入新的外部依赖(FlexAttention 是 PyTorch ≥ 2.5 原生 API)。
- 不做训练 loss / 蒸馏端的改造。

---

## 2. 改动概览(文件级 footprint)

`diffsynth/models/wan_video_dit.py` 内增量:
- **新增 1 个类**: `BlockSparseAttention(nn.Module)`,封装 pad / 块连续重排 / 计数加权块均值 / top-K / 稀疏 attention / 反重排 全流程。
- **`SelfAttention.__init__`**: 增加 4 个 kwargs:`bsa_enable / bsa_block_size / bsa_sparse_ratio / bsa_backend`(默认 `False / (2,2,2) / 0.5 / 'flex'`)。
  - `self.attn` 改成根据开关选 `AttentionModule(num_heads)` 或 `BlockSparseAttention(num_heads, bsa_block_size, bsa_sparse_ratio, bsa_backend)`。
- **`SelfAttention.forward`**: 新增 `video_shape: Optional[Tuple[int,int,int]] = None` kwarg。仅当 `self.attn` 为 `BlockSparseAttention` 时下传 `video_shape`。
- **`DiTBlock.__init__`**: 透传 BSA 4 个 kwargs 给 `SelfAttention`。
- **`DiTBlock.forward`**: 新增 `video_shape` kwarg,透传给 `self_attn`。
- **`WanModel.__init__`**: 新增 BSA 4 个 kwargs(默认 `bsa_enable=False`),透传给每个 `DiTBlock`。
- **`WanModel.forward`**: 把已有的 `(f, h, w)` 以 `video_shape=(f, h, w)` 形式传给每个 block(`gradient_checkpoint_forward` 调用同步加 kwarg)。

新增文件:
- `tests/test_wan_bsa_visualization.py` —— 可视化 + 程序化自检脚本。
- `tests/bsa_viz/*.png` —— 测试运行产生,提交 `.gitignore` 不动(只是输出目录约定)。

不动:`AttentionModule`、`flash_attention`、`CrossAttention`、`GateModule`、`MLP`、`Head`、`patchify`、`unpatchify`、`WanModel` 其他逻辑。

---

## 3. 算法详解(严格对齐模板,叠加 boundary 扩展)

### 3.1 输入契约

`BlockSparseAttention.forward(q, k, v, video_shape)`:
- `q, k, v`:`(B, L, n·d)`,已被 `SelfAttention` 完成 `qkv proj + RMSNorm + RoPE`。token 按 **(f, h, w) 光栅序**排列(由 `patchify` + `unpatchify` + `freqs` reshape 已约束)。
- `video_shape = (f, h, w)`,满足 `L = f·h·w`。
- 输出形状:`(B, L, n·d)`,与 `AttentionModule.forward` 完全相同的契约。

### 3.2 Step A — Pad + 块连续重排

设 `(bt, bh, bw) = block_size`,且:
- `F_pad = ceil(f / bt) · bt`
- `H_pad = ceil(h / bh) · bh`
- `W_pad = ceil(w / bw) · bw`
- `new_T = F_pad / bt`,`new_H = H_pad / bh`,`new_W = W_pad / bw`
- `num_blocks = new_T · new_H · new_W`
- `block_size_total = bt · bh · bw`

操作:
```
(B, L, n·d) → (B, n, L, d) → (B, n, f, h, w, d)
            zero-pad → (B, n, F_pad, H_pad, W_pad, d)
            rearrange ' b n (tT bt) (hH bh) (wW bw) d
                      -> b n (tT hH wW) (bt bh bw) d '
            (即 (B, n, num_blocks, block_size_total, d))
```

这一步等价于模板 `split_and_squeeze_3D` 隐含的 layout —— 模板在 `view(B*n, new_T, bt, new_H, bh, new_W, bw, C)` 时假设 token 序列就是按 (f h w) 排,这里我们显式 rearrange 到块连续,语义完全一致。

### 3.3 Step B — 真实 token 计数 + 有效位置 mask(boundary 不损失精度的核心)

```python
# count[t_out, h_out, w_out] = 该 block 内真实 token 数
eff_t[t_out] = min(bt, f - t_out * bt)   # 形状 (new_T,)
eff_h[h_out] = min(bh, h - h_out * bh)   # 形状 (new_H,)
eff_w[w_out] = min(bw, w - w_out * bw)   # 形状 (new_W,)
count = eff_t[:, None, None] * eff_h[None, :, None] * eff_w[None, None, :]
        # 形状 (new_T, new_H, new_W),flatten 成 (num_blocks,)
```

`valid_token_mask`:形状 `(F_pad, H_pad, W_pad)` 的 bool,True=真实 token,False=padding。flatten 后用于稀疏 attention 时屏蔽 padded key。

两者只与 `(f, h, w)` 与 `block_size` 有关,**整次 forward 仅计算一次**,无 batch / head 依赖,可缓存(同一 shape 复用)。

### 3.4 Step C — 块级 mean(等价模板)

```python
# Q_padded: (B, n, num_blocks, block_size_total, d)
# pad 位置 Q/K 为 0,sum 等价于"只对真实 token 求和"
count_f = count.to(dtype=Q_padded.dtype).view(1, 1, num_blocks, 1)
Q_block = Q_padded.sum(dim=-2) / count_f   # (B, n, num_blocks, d)
K_block = K_padded.sum(dim=-2) / count_f
```

正确性:
- **全 block** (`count == block_size_total`):退化为 `mean(dim=-2)`,与模板 `split_and_squeeze_3D` 的 `mean(dim=[2,4,6])` 数值完全相同。
- **boundary block** (`count < block_size_total`):pad 位置零向量不贡献 sum,除以真实计数 → 真实 token 上的无偏均值。**这正是 boundary 不影响 similarity / top-K 精度的保证**。
- 任何 boundary block 至少有 1 个真实 token(因为该 block 必然落在 `(f, h, w)` 网格内),除零安全。

### 3.5 Step D — 块级相似度 + top-K(与模板 sign 约定逐字一致)

```python
sim = Q_block @ K_block.transpose(-1, -2)              # (B, n, num_blocks, num_blocks)
K_drop = int(num_blocks * sparse_ratio)
top_index = (-sim).topk(K_drop, dim=-1).indices        # (B, n, num_blocks, K_drop)
attend_block = torch.ones(B, n, num_blocks, num_blocks, dtype=torch.bool, device=...)
attend_block = attend_block.scatter_(-1, top_index, False)
```

**约定说明**(与模板完全一致):
- `(-sim).topk(K_drop)` 取的是 sim **最小的 K_drop 个** key block,对应"最不相似"的 K_drop 个块。
- `attend_block` 在这 K_drop 个位置上为 `False`,其余为 `True`。
- 即 **`sparse_ratio` 是"丢弃比例"**;`sparse_ratio=0.5` 丢掉一半最不相似的块,保留另一半参与 attention;`sparse_ratio=0` 等价于 dense。
- 模板源码注释 "把这部分值mask为 -inf" 与该约定吻合。

### 3.6 Step E — 稀疏 attention 计算(双 backend)

把 `(B, n, num_blocks, block_size_total, d)` 视作 `(B, n, num_blocks·block_size_total, d)` 的"块连续 padded permuted 序列",其中 `L_pad = num_blocks·block_size_total = F_pad·H_pad·W_pad`。

#### Backend `'flex'`(默认)

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
# attend_block: (B, n, num_blocks, num_blocks) bool
# valid_token_mask_flat: (L_pad,) bool

def mask_mod(b, h_, q_idx, kv_idx):
    q_block = q_idx // block_size_total
    kv_block = kv_idx // block_size_total
    return attend_block[b, h_, q_block, kv_block] & valid_token_mask_flat[kv_idx]

block_mask = create_block_mask(
    mask_mod, B=B, H=n_heads,
    Q_LEN=L_pad, KV_LEN=L_pad,
    BLOCK_SIZE=block_size_total,
)
out = flex_attention(Q_perm, K_perm, V_perm, block_mask=block_mask)
# out: (B, n, L_pad, d)
```

要点:
- `BLOCK_SIZE` 优先取 `block_size_total`(让 kernel 的稀疏粒度与逻辑块对齐,跳过未选中 KV)。但 FlexAttention 内部 kernel 通常要求 `BLOCK_SIZE` 是 16/32 的倍数。当 `block_size_total` 太小或非对齐(例如 debug 场景 `block_size_total=8`)时,取 `BLOCK_SIZE = lcm_round_up(block_size_total, 64)`;`mask_mod` 仍按 token idx 解出 `q_block / kv_block`,结果数值等价,只是 kernel 内部多走几条空 row。
- `mask_mod` 同时屏蔽 padded key,padded query 行的输出会算出来但稍后被裁掉。
- `block_mask` 自身大小 `(B, n, num_blocks, num_blocks) bool`;生产 case `num_blocks=7·13·5=455` 时 `≈ 30 · 455² · 1B ≈ 6 MB`,完全可接受。

#### Backend `'sdpa_chunked'`

适合调试 / 小 grid / FlexAttention 不可用环境:

```python
# 默认 chunk_size = 64(可在构造时配置);若 num_blocks <= chunk_size,直接一批做完
for chunk_start in range(0, num_blocks, chunk_size):
    q_block_ids = arange(chunk_start, min(chunk_start + chunk_size, num_blocks))
    q = Q_blocks[:, :, q_block_ids]                       # (B, n, c, bst, d)
    sel = top_index_keep[:, :, q_block_ids]               # (B, n, c, num_keep)
    # 通过 gather 取出选中的 K/V blocks 然后 flatten
    K_g = gather_blocks(K_blocks, sel)                    # (B, n, c, num_keep·bst, d)
    V_g = gather_blocks(V_blocks, sel)
    pad_mask = build_pad_mask(sel, valid_token_mask)      # 屏蔽 padded key
    q_flat = q.flatten(2, 3)                              # (B, n, c·bst, d) — 但每个 c 内只看自己 K_g 切片,改为分维
    # 实际实现:把 c 视作 batch 维,逐 c 调用一次 SDPA,或 reshape 成 (B·c, n, bst, d)
    out_chunk = sdpa(q_reshape, K_g, V_g, attn_mask=pad_mask)
    write_back(out_chunk → Out_blocks at q_block_ids)
```

要点:
- 单批内不会跨 c 维做 attention(每个 query block 只看自己 gather 出来的 keys),通过把 `c` 升到 batch 维实现。
- 内存上界 ≈ `B·c·n·num_keep·block_size_total·d·dtype_size`(K_gathered + V_gathered 共两份)。生产 case `(num_keep≈228, block_size_total=18, n=30, d=128)` 取 `chunk_size=64` 约 1 GB,完全可跑。
- **两个 backend 在数值上等价**(对同一份 `attend_block` 求 attention)。生产分辨率推荐 `'flex'` 主要是因为它避免 Python 循环 launch overhead 且 kernel 自身更优;`'sdpa_chunked'` 在 FlexAttention 不可用 / 调试 / 小 grid / 老版本 PyTorch 上同样可用。

### 3.7 Step F — 反重排 + 裁剪

```
(B, n, num_blocks, block_size_total, d)
  rearrange ' b n (tT hH wW) (bt bh bw) d
            -> b n (tT bt) (hH bh) (wW bw) d '
  crop → (B, n, f, h, w, d)
  flatten + transpose → (B, L, n·d)
```

返回。

---

## 4. 配置接口

### `WanModel.__init__` 新增参数

| 参数 | 默认值 | 类型 | 说明 |
|---|---|---|---|
| `bsa_enable` | `False` | bool | 是否启用 BSA。默认 False → 行为与现状逐字节一致 |
| `bsa_block_size` | `(2, 2, 2)` | Tuple[int, int, int] | `(block_t, block_h, block_w)` |
| `bsa_sparse_ratio` | `0.5` | float ∈ [0, 1) | 丢弃比例(与模板同义) |
| `bsa_backend` | `'flex'` | Literal['flex', 'sdpa_chunked'] | 稀疏 attention 后端 |

`DiTBlock` 与 `SelfAttention` 透传相同 4 个 kwargs。`SelfAttention.__init__` 内:

```python
if bsa_enable:
    self.attn = BlockSparseAttention(num_heads, bsa_block_size, bsa_sparse_ratio, bsa_backend)
else:
    self.attn = AttentionModule(num_heads)
```

### Forward 签名变化

`DiTBlock.forward(x, context, t_mod, freqs, video_shape=None)`
`SelfAttention.forward(x, freqs, video_shape=None)`
`WanModel.forward(...)` 内调用 `block(x, context, t_mod, freqs, video_shape=(f, h, w))`,`gradient_checkpoint_forward` 同步加这个 kwarg。

`video_shape` 在 BSA 关闭时被忽略 → 不会破坏现有调用方。

---

## 5. 调试 / 可视化机制

### `BlockSparseAttention` 内的轻量 debug hook

```python
self._debug_record: bool = False
# 当 True 时,forward 末尾把以下中间量挂到模块属性:
#   self._dbg_attend_block:   (B, n, num_blocks, num_blocks) bool
#   self._dbg_top_index:      (B, n, num_blocks, K_drop) int
#   self._dbg_count:          (num_blocks,) int
#   self._dbg_valid_mask:     (F_pad, H_pad, W_pad) bool
#   self._dbg_pad_shape:      (F_pad, H_pad, W_pad)
#   self._dbg_video_shape:    (f, h, w)
#   self._dbg_block_size:     (bt, bh, bw)
```

默认 False,生产 forward 路径上仅一个 if 判断,无开销。

### 测试脚本 `tests/test_wan_bsa_visualization.py`

依赖:`torch`、`numpy`、`matplotlib`(项目已用)、`einops`(项目已用)。无需 GPU / 真权重。

**主流程**:
1. 对每个测试场景:
   - 用固定 seed 造 `(B=1, L_real, n·d)` 随机 Q/K/V(已 RMSNorm + 任意确定性变换)。
   - 实例化 `BlockSparseAttention(num_heads, block_size, sparse_ratio, backend)`,设 `_debug_record = True`。
   - 调一次 forward,得到 `out_bsa`,取出 `_dbg_attend_block` 等。
2. **可视化**(每个场景输出 1~2 张 PNG):
   - **token-level 图**(`L_real ≤ 256` 时输出):
     - 把 `attend_block` lift 到 token 级:`attend_token[i, j] = attend_block[b0, h0, block(i), block(j)]`,只在真实 token 上取值,矩阵大小 `L_real × L_real`。
     - 颜色:`True → #FFD700`(黄)/ `False → #800080`(紫)。`matplotlib.colors.ListedColormap(['#800080', '#FFD700'])` + `imshow(attend_token, cmap, vmin=0, vmax=1)`。
     - 细网格线:每个 token 一格(`ax.set_xticks(arange(-0.5, L_real, 1), minor=True)` + grid)。
     - 粗网格线:block 边界,通过累加 effective 边界位置 `cum_t = cumsum(eff_t)` 等,在这些位置加粗线。
     - 坐标轴标 token 序号(每 K 个标一次以防过密)。
     - 在每个 block 中心叠一个小字 `(t_out, h_out, w_out)`。
     - 标题包含 `f / h / w / block / sparse_ratio / num_blocks / K_drop / head=0, batch=0`。
   - **block-level 图**(始终输出):
     - 直接 `imshow(attend_block[0, 0])`,`num_blocks × num_blocks`,同配色。
     - 网格 + block 序号 + 标题。
3. **程序化正确性自检**:
   - 用 `attend_token`(真实 token × 真实 token bool)构造 `attn_mask`(True=参与),跑 `F.scaled_dot_product_attention(Q_full, K_full, V_full, attn_mask=attn_mask)` 得参考输出 `out_ref`。
   - 与 BSA forward 的 `out_bsa` 逐元素对比:`(out_ref - out_bsa).abs().max() <= 1e-5`(fp32)/`<= 1e-3`(fp16/bf16),否则 assert 失败。
   - 两个 backend 都跑一次,且互比 `'flex'` vs `'sdpa_chunked'` 输出。
   - 额外打印 `count`、boundary block 列表、`K_drop`、每 block 实际"被丢弃数"统计 → 便于人工核对。

**场景表**:

| 场景 key | (f, h, w) | block | sparse_ratio | backend | token 图 | block 图 |
|---|---|---|---|---|---|---|
| `debug_clean` | (4, 4, 4) | (2, 2, 2) | 0.5 | flex + sdpa_chunked | ✓ (64×64) | ✓ (8×8) |
| `boundary` | (3, 4, 5) | (2, 2, 2) | 0.5 | flex + sdpa_chunked | ✓ (60×60) | ✓ (12×12) |
| `prod_scale` | (21, 26, 15) | (3, 2, 3) | 0.5 | flex | 跳过(过大) | ✓ (455×455 小图) |

**输出**:`tests/bsa_viz/wan_bsa_<scene>_<token|block>.png`。

---

## 6. 验证标准(用户验收前必须全部 ✓)

1. 默认 `bsa_enable=False` 时,跑一遍现有最小 `WanModel.forward` 单测(或手工随机输入),输出与改动前**逐字节相同**。
2. `bsa_enable=True, backend='flex'`,`tests/test_wan_bsa_visualization.py` 三场景:
   - 程序化自检全部 pass(BSA vs 参考 dense-with-mask SDPA 数值一致)。
   - 两个 backend 输出互比一致(`debug_clean`、`boundary` 两个场景)。
3. PNG 视觉检查:
   - `debug_clean`:整除场景,所有 block 同尺寸 8 tokens,黄/紫格子排列按 `attend_block` 与 block-token 映射严丝合缝。
   - `boundary`:`f=3, w=5` 不整除,共 12 个 block,其中 6 个 boundary block,粗网格清晰显示三种 boundary 尺寸 `2·2·1=4`(w 边界)/ `1·2·2=4`(t 边界)/ `1·2·1=2`(t+w 双重边界),其余 4 个 `2·2·2=8` 全 block;真实 token 合计 `4·8 + 2·4 + 4·4 + 2·2 = 60 = 3·4·5` ✓。
   - `prod_scale`:block 图小,但能看出每行恰好有 `num_blocks - K_drop` 个黄格(与 `sparse_ratio` 吻合)。

---

## 7. 不变量与边角清单

- token 排列严格保持 (f, h, w) 光栅序;reshape / rearrange 全部用 einops `' b n (tT bt) (hH bh) (wW bw) d -> ... '` 显式标维,**不会**混入 (f w h) 之类的隐式顺序错位。
- 任何 boundary block 至少 1 个真实 token(否则它根本不应该出现在 `new_T·new_H·new_W` 网格内);除零安全。
- `BlockSparseAttention` 与 `AttentionModule` 接口完全相同(`(q, k, v)` in、`(B, L, n·d)` out),只是 BSA 多吃一个 `video_shape`;调用点显式区分。
- FlexAttention 调用要求 `Q_LEN = KV_LEN = L_pad` 是 `block_size_total` 的整数倍 —— 由 pad 步骤保证。
- `gradient_checkpoint_forward` 调用同步传 `video_shape`;训练 + checkpoint 路径无影响。
- 不擅自把已有的 `flash_attention` / `sageattention` / FlashAttn 路径改成 BSA;BSA 是平行新路径。

---

## 8. 不在本次范围内的事项(已识别但暂不处理)

- 当前 `WanModel.forward` 第 530 行 `x, (f, h, w) = self.patchify(x)` 解包 与 `patchify` 实际只返回 `x` 的不一致 —— 应是仓库内既有未完成项,与 BSA 无关,不动。
- 训练侧损失 / 蒸馏端的 sparse 监督。
- BSA 与 Ulysses / Context Parallelism 的组合(模板里有,但 wan_video_dit.py 本身不带 CP)。
