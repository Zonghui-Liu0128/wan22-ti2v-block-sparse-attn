# 将 BSA 迁移到内网 B200(LoRA-trained Wan2.2-5B)

适用前提:内网 fork 基于 DiffSynth-Studio,改过 LoRA 训练相关代码,可能动过
`wan_video_dit.py` / `wan_video.py` / 推理入口。BSA 自身**不动 LoRA**(它只
替换 `SelfAttention.attn` 这个无参数的子模块,`q/k/v/o` 的 LoRA 适配层保留)。

## 1. 取需要迁移的 5 个文件

从 `wan22-ti2v-block-sparse-attn` 的 `H20_debug` 分支拿:

```text
diffsynth/models/wan_video_dit.py        ← 含 BlockSparseAttention 类 + 4 个 bsa_* kwargs
diffsynth/pipelines/wan_video.py         ← model_fn_wan_video 块循环加 video_shape=(f,h,w)
examples/wanvideo/model_inference/run_bsa_test_ti2v.py  ← 推理 CLI(可改造为支持 LoRA)
tests/test_wan_bsa_unit.py               ← 28 个单元测试,迁移正确性的唯一标准
tests/test_wan_bsa_realistic_viz.py      ← 可视化(可选)
```

## 2. 三个最小插入点

如果内网 `wan_video_dit.py` / `wan_video.py` 不能整文件覆盖(因为有 LoRA
hook),按下列三处手工 patch:

**(a) `wan_video_dit.py` 顶部 imports 之后**

```python
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    _flex_attention = None
    _create_block_mask = None
```

**(b) `wan_video_dit.py` 整段 `BlockSparseAttention` 类**(直接从我们的 commit
拷过去),并在 `SelfAttention.__init__` / `forward`、`DiTBlock.__init__` /
`forward`、`WanModel.__init__` 各加 4 个 `bsa_*` kwargs + `video_shape` 透传。

> 关键检查:内网若改过 `SelfAttention`(例如插了 LoRA adapter),保留它们,
> **只在最后把 `self.attn = AttentionModule(...)` 换成根据 `bsa_enable` 二选一**。
> LoRA 包的是 `self.q/k/v/o`(`nn.Linear`),与 BSA 完全不冲突。

**(c) `wan_video.py` 的 `model_fn_wan_video` 块循环里**

```python
x = gradient_checkpoint_forward(
    block,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    x, context, t_mod, freqs,
    video_shape=(f, h, w),   # ← 新增这一行
)
```

如果内网用了 VAP/VACE/animate 分支,**同样的位置(它们各自的 block 调用)
也补上 `video_shape=(f, h, w)`**。漏一个就会在该路径下撞 `assert
video_shape is not None`。

## 3. 跑单元测试(必做,这是迁移正确性的硬指标)

```bash
pip install pytest matplotlib
cd <your-diffsynth-root>
python -m pytest tests/test_wan_bsa_unit.py -v
```

H20 上 28/28 PASS。B200 上 torch ≥ 2.5 + CUDA 时同样应该 28/28。**任一失败
都不要继续往下跑**,优先用失败 case 的报错定位插入点漏改。

## 4. 在 LoRA 推理入口接入 BSA

LoRA 模型加载完毕后,在第一次 forward 之前调用一次:

```python
from diffsynth.models.wan_video_dit import BlockSparseAttention

def enable_bsa(model, block_size=(2,4,4), sparse_ratio=0.5, backend="flex", chunk_size=64):
    for blk in model.blocks:
        sa = blk.self_attn
        ref = sa.q.weight  # LoRA 之后 q.weight 仍存在
        sa.attn = BlockSparseAttention(
            num_heads=sa.num_heads, block_size=block_size,
            sparse_ratio=sparse_ratio, backend=backend, chunk_size=chunk_size,
        ).to(device=ref.device, dtype=ref.dtype)
        sa.bsa_enable = True

# 用法
pipe = WanVideoPipeline.from_pretrained(...)
pipe.load_lora_weights("/path/to/lora.safetensors", alpha=1.0)  # 内网原有 LoRA 加载逻辑
enable_bsa(pipe.dit, block_size=(2,4,4), sparse_ratio=0.5, backend="flex")
video = pipe(...)
```

LoRA 的 q/k/v/o adapter 仍跑;BSA 只接管 `q·kᵀ` 之后的稀疏 attention 计算。

## 5. 单图烟雾测试(确认推理路径没坏)

```bash
# 先跑 sparse_ratio=0(dense baseline),确认 LoRA + BSA plumbing 没破坏原有行为
python my_inference_entrypoint.py --sparse-ratio 0.0 --output dense.mp4

# 再跑 0.5,看看 LoRA 风格 + 50% 稀疏是否退化可接受
python my_inference_entrypoint.py --sparse-ratio 0.5 --output bsa50.mp4
```

dense 与原 LoRA 推理结果应**逐 token 一致**(`bsa_enable=False` 时 forward
完全等价旧路径)。0.5 通常画面整体还在,细节略糊;>0.75 会有明显伪影。

## 6. 远程 360° 数据特有的注意点

- **block_size**:水平 360° 视频强调时间维一致性,推荐 `(2, 4, 4)`(`bt=2` 折
  叠相邻帧,`bh=bw=4` 把空间窗口拉到 ~16 token 的 GPU 友好尺度)。如果 LoRA
  训练时用的是更短/更长帧数,latent_t 不一定能被 2 整除 —— BSA 的 boundary
  支持会自动处理,但要确认 `_compute_block_info` 的 `count` 在边界 block 上 ≥1
  (`python -c "from ...; print(BlockSparseAttention(8,(2,4,4),0.5,'flex')._compute_block_info((你的f,你的h,你的w), torch.device('cuda'))['count'])"`)。
- **手办 360°** 的核心信息往往集中在中央前景:从 `--dump-attention-png` 出来
  的 attend pattern 应该能看到模型在中央列上形成 *hot keys*(类似我们 90% 推理
  时的现象);若 90% 稀疏后只剩前景列被关注,说明 BSA 的语义在该 LoRA 上是合理
  的,可以继续往低稀疏率(0.3~0.5)调以保细节。

## 7. 故障排查表

| 现象 | 原因 | 修法 |
|---|---|---|
| `AssertionError: video_shape required` | `model_fn_wan_video` 某条分支没传 `video_shape` | 找到该分支(grep `block(.*context.*t_mod.*freqs`)补上 |
| `RuntimeError: ... flex_attention` 在 B200 上 | torch < 2.5 或 kernel 不兼容 | 换 `--bsa-backend sdpa_chunked`,或升级 torch |
| LoRA 加载后输出全 NaN | LoRA fp16/dtype 与 BSA 内部 fp 不一致 | `enable_bsa` 时 `.to(dtype=ref.dtype)` 已处理;若仍 NaN,核对 `count.to(blocked.dtype)` 一行,在 bf16 下 OK |
| `bsa_enable=False` 输出与改前不一致 | `forward` 的 if/else 改坏了原 `AttentionModule` 路径 | 跑 `test_self_attention_forward_bsa_disabled_matches_pre_bsa_path` 定位 |
| 显存比 dense 还大 | flex 没 compile 时确实会增大显存 | `--bsa-backend sdpa_chunked --bsa-chunk-size 16` 或在 flex_attention 外包一层 `torch.compile` |
