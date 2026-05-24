"""绘制 BSA (Block Sparse Attention) 的稀疏 mask 与 Dense 因果 attention 的对比图。

该脚本是 BSA.md 的配套可视化工具,**纯 Python + matplotlib + numpy**,
不依赖 `block_sparse_attn` CUDA 库,因此在任意机器上都能跑出汇报用的 PNG。

它复刻了 `flashvsr_b1/attn/bsa_kernel.py::bsa_forward` 里
`generate_draft_block_mask` 的 5 步逻辑 (mean-pool draft → local mask → top-k → causal),
用随机 Q/K 模拟出 block 粒度的 attention map,方便领导一眼看清"BSA 在哪儿稀疏掉了哪儿"。

典型用法
--------
    python plot_bsa_attention_map.py --grid 4,4,4 --block 2,2,2 --sparsity 0.85
    python plot_bsa_attention_map.py --grid 8,8,8 --block 2,8,8 --sparsity 0.90 --local 3
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("需要 matplotlib: pip install matplotlib") from exc


def parse_triplet(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"需要三元组 a,b,c, 收到 {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def build_local_block_mask(h_blk: int, w_blk: int, win: int) -> np.ndarray:
    """复刻 wan_video_dit.build_local_block_mask_shifted_vec_normal_slide 的纯 numpy 版."""
    H, W = h_blk, w_blk
    r_half = win // 2
    rs = np.arange(H)
    cs = np.arange(W)
    YY, XX = np.meshgrid(rs, cs, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    start_r = r_all - r_half
    end_r = start_r + win - 1
    start_c = c_all - r_half
    end_c = start_c + win - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    return in_row & in_col  # [H*W, H*W]


def simulate_bsa_mask(
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    sparsity: float,
    local_win: int,
    head_dim: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (dense_causal_mask, bsa_mask, draft_scores) 都是 [N_blk, N_blk] 的 bool/float."""
    f, h, w = grid
    bt, bh, bw = block
    assert f % bt == 0 and h % bh == 0 and w % bw == 0, "grid 必须能被 block 整除"
    N_t = f // bt
    N_h = h // bh
    N_w = w // bw
    spatial_blocks = N_h * N_w
    N = N_t * spatial_blocks

    rng = np.random.default_rng(seed)
    Q_blk = rng.standard_normal((N, head_dim)).astype(np.float32)
    K_blk = rng.standard_normal((N, head_dim)).astype(np.float32)

    # 时间因果 mask: 块 j 的时间索引 <= 块 i 的时间索引
    t_q = np.arange(N) // spatial_blocks
    t_k = np.arange(N) // spatial_blocks
    causal = t_k[None, :] <= t_q[:, None]
    dense_mask = causal.copy()

    # 局部空间 mask: 用 local_win × local_win 的空间窗口
    local_spatial = build_local_block_mask(N_h, N_w, local_win)  # [Nh*Nw, Nh*Nw]
    # 在时间维度上 tile: 任何 t_q, t_k 组合都用同一个空间局部 mask
    local_full = np.kron(np.ones((N_t, N_t), dtype=bool), local_spatial)

    # ① Draft attention: softmax(Q K^T / sqrt(d) + local 加法 mask)
    scores = Q_blk @ K_blk.T / math.sqrt(head_dim)
    scores_with_local = scores.copy()
    scores_with_local[~local_full] = -np.inf  # 局部窗口外 -inf
    # softmax 沿 key 维
    exp = np.exp(scores_with_local - scores_with_local.max(axis=1, keepdims=True))
    draft = exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)

    # ② Top-k 选块: 每行选最大的 topk 个
    topk = max(1, int(round(N * (1.0 - sparsity))))
    # 取每行第 topk 大的值作为阈值
    thresholds = np.sort(draft, axis=1)[:, -topk - 1] if topk < N - 1 else np.full(N, -np.inf)
    topk_mask = draft > thresholds[:, None]

    # ③ BSA 最终 mask = top-k ∩ causal (local 已经隐式包含在 top-k 高分项里)
    bsa_mask = topk_mask & causal

    return dense_mask, bsa_mask, draft


def render_two_panels(
    dense_mask: np.ndarray,
    bsa_mask: np.ndarray,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    sparsity: float,
    out_path: Path,
) -> None:
    """画 dense vs BSA 的并排 mask 图,带统计信息."""
    f, h, w = grid
    bt, bh, bw = block
    N_t = f // bt
    spatial_blocks = (h // bh) * (w // bw)
    N = N_t * spatial_blocks

    dense_active = dense_mask.sum()
    bsa_active = bsa_mask.sum()
    save_ratio = 1.0 - bsa_active / max(1, dense_active)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), constrained_layout=True)
    for ax, mask, title in [
        (axes[0], dense_mask, f"Dense causal\nN_blk={N}, active={dense_active} ({dense_active/N/N:.1%})"),
        (axes[1], bsa_mask, f"BSA (sparsity={sparsity:.2f})\nactive={bsa_active} ({bsa_active/N/N:.1%}, 节省 {save_ratio:.1%})"),
    ]:
        ax.imshow(mask, cmap="Greys", aspect="equal", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"KV 块  (N={N} 块, 块大小={bt}·{bh}·{bw})")
        ax.set_ylabel("Q 块")
        # 在每个时间块边界画浅红虚线
        for i in range(1, N_t):
            ax.axhline(i * spatial_blocks - 0.5, color="tab:red", linewidth=0.5, alpha=0.6, ls="--")
            ax.axvline(i * spatial_blocks - 0.5, color="tab:red", linewidth=0.5, alpha=0.6, ls="--")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"BSA vs Dense 块级 attention 对比 — grid={grid}, block={block}",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[ok] saved → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=parse_triplet, default=(4, 4, 4),
                    help="latent 三维 grid, 形如 f,h,w (默认 4,4,4)")
    ap.add_argument("--block", type=parse_triplet, default=(2, 2, 2),
                    help="block 三维大小, 形如 bt,bh,bw (默认 2,2,2)")
    ap.add_argument("--sparsity", type=float, default=0.85,
                    help="稀疏率 ∈ [0,1), top-k = round(N·(1-sparsity)) (默认 0.85)")
    ap.add_argument("--local", type=int, default=3,
                    help="局部空间窗口大小 (奇数), 默认 3 (即 3×3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "bsa_attention_map.png")
    args = ap.parse_args()

    dense_mask, bsa_mask, _ = simulate_bsa_mask(
        grid=args.grid, block=args.block,
        sparsity=args.sparsity, local_win=args.local, seed=args.seed,
    )
    render_two_panels(
        dense_mask, bsa_mask,
        grid=args.grid, block=args.block,
        sparsity=args.sparsity, out_path=args.out,
    )


if __name__ == "__main__":
    main()
