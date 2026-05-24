"""绘制 LSWA (Local Sparse Window Attention) 的可视化图。

该脚本是 LSWA.md 的配套可视化工具,**纯 Python + matplotlib + numpy**,
不依赖任何 CUDA 扩展。复刻 `flashvsr_b1/attn/lswa.py` 的 mask 语义,
输出两幅子图:
    (左) 帧对帧 mask: query 帧 t 是否能 attend 到 key 帧 t' (受 window_size[0] 和因果约束)
    (右) 单 token 邻域 mask: 当 query 在中心位置时, 一帧内哪些 (r,c) 进入 softmax (受 window_size[1:] 约束)

典型用法
--------
    python plot_lswa_attention_map.py --grid 4,8,8 --window 2,5,5
    python plot_lswa_attention_map.py --grid 21,45,80 --window 2,21,21
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError as exc:
    raise SystemExit("需要 matplotlib: pip install matplotlib") from exc


def parse_triplet(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"需要三元组 a,b,c, 收到 {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def build_frame_to_frame_mask(f: int, wt: int) -> np.ndarray:
    """LSWA 的"帧对帧" mask: query 帧 t 看 key 帧 t' iff t-wt+1 <= t' <= t."""
    qs = np.arange(f)[:, None]
    ks = np.arange(f)[None, :]
    return (ks <= qs) & (ks >= qs - wt + 1)


def build_spatial_neighborhood_mask(
    h: int, w: int, wh: int, ww: int, center: tuple[int, int],
) -> np.ndarray:
    """单 token 的邻域 mask, 边角越界位置标 False (与 lswa.py 的 valid 一致)."""
    cr, cc = center
    r_half, c_half = wh // 2, ww // 2
    rs = np.arange(h)
    cs = np.arange(w)
    YY, XX = np.meshgrid(rs, cs, indexing="ij")
    in_row = (YY >= cr - r_half) & (YY < cr - r_half + wh)
    in_col = (XX >= cc - c_half) & (XX < cc - c_half + ww)
    return in_row & in_col


def lswa_pair_count(grid: tuple[int, int, int], window: tuple[int, int, int]) -> int:
    """估算 LSWA 总 attention pair 数 (考虑帧边界和空间边界)."""
    f, h, w = grid
    wt, wh, ww = window
    total = 0
    rs = np.arange(h)
    cs = np.arange(w)
    YY, XX = np.meshgrid(rs, cs, indexing="ij")
    r_half, c_half = wh // 2, ww // 2
    # 每个 query token (r,c) 的空间有效 key 数
    per_tok_spatial = (
        ((YY[..., None] >= YY.flatten()[None, None] - r_half)
         & (YY[..., None] < YY.flatten()[None, None] - r_half + wh)
         & (XX[..., None] >= XX.flatten()[None, None] - c_half)
         & (XX[..., None] < XX.flatten()[None, None] - c_half + ww))
    )  # 太重, 用简化估算:
    valid_per_q = np.zeros(h * w, dtype=np.int64)
    for r in range(h):
        for c in range(w):
            in_r = max(0, min(h - 1, r + (wh - 1) // 2) - max(0, r - wh // 2) + 1)
            in_c = max(0, min(w - 1, c + (ww - 1) // 2) - max(0, c - ww // 2) + 1)
            valid_per_q[r * w + c] = in_r * in_c
    spatial_total = int(valid_per_q.sum())

    for t in range(f):
        # 时间窗口内有效 key 帧数
        n_kv_frames = min(wt, t + 1)
        total += spatial_total * n_kv_frames
    return total


def render_two_panels(
    grid: tuple[int, int, int],
    window: tuple[int, int, int],
    out_path: Path,
) -> None:
    f, h, w = grid
    wt, wh, ww = window

    frame_mask = build_frame_to_frame_mask(f, wt)
    center = (h // 2, w // 2)
    spatial_mask = build_spatial_neighborhood_mask(h, w, wh, ww, center)

    pair_lswa = lswa_pair_count(grid, window)
    pair_dense = sum((t + 1) * (h * w) ** 2 for t in range(f))  # causal dense
    save_ratio = 1.0 - pair_lswa / max(1, pair_dense)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), constrained_layout=True)

    # (左) 帧对帧 mask
    ax = axes[0]
    ax.imshow(frame_mask, cmap="Greys", aspect="equal", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"帧对帧 mask  (wt={wt})\nf={f}, 时间复杂度 O(f·wt) 而非 O(f²)", fontsize=11)
    ax.set_xlabel("key 帧 t'")
    ax.set_ylabel("query 帧 t")
    ax.set_xticks(range(f))
    ax.set_yticks(range(f))

    # (右) 单 token 空间邻域
    ax = axes[1]
    ax.imshow(spatial_mask, cmap="Blues", aspect="equal", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(
        f"中心 token 的空间邻域  (wh={wh}, ww={ww})\n"
        f"h={h}, w={w}, 有效邻居={int(spatial_mask.sum())}/{wh*ww}",
        fontsize=11,
    )
    ax.set_xlabel("列 c")
    ax.set_ylabel("行 r")
    # 标出 query token 位置
    cr, cc = center
    ax.add_patch(Rectangle((cc - 0.5, cr - 0.5), 1, 1, fill=False, edgecolor="tab:red", linewidth=2))
    ax.text(cc, cr, "★", color="tab:red", ha="center", va="center", fontsize=12)
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))

    fig.suptitle(
        f"LSWA 可视化 — grid={grid}, window={window}\n"
        f"总 attention pair: {pair_lswa:,} (vs Dense causal {pair_dense:,}, 节省 {save_ratio:.1%})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[ok] saved → {out_path}")
    print(f"     pair LSWA  = {pair_lswa:,}")
    print(f"     pair Dense = {pair_dense:,}")
    print(f"     节省比例  = {save_ratio:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=parse_triplet, default=(4, 8, 8),
                    help="latent 三维 grid, 形如 f,h,w (默认 4,8,8)")
    ap.add_argument("--window", type=parse_triplet, default=(2, 5, 5),
                    help="LSWA 三维窗口, 形如 wt,wh,ww (默认 2,5,5)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "lswa_attention_map.png")
    args = ap.parse_args()

    if args.window[1] % 2 == 0 or args.window[2] % 2 == 0:
        print(f"[warn] window 推荐用奇数边长 (得到对称邻域), 收到 {args.window}")
    render_two_panels(grid=args.grid, window=args.window, out_path=args.out)


if __name__ == "__main__":
    main()
