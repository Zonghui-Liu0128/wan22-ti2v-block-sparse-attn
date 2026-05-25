import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward
from .wantodance import WanToDanceRotaryEmbedding, WanToDanceMusicEncoderLayer

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    _flex_attention = None
    _create_block_mask = None


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:        
            return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class BlockSparseAttention(nn.Module):
    """Block Sparse Attention used by SelfAttention when bsa_enable=True.

    Algorithm mirrors template_BSA.py: pad the (f,h,w) volume to block-aligned size,
    rearrange so tokens within the same spatial block are contiguous, compute a
    weighted block-mean of Q/K (boundary blocks normalised by real-token count so
    similarity/top-K precision is unaffected), pick top-K least-similar key blocks
    per query block to *drop* (sparse_ratio == drop fraction), then attend only
    over the surviving block pairs and crop padding away.
    """

    def __init__(
        self,
        num_heads: int,
        block_size: Tuple[int, int, int],
        sparse_ratio: float,
        backend: str = "flex",
        chunk_size: int = 64,
    ):
        super().__init__()
        assert backend in ("flex", "sdpa_chunked"), f"Unknown BSA backend: {backend}"
        assert 0.0 <= sparse_ratio < 1.0, "sparse_ratio must lie in [0, 1)"
        self.num_heads = num_heads
        self.block_size = tuple(block_size)
        self.sparse_ratio = float(sparse_ratio)
        self.backend = backend
        self.chunk_size = chunk_size

        # Debug hook — flip to True from outside to capture intermediates.
        self._debug_record: bool = False
        self._dbg_attend_block = None    # (B, n, num_blocks, num_blocks) bool
        self._dbg_top_index = None       # (B, n, num_blocks, K_drop) long
        self._dbg_count = None           # (num_blocks,) long
        self._dbg_valid_mask = None      # (F_pad, H_pad, W_pad) bool
        self._dbg_pad_shape = None       # (F_pad, H_pad, W_pad)
        self._dbg_video_shape = None     # (f, h, w)
        self._dbg_block_size = None      # (bt, bh, bw)

    def _compute_block_info(
        self, video_shape: Tuple[int, int, int], device: torch.device
    ) -> dict:
        f, h, w = video_shape
        bt, bh, bw = self.block_size
        # ceil-divide; we pad up to a multiple of block.
        new_T = (f + bt - 1) // bt
        new_H = (h + bh - 1) // bh
        new_W = (w + bw - 1) // bw
        F_pad = new_T * bt
        H_pad = new_H * bh
        W_pad = new_W * bw

        # Effective real-token count along each axis per outer block index.
        idx_t = torch.arange(new_T, device=device)
        idx_h = torch.arange(new_H, device=device)
        idx_w = torch.arange(new_W, device=device)
        eff_t = torch.clamp(f - idx_t * bt, max=bt)   # (new_T,)
        eff_h = torch.clamp(h - idx_h * bh, max=bh)   # (new_H,)
        eff_w = torch.clamp(w - idx_w * bw, max=bw)   # (new_W,)

        # Per-block real-token count, flattened in (t_out, h_out, w_out) raster.
        count = (
            eff_t[:, None, None] * eff_h[None, :, None] * eff_w[None, None, :]
        ).reshape(-1).to(torch.long)

        # Valid-token mask in padded volume (F_pad, H_pad, W_pad)
        valid_mask = torch.zeros(F_pad, H_pad, W_pad, dtype=torch.bool, device=device)
        valid_mask[:f, :h, :w] = True

        return {
            "pad_shape": (F_pad, H_pad, W_pad),
            "new_grid": (new_T, new_H, new_W),
            "num_blocks": new_T * new_H * new_W,
            "block_size_total": bt * bh * bw,
            "count": count,
            "valid_mask": valid_mask,
        }

    def _pad_and_permute(
        self,
        x: torch.Tensor,
        video_shape: Tuple[int, int, int],
        info: dict,
    ) -> torch.Tensor:
        """(B, n, L=f*h*w, d) -> (B, n, num_blocks, block_size_total, d) with zero pad."""
        from einops import rearrange  # local to avoid polluting module-level if it ever moves
        f, h, w = video_shape
        bt, bh, bw = self.block_size
        F_pad, H_pad, W_pad = info["pad_shape"]
        nT, nH, nW = info["new_grid"]
        B, n, L, d = x.shape
        assert L == f * h * w, f"x.shape[2]={L} != f*h*w={f*h*w}"
        x = x.view(B, n, f, h, w, d)
        if (F_pad, H_pad, W_pad) != (f, h, w):
            # F.pad expects pad spec from last dim backward: (d, w, h, f) ordering.
            # We don't pad d; we pad w by (0, W_pad-w), h by (0, H_pad-h), f by (0, F_pad-f).
            x = F.pad(x, (0, 0, 0, W_pad - w, 0, H_pad - h, 0, F_pad - f))
        # block-contiguous rearrangement
        x = rearrange(
            x,
            "b n (tT bt) (hH bh) (wW bw) d -> b n (tT hH wW) (bt bh bw) d",
            tT=nT, hH=nH, wW=nW, bt=bt, bh=bh, bw=bw,
        )
        return x

    def _inverse_permute_and_crop(
        self,
        blocked: torch.Tensor,
        video_shape: Tuple[int, int, int],
        info: dict,
    ) -> torch.Tensor:
        """(B, n, num_blocks, block_size_total, d) -> (B, n, L, d), cropping padding."""
        from einops import rearrange
        f, h, w = video_shape
        bt, bh, bw = self.block_size
        F_pad, H_pad, W_pad = info["pad_shape"]
        nT, nH, nW = info["new_grid"]
        x = rearrange(
            blocked,
            "b n (tT hH wW) (bt bh bw) d -> b n (tT bt) (hH bh) (wW bw) d",
            tT=nT, hH=nH, wW=nW, bt=bt, bh=bh, bw=bw,
        )
        x = x[:, :, :f, :h, :w, :].contiguous()
        B, n, _, _, _, d = x.shape
        return x.view(B, n, f * h * w, d)

    def _block_mean(
        self,
        blocked: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:
        """Weighted block-mean: sum over block_size_total dim divided by real-token count.

        Equivalent to mean over real tokens only (padded zeros do not contribute to sum).
        Shape: (B, n, num_blocks, block_size_total, d) -> (B, n, num_blocks, d).
        """
        count = info["count"]  # (num_blocks,) long
        # cast count to match the data dtype to avoid integer divide
        count_f = count.to(dtype=blocked.dtype).view(1, 1, -1, 1)
        return blocked.sum(dim=-2) / count_f

    def _compute_attend_block(
        self,
        Q_block: torch.Tensor,
        K_block: torch.Tensor,
    ) -> torch.Tensor:
        """Block-level similarity then drop K_drop = floor(num_blocks * sparse_ratio)
        least-similar key blocks per query block. Returns (B, n, num_blocks, num_blocks)
        bool where True=attend.
        """
        sim = Q_block @ K_block.transpose(-1, -2)             # (B, n, num_blocks, num_blocks)
        B, n, num_blocks, _ = sim.shape
        K_drop = int(num_blocks * self.sparse_ratio)
        attend = torch.ones(B, n, num_blocks, num_blocks, dtype=torch.bool, device=sim.device)
        if K_drop > 0:
            top_index = (-sim).topk(K_drop, dim=-1).indices    # (B, n, num_blocks, K_drop) long
            attend.scatter_(-1, top_index, False)
            self._last_top_index = top_index  # used for sdpa_chunked backend later
        else:
            self._last_top_index = None
        return attend

    def _sparse_attn_sdpa_chunked(
        self,
        Q_blocked: torch.Tensor,
        K_blocked: torch.Tensor,
        V_blocked: torch.Tensor,
        attend_block: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:
        """Chunked sparse attention using SDPA per query-block chunk.

        Inputs are block-contiguous: (B, n, num_blocks, block_size_total, d).
        For each query block i, gather its kept key blocks and run SDPA;
        the kept set is derived from attend_block (True == keep).
        Returns (B, n, num_blocks, block_size_total, d).
        """
        B, n, num_blocks, bst, d = Q_blocked.shape
        device = Q_blocked.device
        dtype = Q_blocked.dtype
        # num_keep is uniform across rows because we always drop exactly K_drop.
        K_drop = int(num_blocks * self.sparse_ratio)
        num_keep = num_blocks - K_drop

        # keep_index: (B, n, num_blocks, num_keep) long — the kept key-block indices per query.
        # Sort attend_block (bool) descending so True comes first, take first num_keep.
        # attend_block True == keep; sort by attend along last dim (largest first).
        _, sort_idx = attend_block.to(torch.int8).sort(dim=-1, descending=True, stable=True)
        keep_index = sort_idx[..., :num_keep]    # (B, n, num_blocks, num_keep) long

        # valid_mask per key token in block-contiguous layout.
        # _pad_and_permute reorders padded raster (F_pad,H_pad,W_pad) into
        # block-contiguous (num_blocks, block_size_total); apply the same rearrange to
        # the bool validity mask so vm_blocked[block_idx, intra_idx] corresponds to the
        # SAME spatial position as K_blocked[..., block_idx, intra_idx, :].
        from einops import rearrange
        F_pad, H_pad, W_pad = info["pad_shape"]
        bt, bh, bw = self.block_size
        nT, nH, nW = info["new_grid"]
        vm_blocked = info["valid_mask"].to(device)             # (F_pad, H_pad, W_pad) bool
        vm_blocked = rearrange(
            vm_blocked,
            "(tT bt) (hH bh) (wW bw) -> (tT hH wW) (bt bh bw)",
            tT=nT, hH=nH, wW=nW, bt=bt, bh=bh, bw=bw,
        )                                                       # (num_blocks, bst) bool

        out = torch.zeros_like(Q_blocked)
        chunk_size = max(1, min(self.chunk_size, num_blocks))
        for chunk_start in range(0, num_blocks, chunk_size):
            chunk_end = min(num_blocks, chunk_start + chunk_size)
            c = chunk_end - chunk_start
            q_chunk = Q_blocked[:, :, chunk_start:chunk_end]                # (B, n, c, bst, d)
            sel = keep_index[:, :, chunk_start:chunk_end]                   # (B, n, c, num_keep) long

            # Gather K and V along the num_blocks dim.
            # K_blocked: (B, n, num_blocks, bst, d) -> we expand sel to (B, n, c, num_keep, bst, d)
            sel_exp = sel.unsqueeze(-1).unsqueeze(-1).expand(B, n, c, num_keep, bst, d)
            K_exp = K_blocked.unsqueeze(2).expand(B, n, c, num_blocks, bst, d)
            V_exp = V_blocked.unsqueeze(2).expand(B, n, c, num_blocks, bst, d)
            K_g = torch.gather(K_exp, dim=3, index=sel_exp)                 # (B, n, c, num_keep, bst, d)
            V_g = torch.gather(V_exp, dim=3, index=sel_exp)
            K_g = K_g.reshape(B, n, c, num_keep * bst, d)
            V_g = V_g.reshape(B, n, c, num_keep * bst, d)

            # Mask out padded key positions: gather per-block valid into per-keep mask.
            vm_per_block = vm_blocked.unsqueeze(0).unsqueeze(0).expand(B, n, num_blocks, bst)
            vm_exp = vm_per_block.unsqueeze(2).expand(B, n, c, num_blocks, bst)
            vm_g = torch.gather(
                vm_exp, dim=3, index=sel.unsqueeze(-1).expand(B, n, c, num_keep, bst)
            )  # (B, n, c, num_keep, bst) bool
            vm_g = vm_g.reshape(B, n, c, num_keep * bst)

            # Reshape (B, n, c, *, d) -> (B*c, n, *, d) so SDPA treats each c-block independently.
            # For correctness with B>1 we transpose batch and chunk dims before flattening.
            Bc = B * c
            q_flat = q_chunk.permute(0, 2, 1, 3, 4).reshape(Bc, n, bst, d)
            k_flat = K_g.permute(0, 2, 1, 3, 4).reshape(Bc, n, num_keep * bst, d)
            v_flat = V_g.permute(0, 2, 1, 3, 4).reshape(Bc, n, num_keep * bst, d)
            # vm_g: (B, n, c, num_keep*bst) -> (B, c, n, num_keep*bst) -> (Bc, n, 1, num_keep*bst)
            mask_flat = vm_g.permute(0, 2, 1, 3).reshape(Bc, n, 1, num_keep * bst).expand(Bc, n, bst, num_keep * bst)

            out_chunk = F.scaled_dot_product_attention(
                q_flat, k_flat, v_flat, attn_mask=mask_flat
            )                                                                # (Bc, n, bst, d)
            out[:, :, chunk_start:chunk_end] = out_chunk.reshape(B, c, n, bst, d).permute(0, 2, 1, 3, 4)
        return out.to(dtype=dtype)

    def _sparse_attn_flex(
        self,
        Q_blocked: torch.Tensor,
        K_blocked: torch.Tensor,
        V_blocked: torch.Tensor,
        attend_block: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:
        """FlexAttention sparse path. Returns (B, n, num_blocks, block_size_total, d)."""
        if not FLEX_ATTN_AVAILABLE:
            raise RuntimeError(
                "FlexAttention not available — install PyTorch >= 2.5 or switch backend='sdpa_chunked'."
            )
        B, n, num_blocks, bst, d = Q_blocked.shape
        L_pad = num_blocks * bst
        device = Q_blocked.device

        # Permuted padded sequence view: (B, n, L_pad, d)
        Q_perm = Q_blocked.reshape(B, n, L_pad, d).contiguous()
        K_perm = K_blocked.reshape(B, n, L_pad, d).contiguous()
        V_perm = V_blocked.reshape(B, n, L_pad, d).contiguous()

        # Valid token mask in block-contiguous (permuted) order.
        F_pad, H_pad, W_pad = info["pad_shape"]
        bt, bh, bw = self.block_size
        nT, nH, nW = info["new_grid"]
        from einops import rearrange
        vm = info["valid_mask"].to(device)
        vm_perm = rearrange(
            vm,
            "(tT bt) (hH bh) (wW bw) -> (tT hH wW bt bh bw)",
            tT=nT, hH=nH, wW=nW, bt=bt, bh=bh, bw=bw,
        ).contiguous()                                       # (L_pad,) bool

        # FlexAttention kernel BLOCK_SIZE must be a multiple of 16 (typical kernel constraint).
        # We pick the smallest multiple of 16 that is >= block_size_total.
        kernel_block_size = max(16, ((bst + 15) // 16) * 16)

        def mask_mod(b_idx, h_idx, q_idx, kv_idx):
            q_block = q_idx // bst
            kv_block = kv_idx // bst
            return attend_block[b_idx, h_idx, q_block, kv_block] & vm_perm[kv_idx]

        block_mask = _create_block_mask(
            mask_mod,
            B=B,
            H=n,
            Q_LEN=L_pad,
            KV_LEN=L_pad,
            device=device,
            BLOCK_SIZE=kernel_block_size,
        )
        out_perm = _flex_attention(Q_perm, K_perm, V_perm, block_mask=block_mask)
        return out_perm.view(B, n, num_blocks, bst, d)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        video_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        from einops import rearrange
        B, L, nd = q.shape
        n = self.num_heads
        d = nd // n
        f, h, w = video_shape
        assert L == f * h * w, f"L={L} != f*h*w={f*h*w}"
        device = q.device

        q_nd = rearrange(q, "b l (n d) -> b n l d", n=n)
        k_nd = rearrange(k, "b l (n d) -> b n l d", n=n)
        v_nd = rearrange(v, "b l (n d) -> b n l d", n=n)

        info = self._compute_block_info(video_shape, device=device)
        Q_b = self._pad_and_permute(q_nd, video_shape, info)
        K_b = self._pad_and_permute(k_nd, video_shape, info)
        V_b = self._pad_and_permute(v_nd, video_shape, info)
        Q_mean = self._block_mean(Q_b, info)
        K_mean = self._block_mean(K_b, info)
        attend_block = self._compute_attend_block(Q_mean, K_mean)

        if self.backend == "flex":
            out_b = self._sparse_attn_flex(Q_b, K_b, V_b, attend_block, info)
        elif self.backend == "sdpa_chunked":
            out_b = self._sparse_attn_sdpa_chunked(Q_b, K_b, V_b, attend_block, info)
        else:  # already asserted in __init__, kept for safety
            raise ValueError(f"Unknown backend: {self.backend}")

        out_nd = self._inverse_permute_and_crop(out_b, video_shape, info)
        out = rearrange(out_nd, "b n l d -> b l (n d)").contiguous()

        if self._debug_record:
            self._dbg_attend_block = attend_block.detach()
            self._dbg_top_index = getattr(self, "_last_top_index", None)
            self._dbg_count = info["count"].detach()
            self._dbg_valid_mask = info["valid_mask"].detach()
            self._dbg_pad_shape = info["pad_shape"]
            self._dbg_video_shape = tuple(video_shape)
            self._dbg_block_size = tuple(self.block_size)

        return out


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        bsa_enable: bool = False,
        bsa_block_size: Tuple[int, int, int] = (2, 2, 2),
        bsa_sparse_ratio: float = 0.5,
        bsa_backend: str = "flex",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.bsa_enable = bsa_enable
        if bsa_enable:
            self.attn = BlockSparseAttention(
                num_heads=num_heads,
                block_size=bsa_block_size,
                sparse_ratio=bsa_sparse_ratio,
                backend=bsa_backend,
            )
        else:
            self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, video_shape: Optional[Tuple[int, int, int]] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if self.bsa_enable:
            assert video_shape is not None, "video_shape required when bsa_enable=True"
            x = self.attn(q, k, v, video_shape=video_shape)
        else:
            x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        bsa_enable: bool = False,
        bsa_block_size: Tuple[int, int, int] = (2, 2, 2),
        bsa_sparse_ratio: float = 0.5,
        bsa_backend: str = "flex",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(
            dim, num_heads, eps,
            bsa_enable=bsa_enable,
            bsa_block_size=bsa_block_size,
            bsa_sparse_ratio=bsa_sparse_ratio,
            bsa_backend=bsa_backend,
        )
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, video_shape: Optional[Tuple[int, int, int]] = None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, video_shape=video_shape))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


def wantodance_torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)
    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = wantodance_torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class WanToDanceInjector(nn.Module):
    def __init__(self, all_modules, all_modules_names, dim=2048, num_heads=32, inject_layer=[0, 27]):
        super().__init__()
        self.injected_block_id = {}
        injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'root.transformer_blocks.{inject_id}' == mod_name:
                        self.injected_block_id[inject_id] = injector_id
                        injector_id += 1

        self.injector = nn.ModuleList(
            [
                CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )


class WanModel(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
        bsa_enable: bool = False,
        bsa_block_size: Tuple[int, int, int] = (2, 2, 2),
        bsa_sparse_ratio: float = 0.5,
        bsa_backend: str = "flex",
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(
                has_image_input, dim, num_heads, ffn_dim, eps,
                bsa_enable=bsa_enable,
                bsa_block_size=bsa_block_size,
                bsa_sparse_ratio=bsa_sparse_ratio,
                bsa_backend=bsa_backend,
            )
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        if wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            end = int(22350 / 8 + 0.5) # 149f * 30fps * 5s = 22350
            self.freqs = precompute_freqs_cis_3d(head_dim, end=end)
        else:
            self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        self.prepare_wantodance(in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
                                wantodance_enable_music_inject, wantodance_music_inject_layers, wantodance_enable_refimage, wantodance_enable_refface,
                                wantodance_enable_global, wantodance_enable_dynamicfps, wantodance_enable_unimodel)

    def prepare_wantodance(
        self,
        in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        if wantodance_enable_music_inject:
            all_modules, all_modules_names = wantodance_torch_dfs(self.blocks, parent_name="root.transformer_blocks")
            self.music_injector = WanToDanceInjector(all_modules, all_modules_names, dim=dim, num_heads=num_heads, inject_layer=wantodance_music_inject_layers)
        if wantodance_enable_refimage:
            self.img_emb_refimage = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_refface:
            self.img_emb_refface = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_global or wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            music_feature_dim = 35
            ff_size = 1024
            dropout = 0.1
            latent_dim = 256
            nhead = 4
            activation = F.gelu
            rotary = WanToDanceRotaryEmbedding(dim=latent_dim)
            self.music_projection = nn.Linear(music_feature_dim, latent_dim)
            self.music_encoder = nn.Sequential()
            for _ in range(2):
                self.music_encoder.append(
                    WanToDanceMusicEncoderLayer(
                        d_model=latent_dim,
                        nhead=nhead,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=rotary,
                        device='cuda',
                    )
                )
        if wantodance_enable_unimodel:
            self.patch_embedding_global = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if wantodance_enable_unimodel:
            self.head_global = Head(dim, out_dim, patch_size, eps)
        self.wantodance_enable_music_inject = wantodance_enable_music_inject
        self.wantodance_enable_refimage = wantodance_enable_refimage
        self.wantodance_enable_refface = wantodance_enable_refface
        self.wantodance_enable_global = wantodance_enable_global
        self.wantodance_enable_dynamicfps = wantodance_enable_dynamicfps
        self.wantodance_enable_unimodel = wantodance_enable_unimodel

    def wantodance_after_transformer_block(self, block_idx, hidden_states):
        if self.wantodance_enable_music_inject:
            if block_idx in self.music_injector.injected_block_id.keys():
                audio_attn_id = self.music_injector.injected_block_id[block_idx]
                audio_emb = self.merged_audio_emb  # b f n c
                num_frames = audio_emb.shape[1]
                input_hidden_states = hidden_states.clone()  # b (f h w) c
                input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
                attn_hidden_states = self.music_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
                audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
                attn_audio_emb = audio_emb
                residual_out = self.music_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
                residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
                hidden_states = hidden_states + residual_out
        return hidden_states

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None, enable_wantodance_global=False):
        if enable_wantodance_global:
            x = self.patch_embedding_global(x)
        else:
            x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs,
                    video_shape=(f, h, w),
                )
            else:
                x = block(x, context, t_mod, freqs, video_shape=(f, h, w))

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
