# Remote H20 Wan2.2-TI2V Inference

This note records the working setup on the H20 server:

```bash
ssh -p 18334 root@region-42.seetacloud.com
```

## Locations

Code:

```text
/root/autodl-tmp/DiffSynth-Studio
```

Python environment:

```text
/root/autodl-tmp/diffsynth-venv
```

Model weights:

```text
/root/autodl-tmp/models
```

Key downloaded files:

```text
/root/autodl-tmp/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors
/root/autodl-tmp/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors
/root/autodl-tmp/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors
/root/autodl-tmp/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
/root/autodl-tmp/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
/root/autodl-tmp/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
```

Test image:

```text
/root/autodl-tmp/DiffSynth-Studio/画面中的女生微微转动头面向镜头, 眨了眨眼.jpg
```

Generated test video:

```text
/root/autodl-tmp/outputs/wan22_ti2v_test.mp4
```

The same video was also copied back to the local repo root:

```text
wan22_ti2v_test.mp4
```

## Environment

The remote environment uses the AutoDL network acceleration script and keeps temporary files, model cache, and outputs on `/root/autodl-tmp`.

Verified runtime:

```text
Python 3.12
torch 2.12.0+cu130
GPU: NVIDIA H20
CUDA available: true
```

## Run Inference

SSH into the server, then run:

```bash
WORK=/root/autodl-tmp
cd "$WORK/DiffSynth-Studio"

export TMPDIR=$WORK/tmp
export PIP_CACHE_DIR=$WORK/cache/pip
export MODELSCOPE_CACHE=$WORK/cache/modelscope
export MODELSCOPE_HOME=$WORK/cache/modelscope
export HF_HOME=$WORK/cache/huggingface
export TORCH_HOME=$WORK/cache/torch
export DIFFSYNTH_MODEL_BASE_PATH=$WORK/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /etc/network_turbo 2>/dev/null || true

"$WORK/diffsynth-venv/bin/python" examples/wanvideo/model_inference/run_local_test_ti2v.py \
  --image "画面中的女生微微转动头面向镜头, 眨了眨眼.jpg" \
  --output "$WORK/outputs/wan22_ti2v_test.mp4" \
  --height 832 \
  --width 480 \
  --frames 49 \
  --steps 12 \
  --seed 0
```

The script uses the input image filename stem as the prompt.

## Validate Output

```bash
WORK=/root/autodl-tmp
"$WORK/diffsynth-venv/bin/python" - <<'PY'
from pathlib import Path
import imageio.v3 as iio

video = Path("/root/autodl-tmp/outputs/wan22_ti2v_test.mp4")
print("exists:", video.exists())
print("size:", video.stat().st_size)
print("meta:", iio.immeta(video))
frame0 = iio.imread(video, index=0)
print("frame0:", frame0.shape, int(frame0.min()), int(frame0.max()), float(frame0.mean()))
PY
```

Expected result: the file exists, has nonzero size, reports H.264 video metadata, and the first frame has nonblank pixel values.

## Block Sparse Attention (BSA) Experiments

BSA is a drop-in replacement for `WanModel.SelfAttention.attn` that drops the
`sparse_ratio` fraction of least-similar key blocks per query block.  The
implementation is in `diffsynth/models/wan_video_dit.py` (`BlockSparseAttention`
class + `bsa_enable=False` default keeps the dense path unchanged), the
pipeline plumbing is `model_fn_wan_video`'s block loop in
`diffsynth/pipelines/wan_video.py` which now forwards `video_shape=(f,h,w)`.

### Run a single sparse_ratio

```bash
WORK=/root/autodl-tmp
cd "$WORK/DiffSynth-Studio"
export TMPDIR=$WORK/tmp PIP_CACHE_DIR=$WORK/cache/pip MODELSCOPE_CACHE=$WORK/cache/modelscope MODELSCOPE_HOME=$WORK/cache/modelscope HF_HOME=$WORK/cache/huggingface TORCH_HOME=$WORK/cache/torch DIFFSYNTH_MODEL_BASE_PATH=$WORK/models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /etc/network_turbo 2>/dev/null || true

"$WORK/diffsynth-venv/bin/python" examples/wanvideo/model_inference/run_bsa_test_ti2v.py \
  --image "画面中的女生微微转动头面向镜头, 眨了眨眼.jpg" \
  --output "$WORK/outputs/bsa/sparse_0_50.mp4" \
  --height 832 --width 480 --frames 49 --steps 12 --seed 0 \
  --sparse-ratio 0.50 --block-size 2,4,4 --bsa-backend flex
```

CLI flags:
- `--sparse-ratio` — drop fraction (`0.0` = dense baseline, `0.5` = drop half, `0.9` = drop 90%)
- `--block-size BT,BH,BW` — 3D block dimensions (default `2,4,4`; the spec also evaluates `3,2,3`)
- `--bsa-backend {flex,sdpa_chunked}` — `flex` is recommended on CUDA (PyTorch ≥ 2.5), `sdpa_chunked` is the universal fallback (slower due to Python loop)
- `--dump-attention-png PATH` — capture block 0's attend pattern from the LAST real generation step and save as PNG (useful at high sparsity to see what the model actually selected)

### Sweep all sparse ratios (the experiment in this branch)

```bash
WORK=/root/autodl-tmp
cd "$WORK/DiffSynth-Studio"
export TMPDIR=$WORK/tmp PIP_CACHE_DIR=$WORK/cache/pip MODELSCOPE_CACHE=$WORK/cache/modelscope MODELSCOPE_HOME=$WORK/cache/modelscope HF_HOME=$WORK/cache/huggingface TORCH_HOME=$WORK/cache/torch DIFFSYNTH_MODEL_BASE_PATH=$WORK/models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /etc/network_turbo 2>/dev/null || true
PY="$WORK/diffsynth-venv/bin/python"
mkdir -p "$WORK/outputs/bsa"

# Dense baseline
"$PY" examples/wanvideo/model_inference/run_bsa_test_ti2v.py \
  --image "画面中的女生微微转动头面向镜头, 眨了眨眼.jpg" \
  --output "$WORK/outputs/bsa/sparse_00.mp4" \
  --height 832 --width 480 --frames 49 --steps 12 --seed 0 --sparse-ratio 0.0

# Increasing block-drop fraction
for R in 0.25 0.50 0.75 0.90; do
  TAG=$(echo $R | tr . _)
  EXTRA=()
  if [ "$R" = "0.90" ]; then
    EXTRA=(--dump-attention-png "tests/bsa_viz/wan_bsa_real_inference_90pct_block.png")
  fi
  "$PY" examples/wanvideo/model_inference/run_bsa_test_ti2v.py \
    --image "画面中的女生微微转动头面向镜头, 眨了眨眼.jpg" \
    --output "$WORK/outputs/bsa/sparse_$TAG.mp4" \
    --height 832 --width 480 --frames 49 --steps 12 --seed 0 \
    --sparse-ratio $R --block-size 2,4,4 --bsa-backend flex "${EXTRA[@]}"
done
```

### Block-size variant (3,2,3) (spec default)

The spec proposed `(bt, bh, bw) = (3, 2, 3)`, which for 832x480/49 frames maps
to latent grid `(13, 26, 15)` after VAE + patch_size `(1, 2, 2)` — every axis
divides cleanly.  Swap `--block-size 2,4,4` for `--block-size 3,2,3` to
reproduce that variant:

```bash
"$PY" examples/wanvideo/model_inference/run_bsa_test_ti2v.py \
  --image "画面中的女生微微转动头面向镜头, 眨了眨眼.jpg" \
  --output "$WORK/outputs/bsa/sparse_0_50_block_3_2_3.mp4" \
  --height 832 --width 480 --frames 49 --steps 12 --seed 0 \
  --sparse-ratio 0.5 --block-size 3,2,3 --bsa-backend flex
```

### Validate the BSA implementation itself

The 28 unit tests in `tests/` cover both backends (sdpa_chunked + flex),
verify the boundary-block weighted-mean is unbiased, and assert that BSA
forward output matches a dense+lifted-mask SDPA reference within `1e-5`
(`fp32`).  On H20 with PyTorch 2.12+cu130 + `flex_attention` available, all 28
tests pass (none skipped):

```bash
"$WORK/diffsynth-venv/bin/python" -m pytest tests/ -v
```

### Attention map visualisation

For a "more realistic" attention-map preview at 80% sparsity with two
production-scale latent shapes, run:

```bash
"$WORK/diffsynth-venv/bin/python" tests/test_wan_bsa_realistic_viz.py \
  --out-dir tests/bsa_viz --device cuda --seed 2026
```

Outputs:
- `tests/bsa_viz/wan_bsa_realistic_80pct_realistic_13_52_30.png` — synthetic correlated Q/K at the actual 832x480/49 latent (13x52x30 with block `2,4,4`) and 80% drop, showing spatial-locality bias the BSA top-K naturally produces.
- `tests/bsa_viz/wan_bsa_realistic_80pct_prod_21_26_15.png` — same recipe at the spec's `(21, 26, 15)` / block `(3, 2, 3)` shape.
- `tests/bsa_viz/wan_bsa_real_inference_90pct_block.png` — block 0's attend pattern captured during the real `--sparse-ratio 0.9` inference run; shows actual model-selected sparsity at the real latent shape `(13, 26, 15)`.

Yellow = block attended, purple = block dropped.  Each query block always
drops exactly `floor(num_blocks * sparse_ratio)` keys, so every row in the
image has the same yellow count.

### Outputs produced by this branch

- `outputs/bsa/sparse_00.mp4` — dense baseline
- `outputs/bsa/sparse_0_25.mp4` — 25% drop
- `outputs/bsa/sparse_0_50.mp4` — 50% drop
- `outputs/bsa/sparse_0_75.mp4` — 75% drop
- `outputs/bsa/sparse_0_90.mp4` — 90% drop (+ `wan_bsa_real_inference_90pct_block.png` for the matching attention map)

All five run with the same seed (`0`), prompt (filename stem of the input
image), and inference steps (`12`), so degradation as `sparse_ratio` grows is
visible side-by-side.
