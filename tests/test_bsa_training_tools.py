import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

from diffsynth.diffusion.training_metrics import TrainingMetricsWriter, compute_wan_video_tokens
from diffsynth.models.wan_bsa import (
    configure_wan_bsa,
    parse_bsa_block_size,
    set_wan_bsa_sparse_ratio,
)
from diffsynth.models.wan_video_dit import AttentionModule, BlockSparseAttention, WanModel
from examples.wanvideo.model_inference.run_bsa_test_ti2v import parse_args as parse_bsa_inference_args
from examples.wanvideo.model_training.train import bsa_sparse_ratio_for_step, build_wan_special_operator_map, wan_parser
from examples.wanvideo.model_training.prepare_hq_vsr_smoke_dataset import select_mp4_members


def _tiny_wan_model():
    return WanModel(
        dim=16,
        in_dim=4,
        ffn_dim=32,
        out_dim=4,
        text_dim=8,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=2,
        has_image_input=False,
    )


def test_configure_wan_bsa_replaces_every_self_attention_and_updates_ratio():
    model = _tiny_wan_model()

    replaced = configure_wan_bsa(
        model,
        enable=True,
        block_size=parse_bsa_block_size("2,2,3"),
        sparse_ratio=0.90,
        backend="sdpa_chunked",
        chunk_size=7,
    )

    assert replaced == 2
    for block in model.blocks:
        attn = block.self_attn.attn
        assert isinstance(attn, BlockSparseAttention)
        assert attn.block_size == (2, 2, 3)
        assert attn.sparse_ratio == 0.90
        assert attn.backend == "sdpa_chunked"
        assert attn.chunk_size == 7

    updated = set_wan_bsa_sparse_ratio(model, 0.95)

    assert updated == 2
    assert [block.self_attn.attn.sparse_ratio for block in model.blocks] == [0.95, 0.95]

    restored = configure_wan_bsa(model, enable=False)

    assert restored == 2
    for block in model.blocks:
        assert isinstance(block.self_attn.attn, AttentionModule)
        assert block.self_attn.bsa_enable is False


def test_training_metrics_writer_records_loss_and_video_token_throughput(tmp_path):
    tokens_per_video = compute_wan_video_tokens(
        height=480,
        width=832,
        num_frames=81,
        vae_upsampling_factor=16,
        patch_size=(1, 2, 2),
    )
    assert tokens_per_video == 8190

    writer = TrainingMetricsWriter(
        output_path=tmp_path,
        enabled=True,
        tokens_per_sample=tokens_per_video,
        log_steps=1,
        use_tensorboard=False,
    )
    writer.log_step(
        step=1,
        loss=torch.tensor(2.0),
        learning_rate=1e-4,
        samples=2,
        bsa_sparse_ratio=0.90,
        elapsed_seconds=10.0,
        step_seconds=10.0,
    )
    writer.close()

    csv_path = tmp_path / "training_metrics.csv"
    jsonl_path = tmp_path / "training_metrics.jsonl"
    html_path = tmp_path / "training_metrics.html"
    assert csv_path.exists()
    assert jsonl_path.exists()
    assert html_path.exists()

    csv_rows = list(csv.DictReader(csv_path.open()))
    json_row = json.loads(jsonl_path.read_text().splitlines()[0])
    assert csv_rows[0]["step"] == "1"
    assert float(csv_rows[0]["loss"]) == 2.0
    assert int(csv_rows[0]["tokens_this_step"]) == 16380
    assert int(csv_rows[0]["total_tokens"]) == 16380
    assert float(csv_rows[0]["tokens_per_hour"]) == 5896800.0
    assert float(csv_rows[0]["videos_per_day"]) == 17280.0
    assert json_row["bsa_sparse_ratio"] == 0.90
    assert "Loss" in html_path.read_text()


def test_wan_training_parser_accepts_bsa_logging_and_dataloader_flags():
    args = wan_parser().parse_args([
        "--dataset_base_path", "data",
        "--bsa_enable",
        "--bsa_block_size", "2,2,3",
        "--bsa_backend", "sdpa_chunked",
        "--bsa_sparse_ratio_start", "0.90",
        "--bsa_sparse_ratio", "0.95",
        "--bsa_sparse_ratio_warmup_steps", "500",
        "--bsa_sdpa_chunk_size", "64",
        "--log_steps", "2",
        "--enable_tensorboard",
        "--dataset_pin_memory",
        "--dataset_persistent_workers",
        "--dataset_prefetch_factor", "4",
        "--optimizer_fused",
        "--disable_common_file_redirect",
    ])

    assert args.bsa_enable is True
    assert args.bsa_block_size == "2,2,3"
    assert args.bsa_backend == "sdpa_chunked"
    assert args.bsa_sparse_ratio_start == 0.90
    assert args.bsa_sparse_ratio == 0.95
    assert args.bsa_sparse_ratio_warmup_steps == 500
    assert args.bsa_sdpa_chunk_size == 64
    assert args.log_steps == 2
    assert args.enable_tensorboard is True
    assert args.dataset_pin_memory is True
    assert args.dataset_persistent_workers is True
    assert args.dataset_prefetch_factor == 4
    assert args.optimizer_fused is True
    assert args.disable_common_file_redirect is True


def test_bsa_inference_parser_accepts_lora_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_bsa_test_ti2v.py",
            "--image",
            "input.png",
            "--output",
            "output.mp4",
            "--lora-checkpoint",
            "models/train/step-100.safetensors",
            "--lora-alpha",
            "0.75",
        ],
    )

    args = parse_bsa_inference_args()

    assert args.lora_checkpoint == "models/train/step-100.safetensors"
    assert args.lora_alpha == 0.75


def test_bsa_inference_script_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "examples/wanvideo/model_inference/run_bsa_test_ti2v.py",
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--lora-checkpoint" in result.stdout
    assert "--sparse-ratio" in result.stdout


def test_bsa_sparse_ratio_for_step_linearly_warms_to_target():
    assert bsa_sparse_ratio_for_step(0, start=0.90, target=0.95, warmup_steps=10) == 0.90
    assert bsa_sparse_ratio_for_step(5, start=0.90, target=0.95, warmup_steps=10) == 0.925
    assert bsa_sparse_ratio_for_step(10, start=0.90, target=0.95, warmup_steps=10) == 0.95
    assert bsa_sparse_ratio_for_step(20, start=0.90, target=0.95, warmup_steps=10) == 0.95
    assert bsa_sparse_ratio_for_step(1, start=None, target=0.95, warmup_steps=10) == 0.95


def test_select_mp4_members_from_hq_vsr_zip_orders_by_larger_files_first(tmp_path):
    zip_path = tmp_path / "videos.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("HQ-VSR/small.mp4", b"1" * 10)
        zf.writestr("HQ-VSR/readme.txt", "skip")
        zf.writestr("HQ-VSR/large.mp4", b"1" * 100)
        zf.writestr("__MACOSX/ignored.mp4", b"1" * 1000)

    selected = select_mp4_members(zip_path, limit=2)

    assert [info.filename for info in selected] == ["HQ-VSR/large.mp4", "HQ-VSR/small.mp4"]


def test_wan_special_operator_map_keeps_audio_dependency_optional():
    video_only = build_wan_special_operator_map(
        dataset_base_path="data",
        data_file_keys="video",
        extra_inputs="input_image",
        num_frames=81,
        framewise_decoding=False,
    )
    assert "input_audio" not in video_only
