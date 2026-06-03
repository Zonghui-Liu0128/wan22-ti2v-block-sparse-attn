import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import accelerate
import yaml
import torch

from diffsynth.diffusion.runner import launch_training_task
from diffsynth.diffusion.loss import FlowMatchSFTLoss
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.diffusion.training_metrics import TrainingMetricsWriter, compute_wan_video_tokens
from diffsynth.models.wan_bsa import (
    configure_wan_bsa,
    parse_bsa_block_size,
    set_wan_bsa_sparse_ratio,
)
from diffsynth.models.wan_video_dit import AttentionModule, BlockSparseAttention, WanModel
from examples.wanvideo.model_inference.run_bsa_test_ti2v import parse_args as parse_bsa_inference_args
from examples.wanvideo.model_training.train import (
    WanTrainingModule,
    bsa_sparse_ratio_for_step,
    build_internal_wan_model_configs,
    build_wan_special_operator_map,
    wan_parser,
)
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
        "--max_train_steps", "20",
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
    assert args.max_train_steps == 20
    assert args.disable_common_file_redirect is True


def test_internal_model_path_loader_builds_wan22_ti2v_configs_without_json(tmp_path):
    model_dir = tmp_path / "Wan2.2-TI2V-5B"
    model_dir.mkdir()
    (model_dir / "diffusion_pytorch_model-00001-of-00002.safetensors").touch()
    (model_dir / "diffusion_pytorch_model-00002-of-00002.safetensors").touch()

    model_configs, tokenizer_config = build_internal_wan_model_configs(str(model_dir))

    assert len(model_configs) == 3
    assert sorted(Path(p).name for p in model_configs[0].path) == [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    assert Path(model_configs[1].path).name == "models_t5_umt5-xxl-enc-bf16.pth"
    assert Path(model_configs[2].path).name == "Wan2.2_VAE.pth"
    assert tokenizer_config.path.endswith("google/umt5-xxl")

    _, explicit_tokenizer = build_internal_wan_model_configs(
        str(model_dir),
        tokenizer_path="/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
    )

    assert explicit_tokenizer.path == "/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

    dit_only_configs, dit_only_tokenizer = build_internal_wan_model_configs(str(model_dir), dit_only=True)

    assert len(dit_only_configs) == 1
    assert sorted(Path(p).name for p in dit_only_configs[0].path) == [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    assert dit_only_tokenizer is None


class _TinyDataset(torch.utils.data.Dataset):
    load_from_cache = False

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {"x": torch.tensor(1.0)}


class _TinyTrainingModule(DiffusionTrainingModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.pipe = SimpleNamespace(
            device=torch.device("cpu"),
            vae=None,
            dit=SimpleNamespace(patch_size=(1, 2, 2)),
        )

    def forward(self, data, inputs=None):
        return self.weight * data["x"]


class _StepCountingLogger:
    def __init__(self, output_path):
        self.output_path = output_path
        self.num_steps = 0
        self.epoch_saves = 0

    def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
        self.num_steps += 1

    def on_epoch_end(self, accelerator, model, epoch_id):
        self.epoch_saves += 1

    def on_training_end(self, accelerator, model, save_steps=None):
        pass


def test_launch_training_task_stops_at_max_train_steps(tmp_path):
    args = SimpleNamespace(
        learning_rate=0.01,
        weight_decay=0.0,
        dataset_num_workers=0,
        save_steps=None,
        num_epochs=5,
        max_train_steps=2,
        enable_model_cpu_offload=False,
        enable_optimizer_cpu_offload=False,
        cpu_offload_split_threshold=None,
        optimizer_fused=False,
        dataset_pin_memory=False,
        dataset_persistent_workers=False,
        dataset_prefetch_factor=None,
        height=16,
        width=16,
        num_frames=1,
        disable_training_metrics=True,
        log_steps=1,
        enable_tensorboard=False,
    )
    logger = _StepCountingLogger(tmp_path)

    launch_training_task(
        accelerate.Accelerator(),
        _TinyDataset(),
        _TinyTrainingModule(),
        logger,
        args=args,
    )

    assert logger.num_steps == 2


def test_wan_training_module_bsa_step_hook_updates_sparse_ratio_on_internal_model():
    model = WanTrainingModule.__new__(WanTrainingModule)
    model.bsa_enable = True
    model.bsa_sparse_ratio_start = 0.85
    model.bsa_sparse_ratio_target = 0.90
    model.bsa_sparse_ratio_warmup_steps = 10
    model.current_bsa_sparse_ratio = 0.85
    model.pipe = type("Pipe", (), {})()
    model.pipe.dit = _tiny_wan_model()
    model.pipe.dit2 = None
    model.bsa_block_size = parse_bsa_block_size("3,7,3")
    model.bsa_backend = "sdpa_chunked"
    model.bsa_sdpa_chunk_size = 64
    model._configure_bsa(0.85)

    model.on_train_step_start(global_step=5)

    assert model.current_bsa_sparse_ratio == 0.875
    for block in model.pipe.dit.blocks:
        attn = block.self_attn.attn
        assert isinstance(attn, BlockSparseAttention)
        assert attn.block_size == (3, 7, 3)
        assert attn.backend == "sdpa_chunked"
        assert attn.sparse_ratio == 0.875


def test_flow_match_sft_loss_keeps_first_frame_in_training_target(monkeypatch):
    calls = {}

    class Scheduler:
        timesteps = torch.arange(1000, dtype=torch.float32)

        def add_noise(self, original, noise, timestep):
            return original + noise

        def training_target(self, original, noise, timestep):
            return noise - original

        def training_weight(self, timestep):
            return torch.tensor(1.0)

    class Pipe:
        scheduler = Scheduler()
        torch_dtype = torch.float32
        device = torch.device("cpu")
        in_iteration_models = ("dit",)
        dit = object()

        def model_fn(self, **kwargs):
            calls["latents_shape"] = kwargs["latents"].shape
            return torch.zeros_like(kwargs["input_latents"])

    input_latents = torch.ones(1, 1, 3, 2, 2)
    loss = FlowMatchSFTLoss(
        Pipe(),
        input_latents=input_latents,
        first_frame_latents=torch.full((1, 1, 1, 2, 2), 7.0),
    )

    assert calls["latents_shape"] == input_latents.shape
    assert loss.ndim == 0


def test_diffusion_training_module_keeps_resume_checkpoint_api():
    assert hasattr(DiffusionTrainingModule, "resume_from_checkpoint")


def test_h100_4gpu_accelerate_config_is_deepspeed_bf16_zero2():
    config_path = Path(__file__).resolve().parents[1] / "config_wan22_5B.yaml"

    config = yaml.safe_load(config_path.read_text())

    assert config["distributed_type"] == "DEEPSPEED"
    assert config["mixed_precision"] == "bf16"
    assert config["num_processes"] == 4
    assert config["deepspeed_config"]["zero_stage"] == 2
    assert config["deepspeed_config"]["gradient_accumulation_steps"] == 1


def test_bsa_inference_parser_accepts_lora_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_bsa_test_ti2v.py",
            "--image",
            "input.png",
            "--output",
            "output.mp4",
            "--model-paths",
            "/models/Wan-AI/Wan2.2-TI2V-5B",
            "--tokenizer-path",
            "/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
            "--disable-common-file-redirect",
            "--lora-checkpoint",
            "models/train/step-100.safetensors",
            "--lora-alpha",
            "0.75",
        ],
    )

    args = parse_bsa_inference_args()

    assert args.lora_checkpoint == "models/train/step-100.safetensors"
    assert args.lora_alpha == 0.75
    assert args.model_paths == "/models/Wan-AI/Wan2.2-TI2V-5B"
    assert args.tokenizer_path.endswith("google/umt5-xxl")
    assert args.disable_common_file_redirect is True


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
    assert "--model-paths" in result.stdout
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
