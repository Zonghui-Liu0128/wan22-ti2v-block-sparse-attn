from types import SimpleNamespace
import json

import torch
from safetensors.torch import load_file

from diffsynth.diffusion.dmd_checkpoint import DMDLoRACheckpointManager
from diffsynth.diffusion.dmd_loss import (
    flow_prediction_to_x0,
    parse_dmd_denoising_steps,
    replace_first_frame_latents,
    stop_gradient_mse_with_manual_gradient,
)
from diffsynth.diffusion.dmd_runner import launch_dmd_lora_training_task
from diffsynth.diffusion.training_metrics import compute_wan_video_tokens
from diffsynth.models.wan_bsa import iter_wan_bsa_modules
from diffsynth.models.wan_video_dit import WanModel
from examples.wanvideo.model_training.wan_dmd_lora_training import WanDMDLoRATrainingModule


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


def test_dmd_loss_helpers_parse_steps_convert_x0_and_keep_first_frame():
    assert parse_dmd_denoising_steps("1000,750,500,250") == (1000, 750, 500, 250)

    noisy = torch.full((1, 1, 2, 2, 2), 4.0)
    flow = torch.full_like(noisy, 2.0)
    timestep = torch.tensor([500.0])

    x0 = flow_prediction_to_x0(noisy, flow, timestep)

    assert torch.allclose(x0, torch.full_like(noisy, 3.0))

    first_frame = torch.full((1, 1, 1, 2, 2), 9.0)
    replaced = replace_first_frame_latents(noisy.clone(), first_frame)

    assert torch.equal(replaced[:, :, 0:1], first_frame)
    assert torch.equal(replaced[:, :, 1:], noisy[:, :, 1:])


def test_stop_gradient_mse_with_manual_gradient_backprops_expected_direction():
    generated = torch.tensor([1.0, 2.0], requires_grad=True)
    manual_grad = torch.tensor([0.25, -0.5])

    loss = stop_gradient_mse_with_manual_gradient(generated, manual_grad)
    loss.backward()

    assert torch.allclose(generated.grad, manual_grad / generated.numel())


def test_wan_dmd_lora_prebuilt_components_freeze_roles_and_apply_bsa_only_to_student():
    pipe = SimpleNamespace(
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
        scheduler=SimpleNamespace(set_timesteps=lambda *args, **kwargs: None),
        vae=SimpleNamespace(upsampling_factor=16),
        dit=SimpleNamespace(patch_size=(1, 2, 2)),
    )

    module = WanDMDLoRATrainingModule.from_prebuilt_components(
        pipe=pipe,
        teacher_dit=_tiny_wan_model(),
        student_dit=_tiny_wan_model(),
        fake_score_dit=_tiny_wan_model(),
        lora_target_modules="q,k,v,o",
        student_lora_rank=2,
        fake_score_lora_rank=2,
        dmd_denoising_steps="1000,750,500,250",
        bsa_enable=True,
        bsa_block_size="2,2,2",
        bsa_sparse_ratio=0.85,
        bsa_backend="sdpa_chunked",
        bsa_sdpa_chunk_size=8,
        base_teacher_smoke=True,
    )

    assert module.teacher_dit is not module.student_dit
    assert module.teacher_dit is not module.fake_score_dit
    assert module.student_dit is not module.fake_score_dit
    assert all(not param.requires_grad for param in module.teacher_dit.parameters())

    for name, param in module.student_dit.named_parameters():
        assert param.requires_grad is ("lora_" in name), name
    for name, param in module.fake_score_dit.named_parameters():
        assert param.requires_grad is ("lora_" in name), name

    assert len(list(iter_wan_bsa_modules(module.student_dit))) == 2
    assert list(iter_wan_bsa_modules(module.teacher_dit)) == []
    assert list(iter_wan_bsa_modules(module.fake_score_dit)) == []
    assert module.dmd_denoising_steps == (1000, 750, 500, 250)
    assert module.current_bsa_sparse_ratio == 0.85
    assert module.base_teacher_smoke is True


class _FakeAccelerator:
    is_main_process = True

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, model):
        return model

    def get_state_dict(self, model):
        return model.state_dict()


class _CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.student_dit = torch.nn.Linear(2, 2, bias=False)
        self.fake_score_dit = torch.nn.Linear(2, 2, bias=False)
        self.teacher_dit = torch.nn.Linear(2, 2, bias=False)
        self.student_dit.register_parameter(
            "adapter_lora_A_default_weight",
            torch.nn.Parameter(torch.ones(1, 1)),
        )
        self.fake_score_dit.register_parameter(
            "adapter_lora_A_default_weight",
            torch.nn.Parameter(torch.ones(1, 1) * 2),
        )


def test_dmd_checkpoint_writes_safetensors_for_lora_roles(tmp_path):
    model = _CheckpointModel()
    optimizer_student = torch.optim.AdamW(model.student_dit.parameters(), lr=5e-7)
    optimizer_fake = torch.optim.AdamW(model.fake_score_dit.parameters(), lr=1e-7)
    scheduler_student = torch.optim.lr_scheduler.ConstantLR(optimizer_student)
    scheduler_fake = torch.optim.lr_scheduler.ConstantLR(optimizer_fake)

    manager = DMDLoRACheckpointManager(tmp_path)
    manager.save_training_checkpoint(
        _FakeAccelerator(),
        model,
        optimizer_student,
        optimizer_fake,
        scheduler_student,
        scheduler_fake,
        outer_step=5,
        dmd_config={"dmd_denoising_steps": "1000,750,500,250"},
    )
    final_path = manager.save_final_student_lora(_FakeAccelerator(), model)

    checkpoint_dir = tmp_path / "checkpoint-5"
    assert (checkpoint_dir / "student_lora.safetensors").exists()
    assert (checkpoint_dir / "fake_score_lora.safetensors").exists()
    assert (checkpoint_dir / "student_optimizer.pt").exists()
    assert (checkpoint_dir / "fake_score_optimizer.pt").exists()
    assert final_path.name == "student_lora.safetensors"

    final_state = load_file(final_path)
    assert final_state
    assert all("fake_score" not in key for key in final_state)


class _TinyDataset(torch.utils.data.Dataset):
    load_from_cache = False

    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {"x": torch.tensor(1.0)}


class _TinyDMDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.student = torch.nn.Parameter(torch.tensor(1.0))
        self.fake = torch.nn.Parameter(torch.tensor(1.0))
        self.student_steps = []
        self.fake_steps = []
        self.pipe = SimpleNamespace(
            device=torch.device("cpu"),
            vae=SimpleNamespace(upsampling_factor=16),
            dit=SimpleNamespace(patch_size=(1, 2, 2)),
        )

    def student_trainable_modules(self):
        return [self.student]

    def fake_score_trainable_modules(self):
        return [self.fake]

    def compute_student_dmd_loss(self, batch, outer_step):
        self.student_steps.append(int(outer_step))
        return self.student * batch["x"], {
            "dmd_gradient_norm": torch.tensor(1.0),
            "dit_forward_count_student": 1,
            "dit_forward_count_fake": 1,
            "dit_forward_count_teacher": 2,
        }

    def compute_fake_score_loss(self, batch, outer_step):
        self.fake_steps.append(int(outer_step))
        return self.fake * batch["x"], {
            "fake_score_loss": self.fake.detach(),
            "dit_forward_count_fake": 1,
        }


class _NoopLogger:
    def __init__(self, output_path):
        self.output_path = output_path


class _RunnerAccelerator:
    is_main_process = True
    num_processes = 1
    device = torch.device("cpu")

    def prepare(self, *items):
        return items

    def backward(self, loss):
        loss.backward()

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, model):
        return model

    def get_state_dict(self, model):
        return model.state_dict()


def test_launch_dmd_lora_training_task_uses_ttur_schedule(tmp_path):
    args = SimpleNamespace(
        student_learning_rate=5e-7,
        fake_score_learning_rate=1e-7,
        weight_decay=0.0,
        student_beta1=0.9,
        student_beta2=0.999,
        fake_score_beta1=0.9,
        fake_score_beta2=0.999,
        dataset_num_workers=0,
        dataset_pin_memory=False,
        dataset_persistent_workers=False,
        dataset_prefetch_factor=None,
        max_train_steps=6,
        num_epochs=1,
        save_steps=None,
        fake_score_updates_per_generator_update=5,
        disable_training_metrics=False,
        log_steps=1,
        enable_tensorboard=False,
        height=256,
        width=256,
        num_frames=81,
        dmd_denoising_steps="1000,750,500,250",
        real_guidance_scale=6.0,
        fake_guidance_scale=0.0,
    )
    model = _TinyDMDModel()

    launch_dmd_lora_training_task(
        _RunnerAccelerator(),
        _TinyDataset(),
        model,
        _NoopLogger(tmp_path),
        args=args,
    )

    assert model.student_steps == [0, 5]
    assert model.fake_steps == [0, 1, 2, 3, 4, 5]
    assert compute_wan_video_tokens(256, 256, 81, 16, (1, 2, 2)) == 1344

    rows = [json.loads(line) for line in (tmp_path / "training_metrics.jsonl").read_text().splitlines()]
    assert rows[0]["student_param_delta_norm"] > 0
    assert rows[1]["student_param_delta_norm"] == 0.0
    assert rows[-1]["student_param_delta_norm"] > 0
    assert all(row["fake_score_param_delta_norm"] > 0 for row in rows)
    assert all(row["teacher_param_delta_norm"] == 0.0 for row in rows)
    assert "peak_memory_gb" in rows[0]
