import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class DMDLoRACheckpointManager:
    def __init__(self, output_path):
        self.output_path = Path(output_path)

    @staticmethod
    def prefixed_lora_state_dict(state_dict, prefix):
        return {
            key[len(prefix):]: value.detach().cpu()
            for key, value in state_dict.items()
            if key.startswith(prefix) and "lora_" in key
        }

    @staticmethod
    def final_student_lora_state_dict(state_dict, student_prefix="student_dit."):
        return DMDLoRACheckpointManager.prefixed_lora_state_dict(state_dict, student_prefix)

    @staticmethod
    def _save_safetensors(state_dict, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(dict(state_dict), str(path))

    def save_training_checkpoint(
        self,
        accelerator,
        model,
        optimizer_student,
        optimizer_fake_score,
        scheduler_student,
        scheduler_fake_score,
        outer_step,
        dmd_config,
    ):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return None
        unwrapped = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)
        checkpoint_dir = self.output_path / f"checkpoint-{outer_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_safetensors(
            self.prefixed_lora_state_dict(state_dict, "student_dit."),
            checkpoint_dir / "student_lora.safetensors",
        )
        self._save_safetensors(
            self.prefixed_lora_state_dict(state_dict, "fake_score_dit."),
            checkpoint_dir / "fake_score_lora.safetensors",
        )
        torch.save(optimizer_student.state_dict(), checkpoint_dir / "student_optimizer.pt")
        torch.save(optimizer_fake_score.state_dict(), checkpoint_dir / "fake_score_optimizer.pt")
        torch.save(scheduler_student.state_dict(), checkpoint_dir / "student_scheduler.pt")
        torch.save(scheduler_fake_score.state_dict(), checkpoint_dir / "fake_score_scheduler.pt")
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(
                {
                    "outer_step": int(outer_step),
                    "dmd_config": dmd_config,
                    "model_class": unwrapped.__class__.__name__,
                    "rng_state": {
                        "torch": torch.get_rng_state().tolist(),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return checkpoint_dir

    def save_final_student_lora(self, accelerator, model, student_prefix="student_dit."):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if not accelerator.is_main_process:
            return None
        self.output_path.mkdir(parents=True, exist_ok=True)
        exported = self.final_student_lora_state_dict(state_dict, student_prefix=student_prefix)
        path = self.output_path / "student_lora.safetensors"
        self._save_safetensors(exported, path)
        return path

    def load_training_checkpoint(
        self,
        checkpoint_dir,
        model,
        optimizer_student=None,
        optimizer_fake_score=None,
        scheduler_student=None,
        scheduler_fake_score=None,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        student_lora = load_file(str(checkpoint_dir / "student_lora.safetensors"))
        fake_lora = load_file(str(checkpoint_dir / "fake_score_lora.safetensors"))
        model.load_state_dict({f"student_dit.{key}": value for key, value in student_lora.items()}, strict=False)
        model.load_state_dict({f"fake_score_dit.{key}": value for key, value in fake_lora.items()}, strict=False)
        if optimizer_student is not None and (checkpoint_dir / "student_optimizer.pt").exists():
            optimizer_student.load_state_dict(torch.load(checkpoint_dir / "student_optimizer.pt", map_location="cpu"))
        if optimizer_fake_score is not None and (checkpoint_dir / "fake_score_optimizer.pt").exists():
            optimizer_fake_score.load_state_dict(torch.load(checkpoint_dir / "fake_score_optimizer.pt", map_location="cpu"))
        if scheduler_student is not None and (checkpoint_dir / "student_scheduler.pt").exists():
            scheduler_student.load_state_dict(torch.load(checkpoint_dir / "student_scheduler.pt", map_location="cpu"))
        if scheduler_fake_score is not None and (checkpoint_dir / "fake_score_scheduler.pt").exists():
            scheduler_fake_score.load_state_dict(torch.load(checkpoint_dir / "fake_score_scheduler.pt", map_location="cpu"))
        state_path = checkpoint_dir / "trainer_state.json"
        if state_path.exists():
            return json.loads(state_path.read_text())
        return {}

