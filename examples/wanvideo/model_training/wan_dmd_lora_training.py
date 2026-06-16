import glob
import os

import torch

from diffsynth.core import load_state_dict
from diffsynth.diffusion import DiffusionTrainingModule
from diffsynth.diffusion.dmd_loss import (
    compute_fake_score_x0_loss,
    compute_student_distribution_matching_loss,
    parse_dmd_denoising_steps,
    replace_first_frame_latents,
)
from diffsynth.models.wan_bsa import configure_wan_bsa, parse_bsa_block_size, set_wan_bsa_sparse_ratio
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


def _split_csv(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _bsa_sparse_ratio_for_step(step, start, target, warmup_steps):
    target = float(target)
    if start is None or warmup_steps is None or int(warmup_steps) <= 0:
        return target
    start = float(start)
    warmup_steps = int(warmup_steps)
    if step >= warmup_steps:
        return target
    return round(start + (target - start) * (float(step) / float(warmup_steps)), 12)


def _build_internal_wan_model_configs(model_paths, dit_only=False, tokenizer_path=None):
    if model_paths is None:
        return None, None
    model_root = os.path.abspath(os.path.expanduser(model_paths))
    diffusion_paths = sorted(glob.glob(os.path.join(model_root, "diffusion_pytorch_model*.safetensors")))
    model_configs = [ModelConfig(path=diffusion_paths, offload_device="cpu")]
    tokenizer_config = None
    if not dit_only:
        model_configs.extend([
            ModelConfig(path=os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"), offload_device="cpu"),
            ModelConfig(path=os.path.join(model_root, "Wan2.2_VAE.pth"), offload_device="cpu"),
        ])
        tokenizer_config = ModelConfig(
            path=os.path.join(model_root, "google", "umt5-xxl") if tokenizer_path is None else tokenizer_path,
            offload_device="cpu",
        )
    return model_configs, tokenizer_config


class WanDMDLoRATrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        teacher_model_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        teacher_lora_checkpoint=None,
        student_init_lora_checkpoint=None,
        fake_score_init_lora_checkpoint=None,
        student_lora_rank=32,
        fake_score_lora_rank=32,
        dmd_denoising_steps="1000,750,500,250",
        real_guidance_scale=6.0,
        fake_guidance_scale=0.0,
        fake_score_loss_type="x0",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        bsa_enable=True,
        bsa_block_size=(3, 7, 3),
        bsa_sparse_ratio=0.85,
        bsa_sparse_ratio_start=None,
        bsa_sparse_ratio_warmup_steps=0,
        bsa_backend="sdpa_chunked",
        bsa_sdpa_chunk_size=64,
        redirect_common_files=True,
    ):
        super().__init__()
        self.real_guidance_scale = float(real_guidance_scale)
        self.fake_guidance_scale = float(fake_guidance_scale)
        self.fake_score_loss_type = fake_score_loss_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []

        base_model_paths = model_paths
        teacher_model_paths = teacher_model_paths or model_paths
        pipe = self._load_pipe(
            base_model_paths,
            model_id_with_origin_paths,
            tokenizer_path,
            audio_processor_path,
            fp8_models,
            offload_models,
            device,
            redirect_common_files,
            dit_only=False,
        )
        teacher_pipe = self._load_pipe(
            teacher_model_paths,
            model_id_with_origin_paths if teacher_model_paths is None else None,
            tokenizer_path,
            audio_processor_path,
            fp8_models,
            offload_models,
            device,
            redirect_common_files,
            dit_only=True,
        )
        fake_pipe = self._load_pipe(
            base_model_paths,
            model_id_with_origin_paths if base_model_paths is None else None,
            tokenizer_path,
            audio_processor_path,
            fp8_models,
            offload_models,
            device,
            redirect_common_files,
            dit_only=True,
        )
        student_dit = pipe.dit
        teacher_dit = teacher_pipe.dit
        fake_score_dit = fake_pipe.dit
        self._init_from_components(
            pipe=pipe,
            teacher_dit=teacher_dit,
            student_dit=student_dit,
            fake_score_dit=fake_score_dit,
            lora_target_modules=lora_target_modules,
            teacher_lora_checkpoint=teacher_lora_checkpoint,
            student_init_lora_checkpoint=student_init_lora_checkpoint or teacher_lora_checkpoint,
            fake_score_init_lora_checkpoint=fake_score_init_lora_checkpoint or teacher_lora_checkpoint,
            student_lora_rank=student_lora_rank,
            fake_score_lora_rank=fake_score_lora_rank,
            dmd_denoising_steps=dmd_denoising_steps,
            bsa_enable=bsa_enable,
            bsa_block_size=bsa_block_size,
            bsa_sparse_ratio=bsa_sparse_ratio,
            bsa_sparse_ratio_start=bsa_sparse_ratio_start,
            bsa_sparse_ratio_warmup_steps=bsa_sparse_ratio_warmup_steps,
            bsa_backend=bsa_backend,
            bsa_sdpa_chunk_size=bsa_sdpa_chunk_size,
            base_teacher_smoke=teacher_lora_checkpoint is None,
        )

    @classmethod
    def from_prebuilt_components(
        cls,
        pipe,
        teacher_dit,
        student_dit,
        fake_score_dit,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        teacher_lora_checkpoint=None,
        student_init_lora_checkpoint=None,
        fake_score_init_lora_checkpoint=None,
        student_lora_rank=32,
        fake_score_lora_rank=32,
        dmd_denoising_steps="1000,750,500,250",
        bsa_enable=True,
        bsa_block_size=(3, 7, 3),
        bsa_sparse_ratio=0.85,
        bsa_sparse_ratio_start=None,
        bsa_sparse_ratio_warmup_steps=0,
        bsa_backend="sdpa_chunked",
        bsa_sdpa_chunk_size=64,
        base_teacher_smoke=False,
    ):
        obj = cls.__new__(cls)
        DiffusionTrainingModule.__init__(obj)
        obj.real_guidance_scale = 6.0
        obj.fake_guidance_scale = 0.0
        obj.fake_score_loss_type = "x0"
        obj.use_gradient_checkpointing = False
        obj.use_gradient_checkpointing_offload = False
        obj.extra_inputs = []
        obj._init_from_components(
            pipe=pipe,
            teacher_dit=teacher_dit,
            student_dit=student_dit,
            fake_score_dit=fake_score_dit,
            lora_target_modules=lora_target_modules,
            teacher_lora_checkpoint=teacher_lora_checkpoint,
            student_init_lora_checkpoint=student_init_lora_checkpoint,
            fake_score_init_lora_checkpoint=fake_score_init_lora_checkpoint,
            student_lora_rank=student_lora_rank,
            fake_score_lora_rank=fake_score_lora_rank,
            dmd_denoising_steps=dmd_denoising_steps,
            bsa_enable=bsa_enable,
            bsa_block_size=bsa_block_size,
            bsa_sparse_ratio=bsa_sparse_ratio,
            bsa_sparse_ratio_start=bsa_sparse_ratio_start,
            bsa_sparse_ratio_warmup_steps=bsa_sparse_ratio_warmup_steps,
            bsa_backend=bsa_backend,
            bsa_sdpa_chunk_size=bsa_sdpa_chunk_size,
            base_teacher_smoke=base_teacher_smoke,
        )
        return obj

    def _load_pipe(
        self,
        model_paths,
        model_id_with_origin_paths,
        tokenizer_path,
        audio_processor_path,
        fp8_models,
        offload_models,
        device,
        redirect_common_files,
        dit_only=False,
    ):
        if model_paths is not None and not model_paths.lstrip().startswith("["):
            model_configs, tokenizer_config = _build_internal_wan_model_configs(
                model_paths,
                dit_only=dit_only,
                tokenizer_path=tokenizer_path,
            )
        else:
            model_configs = self.parse_model_configs(
                model_paths,
                model_id_with_origin_paths,
                fp8_models=fp8_models,
                offload_models=offload_models,
                device=device,
            )
            tokenizer_config = None if dit_only else (
                ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
                if tokenizer_path is None else ModelConfig(tokenizer_path)
            )
        audio_processor_config = None if dit_only else self.parse_path_or_model_id(audio_processor_path)
        return WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            redirect_common_files=redirect_common_files,
        )

    def _init_from_components(
        self,
        pipe,
        teacher_dit,
        student_dit,
        fake_score_dit,
        lora_target_modules,
        teacher_lora_checkpoint,
        student_init_lora_checkpoint,
        fake_score_init_lora_checkpoint,
        student_lora_rank,
        fake_score_lora_rank,
        dmd_denoising_steps,
        bsa_enable,
        bsa_block_size,
        bsa_sparse_ratio,
        bsa_sparse_ratio_start,
        bsa_sparse_ratio_warmup_steps,
        bsa_backend,
        bsa_sdpa_chunk_size,
        base_teacher_smoke,
    ):
        self.pipe = pipe
        self.dmd_denoising_steps = parse_dmd_denoising_steps(dmd_denoising_steps)
        self.base_teacher_smoke = bool(base_teacher_smoke)
        self.bsa_enable = bool(bsa_enable)
        self.bsa_sparse_ratio_target = float(bsa_sparse_ratio)
        self.bsa_sparse_ratio_start = None if bsa_sparse_ratio_start is None else float(bsa_sparse_ratio_start)
        self.bsa_sparse_ratio_warmup_steps = int(bsa_sparse_ratio_warmup_steps or 0)
        self.bsa_block_size = parse_bsa_block_size(bsa_block_size)
        self.bsa_backend = bsa_backend
        self.bsa_sdpa_chunk_size = int(bsa_sdpa_chunk_size)
        self.current_bsa_sparse_ratio = None

        self.teacher_dit = self._prepare_teacher_lora(
            teacher_dit,
            lora_target_modules,
            student_lora_rank,
            teacher_lora_checkpoint,
        )
        self.student_dit = self._prepare_trainable_lora(
            student_dit,
            lora_target_modules,
            student_lora_rank,
            student_init_lora_checkpoint,
        )
        self.fake_score_dit = self._prepare_trainable_lora(
            fake_score_dit,
            lora_target_modules,
            fake_score_lora_rank,
            fake_score_init_lora_checkpoint,
        )
        self.pipe.dit = self.student_dit
        if self.bsa_enable:
            self._configure_student_bsa(
                _bsa_sparse_ratio_for_step(
                    0,
                    self.bsa_sparse_ratio_start,
                    self.bsa_sparse_ratio_target,
                    self.bsa_sparse_ratio_warmup_steps,
                )
            )
        if self.base_teacher_smoke:
            print("DMD LoRA base_teacher_smoke=true: teacher LoRA checkpoint not provided.")

    def _prepare_teacher_lora(self, model, target_modules, rank, checkpoint):
        model.requires_grad_(False)
        if checkpoint is not None:
            model = self.add_lora_to_model(
                model,
                target_modules=self.parse_lora_target_modules(model, target_modules),
                lora_rank=rank,
                upcast_dtype=getattr(self.pipe, "torch_dtype", None),
            )
            self._load_lora_checkpoint(model, checkpoint)
        model.requires_grad_(False)
        return model

    def _prepare_trainable_lora(self, model, target_modules, rank, checkpoint):
        model.requires_grad_(False)
        model = self.add_lora_to_model(
            model,
            target_modules=self.parse_lora_target_modules(model, target_modules),
            lora_rank=rank,
            upcast_dtype=getattr(self.pipe, "torch_dtype", None),
        )
        if checkpoint is not None:
            self._load_lora_checkpoint(model, checkpoint)
        self._mark_only_lora_trainable(model)
        return model

    def _load_lora_checkpoint(self, model, checkpoint):
        lora = load_state_dict(checkpoint)
        if hasattr(self.pipe, "lora_loader"):
            lora_loader = self.pipe.lora_loader(
                torch_dtype=getattr(self.pipe, "torch_dtype", torch.float32),
                device=getattr(self.pipe, "device", "cpu"),
            )
            lora = lora_loader.convert_state_dict(lora)
        lora = self.mapping_lora_state_dict(lora)
        load_result = model.load_state_dict(lora, strict=False)
        if len(load_result[1]) > 0:
            print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
        print(f"LoRA checkpoint loaded: {checkpoint}, total {len(lora)} keys")

    @staticmethod
    def _mark_only_lora_trainable(model):
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name

    def _configure_student_bsa(self, sparse_ratio):
        total = configure_wan_bsa(
            self.student_dit,
            enable=True,
            block_size=self.bsa_block_size,
            sparse_ratio=sparse_ratio,
            backend=self.bsa_backend,
            chunk_size=self.bsa_sdpa_chunk_size,
        )
        self.current_bsa_sparse_ratio = float(sparse_ratio)
        print(
            f"DMD student BSA enabled on {total} Wan blocks: block_size={self.bsa_block_size}, "
            f"sparse_ratio={self.current_bsa_sparse_ratio}, backend={self.bsa_backend}, "
            f"sdpa_chunk_size={self.bsa_sdpa_chunk_size}"
        )

    def on_train_step_start(self, global_step, total_steps=None):
        if not self.bsa_enable:
            return
        sparse_ratio = _bsa_sparse_ratio_for_step(
            global_step,
            self.bsa_sparse_ratio_start,
            self.bsa_sparse_ratio_target,
            self.bsa_sparse_ratio_warmup_steps,
        )
        if self.current_bsa_sparse_ratio == sparse_ratio:
            return
        updated = set_wan_bsa_sparse_ratio(self.student_dit, sparse_ratio)
        if updated > 0:
            self.current_bsa_sparse_ratio = sparse_ratio

    def student_trainable_modules(self):
        return (param for param in self.student_dit.parameters() if param.requires_grad)

    def fake_score_trainable_modules(self):
        return (param for param in self.fake_score_dit.parameters() if param.requires_grad)

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": data.get("negative_prompt", "")}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": self.real_guidance_scale,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def prepare_batch(self, data):
        if isinstance(data, tuple) and len(data) == 3:
            inputs_shared, inputs_posi, inputs_nega = data
        elif isinstance(data, dict) and "latents" in data and "context" in data:
            inputs_shared = dict(data)
            inputs_posi = {"context": data["context"]}
            inputs_nega = {"context": data.get("negative_context", data["context"])}
        else:
            self.pipe.scheduler.set_timesteps(1000, training=True)
            inputs_shared, inputs_posi, inputs_nega = self.get_pipeline_inputs(data)
            inputs_shared, inputs_posi, inputs_nega = self.transfer_data_to_device(
                (inputs_shared, inputs_posi, inputs_nega),
                self.pipe.device,
                self.pipe.torch_dtype,
            )
            for unit in self.pipe.units:
                inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                    unit,
                    self.pipe,
                    inputs_shared,
                    inputs_posi,
                    inputs_nega,
                )
        if "first_frame_latents" not in inputs_shared and "input_latents" in inputs_shared:
            inputs_shared["first_frame_latents"] = inputs_shared["input_latents"][:, :, 0:1].detach()
        if "latents" in inputs_shared:
            inputs_shared["latents"] = replace_first_frame_latents(
                inputs_shared["latents"],
                inputs_shared.get("first_frame_latents"),
            )
        return inputs_shared, inputs_posi, inputs_nega

    def compute_student_dmd_loss(self, batch, outer_step=0):
        inputs_shared, inputs_posi, inputs_nega = self.prepare_batch(batch)
        return compute_student_distribution_matching_loss(
            self.pipe,
            self.teacher_dit,
            self.student_dit,
            self.fake_score_dit,
            inputs_shared,
            inputs_posi,
            inputs_nega,
            dmd_denoising_steps=self.dmd_denoising_steps,
            real_guidance_scale=self.real_guidance_scale,
            fake_guidance_scale=self.fake_guidance_scale,
        )

    def compute_fake_score_loss(self, batch, outer_step=0):
        if self.fake_score_loss_type != "x0":
            raise ValueError(f"Unsupported fake_score_loss_type: {self.fake_score_loss_type}")
        inputs_shared, inputs_posi, _ = self.prepare_batch(batch)
        return compute_fake_score_x0_loss(
            self.pipe,
            self.student_dit,
            self.fake_score_dit,
            inputs_shared,
            inputs_posi,
            dmd_denoising_steps=self.dmd_denoising_steps,
        )

    def export_debug_state(self):
        return {
            "base_teacher_smoke": self.base_teacher_smoke,
            "dmd_denoising_steps": list(self.dmd_denoising_steps),
            "student_trainable_params": [name for name, param in self.student_dit.named_parameters() if param.requires_grad],
            "fake_score_trainable_params": [name for name, param in self.fake_score_dit.named_parameters() if param.requires_grad],
        }
