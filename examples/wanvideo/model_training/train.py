import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.models.wan_bsa import configure_wan_bsa, parse_bsa_block_size, set_wan_bsa_sparse_ratio
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bsa_sparse_ratio_for_step(step, start, target, warmup_steps):
    target = float(target)
    if start is None or warmup_steps is None or int(warmup_steps) <= 0:
        return target
    start = float(start)
    warmup_steps = int(warmup_steps)
    if step >= warmup_steps:
        return target
    ratio = start + (target - start) * (float(step) / float(warmup_steps))
    return round(ratio, 12)


def _split_csv(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_wan_special_operator_map(
    dataset_base_path,
    data_file_keys,
    extra_inputs,
    num_frames,
    framewise_decoding=False,
):
    requested_keys = set(_split_csv(data_file_keys)) | set(_split_csv(extra_inputs))
    special_operator_map = {
        "animate_face_video": ToAbsolutePath(dataset_base_path) >> LoadVideo(
            num_frames,
            4,
            1,
            frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
        ),
        "wantodance_music_path": ToAbsolutePath(dataset_base_path),
    }
    if "input_audio" in requested_keys:
        special_operator_map["input_audio"] = ToAbsolutePath(dataset_base_path) >> LoadAudio(sr=16000)
    return special_operator_map


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        bsa_enable=False,
        bsa_block_size=(2, 4, 4),
        bsa_sparse_ratio=0.5,
        bsa_sparse_ratio_start=None,
        bsa_sparse_ratio_warmup_steps=0,
        bsa_backend="flex",
        bsa_sdpa_chunk_size=64,
        redirect_common_files=True,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            redirect_common_files=redirect_common_files,
        )
        self.bsa_enable = bool(bsa_enable)
        self.bsa_sparse_ratio_target = float(bsa_sparse_ratio)
        self.bsa_sparse_ratio_start = None if bsa_sparse_ratio_start is None else float(bsa_sparse_ratio_start)
        self.bsa_sparse_ratio_warmup_steps = int(bsa_sparse_ratio_warmup_steps or 0)
        self.current_bsa_sparse_ratio = None
        if self.bsa_enable:
            self.bsa_block_size = parse_bsa_block_size(bsa_block_size)
            self.bsa_backend = bsa_backend
            self.bsa_sdpa_chunk_size = int(bsa_sdpa_chunk_size)
            initial_sparse_ratio = bsa_sparse_ratio_for_step(
                0,
                self.bsa_sparse_ratio_start,
                self.bsa_sparse_ratio_target,
                self.bsa_sparse_ratio_warmup_steps,
            )
            self._configure_bsa(initial_sparse_ratio)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def _configure_bsa(self, sparse_ratio):
        total = 0
        for model_name in ("dit", "dit2"):
            total += configure_wan_bsa(
                getattr(self.pipe, model_name, None),
                enable=True,
                block_size=self.bsa_block_size,
                sparse_ratio=sparse_ratio,
                backend=self.bsa_backend,
                chunk_size=self.bsa_sdpa_chunk_size,
            )
        self.current_bsa_sparse_ratio = float(sparse_ratio)
        print(
            f"BSA enabled on {total} Wan blocks: block_size={self.bsa_block_size}, "
            f"sparse_ratio={self.current_bsa_sparse_ratio}, backend={self.bsa_backend}, "
            f"sdpa_chunk_size={self.bsa_sdpa_chunk_size}"
        )

    def on_train_step_start(self, global_step, total_steps=None):
        if not self.bsa_enable:
            return
        sparse_ratio = bsa_sparse_ratio_for_step(
            global_step,
            self.bsa_sparse_ratio_start,
            self.bsa_sparse_ratio_target,
            self.bsa_sparse_ratio_warmup_steps,
        )
        if self.current_bsa_sparse_ratio == sparse_ratio:
            return
        updated = 0
        for model_name in ("dit", "dit2"):
            updated += set_wan_bsa_sparse_ratio(getattr(self.pipe, model_name, None), sparse_ratio)
        if updated > 0:
            self.current_bsa_sparse_ratio = sparse_ratio
        
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
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument("--disable_common_file_redirect", default=False, action="store_true", help="Disable Wan common-file redirection so existing original .pth files are reused.")
    parser.add_argument("--bsa_enable", default=False, action="store_true", help="Enable Block Sparse Attention in Wan self-attention during training.")
    parser.add_argument("--bsa_block_size", type=str, default="2,4,4", help="BSA block size as 'bt,bh,bw'. Use '2,2,3' for the 480x832@81 high-sparsity experiment.")
    parser.add_argument("--bsa_sparse_ratio", type=float, default=0.5, help="Target BSA block drop ratio.")
    parser.add_argument("--bsa_sparse_ratio_start", type=float, default=None, help="Optional warmup start ratio. If omitted, training starts at bsa_sparse_ratio.")
    parser.add_argument("--bsa_sparse_ratio_warmup_steps", type=int, default=0, help="Micro steps used to linearly warm sparse ratio from start to target.")
    parser.add_argument("--bsa_backend", type=str, choices=["flex", "sdpa_chunked"], default="flex", help="BSA attention backend.")
    parser.add_argument("--bsa_sdpa_chunk_size", type=int, default=64, help="Requested query-block chunk size for BSA sdpa_chunked backend.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map=build_wan_special_operator_map(
            dataset_base_path=args.dataset_base_path,
            data_file_keys=args.data_file_keys,
            extra_inputs=args.extra_inputs,
            num_frames=args.num_frames,
            framewise_decoding=args.framewise_decoding,
        )
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        bsa_enable=args.bsa_enable,
        bsa_block_size=args.bsa_block_size,
        bsa_sparse_ratio=args.bsa_sparse_ratio,
        bsa_sparse_ratio_start=args.bsa_sparse_ratio_start,
        bsa_sparse_ratio_warmup_steps=args.bsa_sparse_ratio_warmup_steps,
        bsa_backend=args.bsa_backend,
        bsa_sdpa_chunk_size=args.bsa_sdpa_chunk_size,
        redirect_common_files=not args.disable_common_file_redirect,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
