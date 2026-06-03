import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from .training_metrics import TrainingMetricsWriter, compute_wan_video_tokens
from diffsynth.core import OffloadTrainingManager


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        max_train_steps = getattr(args, "max_train_steps", None)
    else:
        max_train_steps = None
    if max_train_steps is not None and max_train_steps <= 0:
        max_train_steps = None

    trainable_params = list(model.trainable_modules())
    optimizer_kwargs = {}
    if args is not None and getattr(args, "optimizer_fused", False) and torch.cuda.is_available():
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay, **optimizer_kwargs)
    except TypeError:
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader_kwargs = {
        "shuffle": True,
        "collate_fn": lambda x: x[0],
        "num_workers": num_workers,
    }
    if args is not None:
        dataloader_kwargs["pin_memory"] = getattr(args, "dataset_pin_memory", False)
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = getattr(args, "dataset_persistent_workers", False)
            if getattr(args, "dataset_prefetch_factor", None) is not None:
                dataloader_kwargs["prefetch_factor"] = args.dataset_prefetch_factor
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    tokens_per_sample = 0
    if args is not None:
        vae_factor = getattr(getattr(model.pipe, "vae", None), "upsampling_factor", 16)
        patch_size = getattr(getattr(model.pipe, "dit", None), "patch_size", (1, 2, 2))
        tokens_per_sample = compute_wan_video_tokens(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            vae_upsampling_factor=vae_factor,
            patch_size=patch_size,
        )
    metrics_writer = TrainingMetricsWriter(
        model_logger.output_path,
        enabled=(args is None or not getattr(args, "disable_training_metrics", False)) and accelerator.is_main_process,
        tokens_per_sample=tokens_per_sample,
        log_steps=1 if args is None else args.log_steps,
        use_tensorboard=False if args is None else args.enable_tensorboard,
    )

    initialize_deepspeed_gradient_checkpointing(accelerator)
    global_step = 0
    total_steps = len(dataloader) * num_epochs
    if max_train_steps is not None:
        total_steps = min(total_steps, max_train_steps)
    stop_training = False
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            if max_train_steps is not None and global_step >= max_train_steps:
                stop_training = True
                break
            if hasattr(model, "on_train_step_start"):
                model.on_train_step_start(global_step, total_steps)
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                global_step += 1
                metrics_writer.log_step(
                    step=global_step,
                    loss=loss,
                    learning_rate=scheduler.get_last_lr()[0],
                    samples=accelerator.num_processes,
                    bsa_sparse_ratio=getattr(model, "current_bsa_sparse_ratio", None),
                )
                if max_train_steps is not None and global_step >= max_train_steps:
                    stop_training = True
                    break
        if not stop_training and save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        if stop_training:
            break

    model_logger.on_training_end(accelerator, model, save_steps)
    metrics_writer.close()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
