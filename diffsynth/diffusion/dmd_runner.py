import torch
from tqdm import tqdm

from .dmd_checkpoint import DMDLoRACheckpointManager
from .training_metrics import TrainingMetricsWriter, compute_wan_video_tokens


def dmd_student_update_for_step(outer_step, fake_score_updates_per_generator_update):
    ratio = int(fake_score_updates_per_generator_update)
    if ratio <= 0:
        raise ValueError("fake_score_updates_per_generator_update must be positive")
    return int(outer_step) % ratio == 0


def _grad_norm(parameters):
    total = torch.tensor(0.0)
    seen = False
    for param in parameters:
        if param.grad is None:
            continue
        seen = True
        total = total + param.grad.detach().float().norm(2).pow(2).cpu()
    if not seen:
        return 0.0
    return float(total.sqrt().item())


def launch_dmd_lora_training_task(
    accelerator,
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    model_logger,
    args=None,
):
    if args is None:
        raise ValueError("DMD LoRA training requires parsed args.")
    for method_name in (
        "student_trainable_modules",
        "fake_score_trainable_modules",
        "compute_student_dmd_loss",
        "compute_fake_score_loss",
    ):
        if not hasattr(model, method_name):
            raise TypeError(f"DMD LoRA model must expose {method_name}().")

    student_params = list(model.student_trainable_modules())
    fake_score_params = list(model.fake_score_trainable_modules())
    if len(student_params) == 0:
        raise ValueError("No trainable student LoRA parameters found.")
    if len(fake_score_params) == 0:
        raise ValueError("No trainable fake_score LoRA parameters found.")

    optimizer_student = torch.optim.AdamW(
        student_params,
        lr=args.student_learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.student_beta1, args.student_beta2),
    )
    optimizer_fake_score = torch.optim.AdamW(
        fake_score_params,
        lr=args.fake_score_learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.fake_score_beta1, args.fake_score_beta2),
    )
    scheduler_student = torch.optim.lr_scheduler.ConstantLR(optimizer_student)
    scheduler_fake_score = torch.optim.lr_scheduler.ConstantLR(optimizer_fake_score)

    dataloader_kwargs = {
        "shuffle": True,
        "collate_fn": lambda x: x[0],
        "num_workers": args.dataset_num_workers,
        "pin_memory": args.dataset_pin_memory,
    }
    if args.dataset_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = args.dataset_persistent_workers
        if args.dataset_prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = args.dataset_prefetch_factor
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    prepared = accelerator.prepare(
        model,
        optimizer_student,
        optimizer_fake_score,
        scheduler_student,
        scheduler_fake_score,
        dataloader,
    )
    model, optimizer_student, optimizer_fake_score, scheduler_student, scheduler_fake_score, dataloader = prepared

    vae_factor = getattr(getattr(getattr(model, "pipe", None), "vae", None), "upsampling_factor", 16)
    patch_size = getattr(getattr(getattr(model, "pipe", None), "dit", None), "patch_size", (1, 2, 2))
    tokens_per_sample = compute_wan_video_tokens(args.height, args.width, args.num_frames, vae_factor, patch_size)
    metrics_writer = TrainingMetricsWriter(
        model_logger.output_path,
        enabled=not args.disable_training_metrics and accelerator.is_main_process,
        tokens_per_sample=tokens_per_sample,
        log_steps=args.log_steps,
        use_tensorboard=args.enable_tensorboard,
    )
    checkpoint_manager = DMDLoRACheckpointManager(model_logger.output_path)

    max_train_steps = args.max_train_steps or len(dataloader) * args.num_epochs
    outer_step = 0
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
    if resume_from_checkpoint:
        state = checkpoint_manager.load_training_checkpoint(
            resume_from_checkpoint,
            model,
            optimizer_student,
            optimizer_fake_score,
            scheduler_student,
            scheduler_fake_score,
        )
        outer_step = int(state.get("outer_step", 0))
        rng_state = state.get("rng_state", {}).get("torch")
        if rng_state is not None:
            torch.set_rng_state(torch.tensor(rng_state, dtype=torch.uint8))
    stop_training = False
    for _epoch in range(args.num_epochs):
        for batch in tqdm(dataloader):
            if outer_step >= max_train_steps:
                stop_training = True
                break
            if hasattr(model, "on_train_step_start"):
                model.on_train_step_start(outer_step, max_train_steps)

            student_update = dmd_student_update_for_step(
                outer_step,
                args.fake_score_updates_per_generator_update,
            )
            student_loss = None
            student_grad_norm = 0.0
            student_log = {}
            if student_update:
                optimizer_student.zero_grad(set_to_none=True)
                student_loss, student_log = model.compute_student_dmd_loss(batch, outer_step=outer_step)
                accelerator.backward(student_loss)
                student_grad_norm = _grad_norm(student_params)
                optimizer_student.step()
                scheduler_student.step()

            optimizer_fake_score.zero_grad(set_to_none=True)
            fake_score_loss, fake_log = model.compute_fake_score_loss(batch, outer_step=outer_step)
            accelerator.backward(fake_score_loss)
            fake_score_grad_norm = _grad_norm(fake_score_params)
            optimizer_fake_score.step()
            scheduler_fake_score.step()

            metrics = {
                "outer_step": outer_step,
                "student_update": student_update,
                "fake_score_update": True,
                "student_dmd_loss": student_loss,
                "fake_score_loss": fake_score_loss,
                "student_grad_norm": student_grad_norm,
                "fake_score_grad_norm": fake_score_grad_norm,
                "teacher_grad_norm": 0.0,
                "lr_student": scheduler_student.get_last_lr()[0],
                "lr_fake_score": scheduler_fake_score.get_last_lr()[0],
            }
            metrics.update(student_log or {})
            for key, value in (fake_log or {}).items():
                if key.startswith("dit_forward_count_") and key in metrics:
                    metrics[key] = int(metrics[key] or 0) + int(value or 0)
                else:
                    metrics[key] = value
            metrics_writer.log_step(
                step=outer_step + 1,
                loss=student_loss if student_loss is not None else fake_score_loss,
                learning_rate=None,
                samples=accelerator.num_processes,
                bsa_sparse_ratio=getattr(model, "current_bsa_sparse_ratio", None),
                dmd_metrics=metrics,
            )

            outer_step += 1
            if args.save_steps is not None and outer_step % args.save_steps == 0:
                checkpoint_manager.save_training_checkpoint(
                    accelerator,
                    model,
                    optimizer_student,
                    optimizer_fake_score,
                    scheduler_student,
                    scheduler_fake_score,
                    outer_step,
                    dmd_config={
                        "dmd_denoising_steps": args.dmd_denoising_steps,
                        "real_guidance_scale": args.real_guidance_scale,
                        "fake_guidance_scale": args.fake_guidance_scale,
                        "fake_score_updates_per_generator_update": args.fake_score_updates_per_generator_update,
                    },
                )
        if stop_training:
            break

    checkpoint_manager.save_final_student_lora(accelerator, model)
    metrics_writer.close()
