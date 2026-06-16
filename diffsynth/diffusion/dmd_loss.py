import random
from typing import Optional

import torch
import torch.nn.functional as F


def parse_dmd_denoising_steps(value) -> tuple[int, ...]:
    if isinstance(value, str):
        steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        steps = tuple(int(item) for item in value)
    if len(steps) == 0:
        raise ValueError("dmd_denoising_steps cannot be empty.")
    if any(step < 0 or step > 1000 for step in steps):
        raise ValueError(f"DMD denoising steps must lie in [0, 1000], got {steps!r}.")
    return steps


def _sigma_from_timestep(timestep, ndim: int, dtype=None, device=None):
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], dtype=dtype or torch.float32, device=device)
    timestep = timestep.to(dtype=dtype or torch.float32, device=device or timestep.device)
    sigma = timestep / 1000.0
    while sigma.ndim < ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (ndim - sigma.ndim)))
    return sigma


def flow_prediction_to_x0(noisy_latents: torch.Tensor, flow_pred: torch.Tensor, timestep) -> torch.Tensor:
    sigma = _sigma_from_timestep(
        timestep,
        noisy_latents.ndim,
        dtype=noisy_latents.dtype,
        device=noisy_latents.device,
    )
    return noisy_latents - sigma * flow_pred


def add_noise_at_timestep(clean_latents: torch.Tensor, noise: torch.Tensor, timestep) -> torch.Tensor:
    sigma = _sigma_from_timestep(
        timestep,
        clean_latents.ndim,
        dtype=clean_latents.dtype,
        device=clean_latents.device,
    )
    return (1.0 - sigma) * clean_latents + sigma * noise


def replace_first_frame_latents(latents: torch.Tensor, first_frame_latents: Optional[torch.Tensor]):
    if first_frame_latents is None:
        return latents
    latents = latents.clone()
    latents[:, :, 0:1] = first_frame_latents.to(dtype=latents.dtype, device=latents.device)
    return latents


def stop_gradient_mse_with_manual_gradient(
    generated: torch.Tensor,
    manual_gradient: torch.Tensor,
    gradient_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target = (generated - manual_gradient).detach()
    if gradient_mask is not None:
        return 0.5 * F.mse_loss(generated[gradient_mask].float(), target[gradient_mask].float())
    return 0.5 * F.mse_loss(generated.float(), target.float())


def _copy_inputs(inputs: dict) -> dict:
    return dict(inputs or {})


def _models_for_pipe(pipe, dit):
    models = {name: getattr(pipe, name, None) for name in getattr(pipe, "in_iteration_models", ("dit",))}
    models["dit"] = dit
    return models


def _model_forward(pipe, dit, inputs_shared, inputs_cond, timestep, progress_id=0):
    call_inputs = _copy_inputs(inputs_shared)
    call_inputs["latents"] = call_inputs["latents"].to(dtype=pipe.torch_dtype, device=pipe.device)
    timestep = torch.as_tensor([float(timestep)], dtype=pipe.torch_dtype, device=pipe.device)
    return pipe.model_fn(
        **_models_for_pipe(pipe, dit),
        **call_inputs,
        **_copy_inputs(inputs_cond),
        timestep=timestep,
        progress_id=progress_id,
    )


def run_student_rollout(
    pipe,
    student_dit,
    inputs_shared: dict,
    inputs_posi: dict,
    dmd_denoising_steps=(1000, 750, 500, 250),
    exit_index: Optional[int] = None,
):
    dmd_denoising_steps = parse_dmd_denoising_steps(dmd_denoising_steps)
    if exit_index is None:
        exit_index = random.randrange(len(dmd_denoising_steps))
    exit_index = int(exit_index)
    if exit_index < 0 or exit_index >= len(dmd_denoising_steps):
        raise ValueError(f"exit_index out of range: {exit_index}")

    latents = inputs_shared["latents"]
    first_frame_latents = inputs_shared.get("first_frame_latents")
    latents = replace_first_frame_latents(latents, first_frame_latents)
    forward_count = 0

    for progress_id, timestep in enumerate(dmd_denoising_steps):
        step_inputs = _copy_inputs(inputs_shared)
        step_inputs["latents"] = latents
        if progress_id == exit_index:
            flow_pred = _model_forward(pipe, student_dit, step_inputs, inputs_posi, timestep, progress_id)
            forward_count += 1
            pred_x0 = flow_prediction_to_x0(latents, flow_pred, timestep)
            pred_x0 = replace_first_frame_latents(pred_x0, first_frame_latents)
            next_timestep = 0 if progress_id + 1 >= len(dmd_denoising_steps) else dmd_denoising_steps[progress_id + 1]
            return pred_x0, {
                "exit_index": exit_index,
                "denoised_timestep_from": timestep,
                "denoised_timestep_to": next_timestep,
                "dit_forward_count_student": forward_count,
            }

        with torch.no_grad():
            flow_pred = _model_forward(pipe, student_dit, step_inputs, inputs_posi, timestep, progress_id)
            pred_x0 = flow_prediction_to_x0(latents, flow_pred, timestep)
            pred_x0 = replace_first_frame_latents(pred_x0, first_frame_latents)
        forward_count += 1
        next_timestep = dmd_denoising_steps[progress_id + 1]
        latents = add_noise_at_timestep(pred_x0, torch.randn_like(pred_x0), next_timestep)
        latents = replace_first_frame_latents(latents, first_frame_latents)

    raise RuntimeError("Student rollout exited without returning a sample.")


def _score_timestep(denoised_timestep_from, denoised_timestep_to, device):
    hi = int(denoised_timestep_from or 1000)
    lo = int(denoised_timestep_to or 0)
    lo = max(1, min(lo, 999))
    hi = max(lo + 1, min(hi, 1000))
    return torch.randint(lo, hi, (1,), device=device, dtype=torch.long).float()


def compute_dmd_gradient(
    pipe,
    teacher_dit,
    fake_score_dit,
    noisy_latents: torch.Tensor,
    generated_x0: torch.Tensor,
    timestep: torch.Tensor,
    inputs_shared: dict,
    inputs_posi: dict,
    inputs_nega: dict,
    real_guidance_scale: float = 6.0,
    fake_guidance_scale: float = 0.0,
    eps: float = 1e-6,
):
    score_inputs = _copy_inputs(inputs_shared)
    score_inputs["latents"] = noisy_latents

    with torch.no_grad():
        fake_cond_flow = _model_forward(pipe, fake_score_dit, score_inputs, inputs_posi, timestep)
        pred_fake_x0 = flow_prediction_to_x0(noisy_latents, fake_cond_flow, timestep)
        fake_forward_count = 1
        if float(fake_guidance_scale) != 0.0:
            fake_uncond_flow = _model_forward(pipe, fake_score_dit, score_inputs, inputs_nega, timestep)
            pred_fake_uncond_x0 = flow_prediction_to_x0(noisy_latents, fake_uncond_flow, timestep)
            pred_fake_x0 = pred_fake_uncond_x0 + fake_guidance_scale * (pred_fake_x0 - pred_fake_uncond_x0)
            fake_forward_count += 1

        real_cond_flow = _model_forward(pipe, teacher_dit, score_inputs, inputs_posi, timestep)
        pred_real_cond_x0 = flow_prediction_to_x0(noisy_latents, real_cond_flow, timestep)
        real_uncond_flow = _model_forward(pipe, teacher_dit, score_inputs, inputs_nega, timestep)
        pred_real_uncond_x0 = flow_prediction_to_x0(noisy_latents, real_uncond_flow, timestep)
        pred_real_x0 = pred_real_uncond_x0 + real_guidance_scale * (pred_real_cond_x0 - pred_real_uncond_x0)

    grad = pred_fake_x0 - pred_real_x0
    normalizer_dims = tuple(range(1, generated_x0.ndim))
    normalizer = (generated_x0 - pred_real_x0).abs().mean(dim=normalizer_dims, keepdim=True).clamp_min(eps)
    grad = torch.nan_to_num(grad / normalizer)
    return grad, {
        "dmd_gradient_norm": grad.detach().float().norm(),
        "dit_forward_count_fake": fake_forward_count,
        "dit_forward_count_teacher": 2,
        "sampled_score_timestep": timestep.detach().float().mean(),
    }


def compute_student_distribution_matching_loss(
    pipe,
    teacher_dit,
    student_dit,
    fake_score_dit,
    inputs_shared: dict,
    inputs_posi: dict,
    inputs_nega: dict,
    dmd_denoising_steps=(1000, 750, 500, 250),
    real_guidance_scale: float = 6.0,
    fake_guidance_scale: float = 0.0,
    exit_index: Optional[int] = None,
):
    generated_x0, rollout_log = run_student_rollout(
        pipe,
        student_dit,
        inputs_shared,
        inputs_posi,
        dmd_denoising_steps=dmd_denoising_steps,
        exit_index=exit_index,
    )
    timestep = _score_timestep(
        rollout_log["denoised_timestep_from"],
        rollout_log["denoised_timestep_to"],
        device=generated_x0.device,
    )
    noisy_latents = add_noise_at_timestep(generated_x0, torch.randn_like(generated_x0), timestep)
    noisy_latents = replace_first_frame_latents(noisy_latents, inputs_shared.get("first_frame_latents"))
    grad, grad_log = compute_dmd_gradient(
        pipe,
        teacher_dit,
        fake_score_dit,
        noisy_latents,
        generated_x0,
        timestep,
        inputs_shared,
        inputs_posi,
        inputs_nega,
        real_guidance_scale=real_guidance_scale,
        fake_guidance_scale=fake_guidance_scale,
    )
    loss = stop_gradient_mse_with_manual_gradient(generated_x0, grad)
    log = dict(rollout_log)
    log.update(grad_log)
    log["student_dmd_loss"] = loss.detach()
    return loss, log


def compute_fake_score_x0_loss(
    pipe,
    student_dit,
    fake_score_dit,
    inputs_shared: dict,
    inputs_posi: dict,
    dmd_denoising_steps=(1000, 750, 500, 250),
    exit_index: Optional[int] = None,
):
    with torch.no_grad():
        generated_x0, rollout_log = run_student_rollout(
            pipe,
            student_dit,
            inputs_shared,
            inputs_posi,
            dmd_denoising_steps=dmd_denoising_steps,
            exit_index=exit_index,
        )
    timestep = _score_timestep(
        rollout_log["denoised_timestep_from"],
        rollout_log["denoised_timestep_to"],
        device=generated_x0.device,
    )
    noise = torch.randn_like(generated_x0)
    noisy_latents = add_noise_at_timestep(generated_x0, noise, timestep)
    noisy_latents = replace_first_frame_latents(noisy_latents, inputs_shared.get("first_frame_latents"))
    score_inputs = _copy_inputs(inputs_shared)
    score_inputs["latents"] = noisy_latents
    fake_flow = _model_forward(pipe, fake_score_dit, score_inputs, inputs_posi, timestep)
    pred_fake_x0 = flow_prediction_to_x0(noisy_latents, fake_flow, timestep)
    loss = F.mse_loss(pred_fake_x0.float(), generated_x0.float())
    return loss, {
        "fake_score_loss": loss.detach(),
        "sampled_score_timestep": timestep.detach().float().mean(),
        "denoised_timestep_from": rollout_log["denoised_timestep_from"],
        "denoised_timestep_to": rollout_log["denoised_timestep_to"],
        "dit_forward_count_student": rollout_log.get("dit_forward_count_student", 0),
        "dit_forward_count_fake": 1,
    }
