import math
from typing import Any, Dict, Iterable, Tuple


def parse_block_size(value: Any) -> Tuple[int, int, int]:
    if isinstance(value, str):
        parts = [int(part.strip()) for part in value.split(",")]
    else:
        parts = [int(part) for part in value]
    if len(parts) != 3:
        raise ValueError(f"block size must have exactly three values, got {value!r}.")
    if any(part <= 0 for part in parts):
        raise ValueError(f"block size values must be positive, got {value!r}.")
    return tuple(parts)


def format_block_size(value: Any) -> str:
    return ",".join(str(part) for part in parse_block_size(value))


def compute_sparsity_for_step(
    step: int,
    start_sparsity: float,
    end_sparsity: float,
    ramp_steps: int,
) -> float:
    start_sparsity = float(start_sparsity)
    end_sparsity = float(end_sparsity)
    if not 0.0 <= start_sparsity < 1.0:
        raise ValueError(f"start_sparsity must be in [0, 1), got {start_sparsity}.")
    if not 0.0 <= end_sparsity < 1.0:
        raise ValueError(f"end_sparsity must be in [0, 1), got {end_sparsity}.")
    if ramp_steps <= 0:
        return end_sparsity
    progress = min(max(float(step) / float(ramp_steps), 0.0), 1.0)
    weight = 0.5 - 0.5 * math.cos(math.pi * progress)
    return start_sparsity + (end_sparsity - start_sparsity) * weight


def _iter_modules(model: Any) -> Iterable[Any]:
    if hasattr(model, "modules"):
        yield from model.modules()
    else:
        yield model


def update_block_sparse_sparsity(model: Any, sparsity: float) -> None:
    for module in _iter_modules(model):
        config = getattr(module, "block_sparse_config", None)
        if config is not None:
            config["sparsity"] = float(sparsity)


def sparse_stats_from_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if config is None:
        return {"enabled": False}
    stats = {
        "enabled": bool(config.get("enabled", True)),
        "sparsity": float(config.get("sparsity", 0.0)),
        "block_size": format_block_size(config.get("block_size", (0, 0, 0))),
        "q_chunk_blocks": int(config.get("q_chunk_blocks", 0)),
    }
    for key in ("mask_density", "k_keep", "num_blocks", "num_valid_blocks"):
        if key in config:
            stats[key] = config[key]
    return stats


def collect_block_sparse_training_stats(model: Any) -> Dict[str, Any]:
    first_config = None
    debug = None
    for module in _iter_modules(model):
        config = getattr(module, "block_sparse_config", None)
        if first_config is None and config is not None:
            first_config = config
        module_debug = getattr(module, "last_block_sparse_debug", None)
        if module_debug is not None:
            debug = module_debug
    stats = sparse_stats_from_model_config(first_config)
    if debug is not None:
        for key in ("mask_density", "k_keep", "num_blocks", "num_valid_blocks"):
            if key in debug:
                stats[key] = debug[key]
    return stats
