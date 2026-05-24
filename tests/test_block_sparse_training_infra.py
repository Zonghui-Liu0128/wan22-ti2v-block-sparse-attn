import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


def load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_logger_writes_loss_throughput_and_sparse_stats(tmp_path, monkeypatch):
    accelerate_stub = types.ModuleType("accelerate")
    accelerate_stub.Accelerator = object
    monkeypatch.setitem(sys.modules, "accelerate", accelerate_stub)
    logger_module = load_module("training_logger", "diffsynth/diffusion/logger.py")

    class AcceleratorStub:
        is_main_process = True

    log_file = tmp_path / "training_log.jsonl"
    logger = logger_module.ModelLogger(tmp_path, training_log_file=log_file)
    logger.on_step_end(
        AcceleratorStub(),
        model=None,
        save_steps=None,
        loss=torch.tensor(1.25),
        step_time=2.0,
        batch_samples=2,
        batch_frames=34,
        learning_rate=1e-5,
        sparse_stats={
            "enabled": True,
            "sparsity": 0.9,
            "block_size": (2, 8, 8),
            "mask_density": 0.1,
            "k_keep": 4,
        },
    )

    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["step"] == 1
    assert record["loss"] == pytest.approx(1.25)
    assert record["seconds_per_step"] == pytest.approx(2.0)
    assert record["samples_per_second"] == pytest.approx(1.0)
    assert record["frames_per_second"] == pytest.approx(17.0)
    assert record["lr"] == pytest.approx(1e-5)
    assert record["sparse_enabled"] is True
    assert record["sparsity"] == pytest.approx(0.9)
    assert record["block_size"] == "2,8,8"
    assert record["mask_density"] == pytest.approx(0.1)
    assert record["k_keep"] == 4


def test_cosine_sparsity_schedule_and_config_parser():
    schedule = load_module("block_sparse_schedule", "diffsynth/core/attention/block_sparse_schedule.py")

    assert schedule.parse_block_size("2,8,8") == (2, 8, 8)
    assert schedule.compute_sparsity_for_step(0, 0.80, 0.95, 4) == pytest.approx(0.80)
    assert schedule.compute_sparsity_for_step(4, 0.80, 0.95, 4) == pytest.approx(0.95)
    midpoint = schedule.compute_sparsity_for_step(2, 0.80, 0.95, 4)
    assert 0.80 < midpoint < 0.95

    stats = schedule.sparse_stats_from_model_config(
        {
            "enabled": True,
            "sparsity": 0.925,
            "block_size": (2, 8, 8),
            "q_chunk_blocks": 4,
        }
    )
    assert stats["enabled"] is True
    assert stats["sparsity"] == pytest.approx(0.925)
    assert stats["block_size"] == "2,8,8"
    assert stats["q_chunk_blocks"] == 4
