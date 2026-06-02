import csv
import html
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch


def compute_wan_video_tokens(
    height: int,
    width: int,
    num_frames: int,
    vae_upsampling_factor: int = 16,
    patch_size=(1, 2, 2),
) -> int:
    if height is None or width is None or num_frames is None:
        return 0
    pt, ph, pw = patch_size
    latent_frames = (int(num_frames) - 1) // 4 + 1
    token_h = math.ceil(int(height) / (int(vae_upsampling_factor) * int(ph)))
    token_w = math.ceil(int(width) / (int(vae_upsampling_factor) * int(pw)))
    return int(latent_frames * token_h * token_w)


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


class TrainingMetricsWriter:
    fieldnames = [
        "step",
        "loss",
        "learning_rate",
        "elapsed_seconds",
        "step_seconds",
        "samples_this_step",
        "total_samples",
        "tokens_per_sample",
        "tokens_this_step",
        "total_tokens",
        "tokens_per_second",
        "tokens_per_hour",
        "tokens_per_day",
        "videos_per_second",
        "videos_per_hour",
        "videos_per_day",
        "bsa_sparse_ratio",
    ]

    def __init__(
        self,
        output_path,
        enabled: bool = True,
        tokens_per_sample: int = 0,
        log_steps: int = 1,
        use_tensorboard: bool = False,
    ):
        self.output_path = Path(output_path)
        self.enabled = bool(enabled)
        self.tokens_per_sample = int(tokens_per_sample or 0)
        self.log_steps = max(1, int(log_steps or 1))
        self.total_samples = 0
        self.total_tokens = 0
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.rows = []
        self.csv_file = None
        self.jsonl_file = None
        self.csv_writer = None
        self.tb_writer = None

        if not self.enabled:
            return

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_path / "training_metrics.csv"
        self.jsonl_path = self.output_path / "training_metrics.jsonl"
        self.html_path = self.output_path / "training_metrics.html"
        self.csv_file = self.csv_path.open("w", newline="")
        self.jsonl_file = self.jsonl_path.open("w")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.csv_writer.writeheader()

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(self.output_path / "tensorboard"))
            except Exception:
                self.tb_writer = None

    def log_step(
        self,
        step: int,
        loss,
        learning_rate: Optional[float],
        samples: int = 1,
        bsa_sparse_ratio: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        step_seconds: Optional[float] = None,
    ):
        samples = int(samples)
        tokens_this_step = samples * self.tokens_per_sample
        self.total_samples += samples
        self.total_tokens += tokens_this_step

        now = time.perf_counter()
        if elapsed_seconds is None:
            elapsed_seconds = max(now - self.start_time, 1e-9)
        if step_seconds is None:
            step_seconds = max(now - self.last_time, 1e-9)
        self.last_time = now

        tokens_per_second = self.total_tokens / max(float(elapsed_seconds), 1e-9)
        videos_per_second = self.total_samples / max(float(elapsed_seconds), 1e-9)
        row = {
            "step": int(step),
            "loss": _as_float(loss),
            "learning_rate": _as_float(learning_rate),
            "elapsed_seconds": float(elapsed_seconds),
            "step_seconds": float(step_seconds),
            "samples_this_step": samples,
            "total_samples": self.total_samples,
            "tokens_per_sample": self.tokens_per_sample,
            "tokens_this_step": tokens_this_step,
            "total_tokens": self.total_tokens,
            "tokens_per_second": tokens_per_second,
            "tokens_per_hour": tokens_per_second * 3600.0,
            "tokens_per_day": tokens_per_second * 86400.0,
            "videos_per_second": videos_per_second,
            "videos_per_hour": videos_per_second * 3600.0,
            "videos_per_day": videos_per_second * 86400.0,
            "bsa_sparse_ratio": None if bsa_sparse_ratio is None else float(bsa_sparse_ratio),
        }
        self.rows.append(row)

        if self.enabled and int(step) % self.log_steps == 0:
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            self.jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.jsonl_file.flush()
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/loss", row["loss"], step)
                self.tb_writer.add_scalar("train/tokens_per_second", row["tokens_per_second"], step)
                self.tb_writer.add_scalar("train/videos_per_second", row["videos_per_second"], step)
                if row["bsa_sparse_ratio"] is not None:
                    self.tb_writer.add_scalar("bsa/sparse_ratio", row["bsa_sparse_ratio"], step)
        return row

    def close(self):
        if not self.enabled:
            return
        render_training_metrics_html(self.rows, self.html_path)
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        if self.csv_file is not None:
            self.csv_file.close()
        if self.jsonl_file is not None:
            self.jsonl_file.close()


def _svg_line(rows, key, color, width=900, height=240):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return "<p>No data.</p>"
    lo = min(values)
    hi = max(values)
    if hi == lo:
        hi = lo + 1.0
    points = []
    for i, value in enumerate(values):
        x = 20 + (width - 40) * (i / max(len(values) - 1, 1))
        y = 20 + (height - 40) * (1.0 - (float(value) - lo) / (hi - lo))
        points.append(f"{x:.2f},{y:.2f}")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#111827"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>'
        f'<text x="20" y="18" fill="#f9fafb" font-size="14">{html.escape(key)} '
        f'min={lo:.4g} max={hi:.4g}</text></svg>'
    )


def render_training_metrics_html(rows, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    latest = rows[-1] if rows else {}
    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Training Metrics</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f9fafb;color:#111827}",
        "section{margin:20px 0} table{border-collapse:collapse}td,th{padding:6px 10px;border:1px solid #d1d5db}</style>",
        "</head><body>",
        "<h1>Training Metrics</h1>",
        "<section><h2>Latest</h2><table>",
    ]
    for key in TrainingMetricsWriter.fieldnames:
        if key in latest:
            body.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(latest[key]))}</td></tr>")
    body.extend([
        "</table></section>",
        "<section><h2>Loss</h2>",
        _svg_line(rows, "loss", "#38bdf8"),
        "</section><section><h2>Tokens Per Second</h2>",
        _svg_line(rows, "tokens_per_second", "#22c55e"),
        "</section><section><h2>Videos Per Hour</h2>",
        _svg_line(rows, "videos_per_hour", "#f97316"),
        "</section></body></html>",
    ])
    output_file.write_text("\n".join(body))
