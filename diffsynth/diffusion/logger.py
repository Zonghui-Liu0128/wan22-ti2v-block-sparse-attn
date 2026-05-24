import json, os, torch

try:
    from accelerate import Accelerator
except ModuleNotFoundError:
    Accelerator = object


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, training_log_file=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.training_log_file = training_log_file


    def _training_log_path(self):
        if self.training_log_file is None:
            return os.path.join(self.output_path, "training_log.jsonl")
        return str(self.training_log_file)


    def _to_float(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().cpu().item())
        return float(value)


    def _format_block_size(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return ",".join(str(int(part)) for part in value)


    def write_training_log(
        self,
        accelerator: Accelerator,
        loss=None,
        step_time=None,
        batch_samples=None,
        batch_frames=None,
        learning_rate=None,
        epoch=None,
        sparse_stats=None,
    ):
        if not getattr(accelerator, "is_main_process", True):
            return
        record = {"step": self.num_steps}
        if epoch is not None:
            record["epoch"] = int(epoch)
        if loss is not None:
            record["loss"] = self._to_float(loss)
        if learning_rate is not None:
            record["lr"] = self._to_float(learning_rate)
        if step_time is not None:
            step_time = self._to_float(step_time)
            record["seconds_per_step"] = step_time
            if batch_samples is not None and step_time > 0:
                record["samples_per_second"] = float(batch_samples) / step_time
            if batch_frames is not None and step_time > 0:
                record["frames_per_second"] = float(batch_frames) / step_time
        sparse_stats = {} if sparse_stats is None else sparse_stats
        if sparse_stats:
            record["sparse_enabled"] = bool(sparse_stats.get("enabled", False))
            for key in ("sparsity", "mask_density"):
                if key in sparse_stats and sparse_stats[key] is not None:
                    record[key] = self._to_float(sparse_stats[key])
            for key in ("k_keep", "num_blocks", "num_valid_blocks", "q_chunk_blocks"):
                if key in sparse_stats and sparse_stats[key] is not None:
                    record[key] = int(sparse_stats[key])
            if "block_size" in sparse_stats:
                record["block_size"] = self._format_block_size(sparse_stats["block_size"])
        path = self._training_log_path()
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        self.write_training_log(
            accelerator,
            loss=kwargs.get("loss"),
            step_time=kwargs.get("step_time"),
            batch_samples=kwargs.get("batch_samples"),
            batch_frames=kwargs.get("batch_frames"),
            learning_rate=kwargs.get("learning_rate"),
            epoch=kwargs.get("epoch"),
            sparse_stats=kwargs.get("sparse_stats"),
        )
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
