# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Speech-specific training observability, matching the production DeepSpeed trainer (``phyagi``).

Adds the metrics the production ``ds_trainer`` logs to MLflow but that Megatron-Bridge's built-in
``training_log`` does not: the speech-specific ``consumed_audios_in_hours`` (audio duration trained,
``audio_frames.sum() * 10ms``), ``ppl``, and per-step throughput, plus a **rank-0** console line
(the built-in megatron line prints on the last rank only).

MLflow is attached to the *existing* AzureML-managed run via ``mlflow.start_run(run_id=<env run>)``
(the platform pre-creates the run; calling ``start_run`` with a new name is rejected). Every MLflow
call is best-effort (``try/except``) so a logging failure can never crash a rank and desync the
collective operations. Installed once by wrapping ``megatron.bridge.training.train.training_log``.
"""

import logging
import math
import os
import time

import torch

from megatron.bridge.utils.common_utils import get_rank_safe


logger = logging.getLogger(__name__)

# Audio seconds consumed on THIS rank since the last logging interval (accumulated per micro-batch).
_AUDIO_SECONDS_INTERVAL: float = 0.0
_FRAME_RATE_MS: int = 10  # LogFbank hop; matches phyagi BatchTracker.frame_rate.

_MLFLOW_STATE: dict[str, bool] = {"tried": False, "enabled": False}
_LAST_LOG_TIME: dict[str, float | None] = {"t": None}
_TOTAL_AUDIO_HOURS: dict[str, float] = {"value": 0.0}


def add_audio_seconds(audio_frames: torch.Tensor | None) -> None:
    """Accumulate this micro-batch's audio duration (``audio_frames.sum() * 10ms``).

    Called once per micro-batch from the forward step. ``audio_frames`` holds the per-utterance
    log-mel frame counts of the packed sequence (the same tensor phyagi's ``BatchTracker`` reads).
    """
    global _AUDIO_SECONDS_INTERVAL
    if isinstance(audio_frames, torch.Tensor) and audio_frames.numel() > 0:
        _AUDIO_SECONDS_INTERVAL += float(audio_frames.sum().item()) * _FRAME_RATE_MS / 1000.0


def _pop_interval_audio_hours() -> float:
    """Sum this rank's interval audio seconds across the data-parallel group, return hours, reset.

    This all-reduce is a **collective**: every rank must call it. It is invoked unconditionally
    inside the (rank-identical) logging gate, before any rank-specific / best-effort work.
    """
    global _AUDIO_SECONDS_INTERVAL
    seconds = _AUDIO_SECONDS_INTERVAL
    _AUDIO_SECONDS_INTERVAL = 0.0
    if torch.distributed.is_initialized() and torch.cuda.is_available():
        from megatron.core import parallel_state

        tensor = torch.tensor([seconds], dtype=torch.float64, device="cuda")
        try:
            group = parallel_state.get_data_parallel_group()
        except Exception:
            group = None
        torch.distributed.all_reduce(tensor, group=group)
        seconds = float(tensor.item())
    return seconds / 3600.0


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _ensure_mlflow() -> bool:
    """Best-effort attach to the AzureML-managed MLflow run (rank 0). Never raises."""
    if _MLFLOW_STATE["tried"]:
        return _MLFLOW_STATE["enabled"]
    _MLFLOW_STATE["tried"] = True
    run_id = os.environ.get("MLFLOW_RUN_ID") or os.environ.get("AZUREML_RUN_ID")
    try:
        import mlflow

        if mlflow.active_run() is None:
            # run_id=None locally starts a fresh run; on AzureML it attaches to the platform run.
            mlflow.start_run(run_id=run_id)
        _MLFLOW_STATE["enabled"] = True
        logger.info(
            "MLflow enabled: tracking_uri=%s run_id=%s",
            mlflow.get_tracking_uri(),
            run_id,
        )
    except Exception:
        logger.warning("MLflow attach failed; metrics will not be logged to MLflow.", exc_info=True)
        _MLFLOW_STATE["enabled"] = False
    return _MLFLOW_STATE["enabled"]


def _log_mlflow(metrics: dict[str, float], step: int) -> None:
    if not _ensure_mlflow():
        return
    try:
        import mlflow

        mlflow.log_metrics(metrics, step=step)
    except Exception:
        logger.warning("MLflow log_metrics failed.", exc_info=True)


def install_speech_obs_logging() -> None:
    """Wrap ``training_log`` to add audio-hours / ppl / throughput to MLflow and a rank-0 line.

    Idempotent. Must be called after ``megatron.bridge.training.train`` is imported (so the module
    global is present) and before the training loop runs.
    """
    from megatron.bridge.training import train as _train_mod

    if getattr(_train_mod.training_log, "_speech_obs_wrapped", False):
        return

    _orig_training_log = _train_mod.training_log

    def _wrapped_training_log(*args, **kwargs):  # noqa: ANN002, ANN003
        report_memory_flag = _orig_training_log(*args, **kwargs)

        # Positional contract from train.py: (loss_dict, total_loss_dict, learning_rate,
        # decoupled_lr, loss_scale, report_memory_flag, skipped_iter, grad_norm, ...,
        # config[10], global_state[11], ...).
        loss_dict = args[0]
        learning_rate = args[2]
        loss_scale = args[4]
        grad_norm = args[7]
        config = args[10]
        global_state = args[11]
        iteration = global_state.train_state.step
        log_interval = config.logger.log_interval

        if iteration % log_interval != 0:
            return report_memory_flag

        # Collective: all ranks all-reduce audio seconds across DP (must precede any rank branch).
        interval_hours = _pop_interval_audio_hours()

        # Everything below is rank-0-only and best-effort (no collectives, cannot desync ranks).
        if get_rank_safe() != 0:
            return report_memory_flag

        _TOTAL_AUDIO_HOURS["value"] += interval_hours

        now = time.time()
        prev = _LAST_LOG_TIME["t"]
        _LAST_LOG_TIME["t"] = now
        step_time = (now - prev) / log_interval if prev is not None else None

        lm_loss = _as_float(loss_dict.get("lm loss"))
        ppl = math.exp(lm_loss) if (lm_loss is not None and lm_loss < 20) else None
        batch_size = config.train.global_batch_size
        seq_length = config.model.seq_length
        samples_per_second = (batch_size / step_time) if step_time else None
        tokens_per_second = (samples_per_second * seq_length) if samples_per_second else None
        gpu_peak_gb = None
        if torch.cuda.is_available():
            gpu_peak_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

        raw_metrics = {
            "train/loss": lm_loss,
            "train/ppl": ppl,
            "train/learning_rate": _as_float(learning_rate),
            "train/grad_norm": _as_float(grad_norm),
            "train/loss_scale": _as_float(loss_scale),
            "train/consumed_audios_in_hours": interval_hours,
            "train/total_consumed_audios_in_hours": _TOTAL_AUDIO_HOURS["value"],
            "train/gpu_peak_gb": gpu_peak_gb,
            "train/samples_per_second": samples_per_second,
            "train/tokens_per_second": tokens_per_second,
        }
        metrics = {k: v for k, v in raw_metrics.items() if v is not None}
        _log_mlflow(metrics, iteration)

        logger.info(
            "[rank0] iter %d | loss %s | ppl %s | audio_h/int %.3f | total_audio_h %.2f | "
            "tok/s %s | gpu_peak %s GB",
            iteration,
            f"{lm_loss:.4f}" if lm_loss is not None else "n/a",
            f"{ppl:.2f}" if ppl is not None else "n/a",
            interval_hours,
            _TOTAL_AUDIO_HOURS["value"],
            f"{tokens_per_second:.0f}" if tokens_per_second else "n/a",
            f"{gpu_peak_gb:.1f}" if gpu_peak_gb is not None else "n/a",
        )
        return report_memory_flag

    _wrapped_training_log._speech_obs_wrapped = True
    _train_mod.training_log = _wrapped_training_log
    logger.info("Installed speech observability logging (audio-hours / ppl / throughput / rank-0 line).")
