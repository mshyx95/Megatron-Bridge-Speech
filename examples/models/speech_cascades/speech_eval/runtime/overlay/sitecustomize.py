"""Activate the explicit launcher's read-only overlay in spawned Python workers."""

import os


if os.environ.get("SPEECH_RUNTIME_MODEL") == "qwen4_audio":
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "Qwen4ExpAudioForCausalLM", "speech_eval.runtime.qwen4_audio:Qwen4ExpAudioForCausalLM"
    )
elif os.environ.get("SPEECH_RUNTIME_MODE"):
    from speech_eval.runtime.source_overlay import install

    install()
