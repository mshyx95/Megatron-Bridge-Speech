"""Activate the explicit launcher's read-only overlay in spawned Python workers."""

import os


if os.environ.get("SPEECH_RUNTIME_MODE"):
    from speech_eval.runtime.source_overlay import install

    install()
