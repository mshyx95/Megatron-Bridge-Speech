#!/usr/bin/env bash
set -euo pipefail

RUNTIME_PYTHON=${RUNTIME_PYTHON:-/opt/venv/bin/python}

uv --no-config pip install --python "$RUNTIME_PYTHON" --no-deps --reinstall \
    transformers==5.16.1 tokenizers==0.23.1 safetensors==0.8.0

uv --no-config run --no-project --python "$RUNTIME_PYTHON" - <<'PY'
import logging
from importlib import metadata

from packaging.requirements import Requirement

logging.basicConfig(level=logging.INFO)
for name, expected in (
    ("transformers", "5.16.1"),
    ("tokenizers", "0.23.1"),
    ("safetensors", "0.8.0"),
):
    installed = metadata.version(name)
    if installed != expected:
        raise RuntimeError(f"{name}: expected {expected}, imported distribution is {installed}")
    for declared in metadata.requires(name) or []:
        requirement = Requirement(declared)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        dependency_version = metadata.version(requirement.name)
        if dependency_version not in requirement.specifier:
            raise RuntimeError(f"{name} requires {requirement}, found {dependency_version}")
    logging.info("Runtime dependency verified: %s==%s", name, installed)

from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

logging.info("Qwen4Exp config and model imports passed")
PY