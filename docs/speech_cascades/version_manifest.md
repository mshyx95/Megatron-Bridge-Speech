# Speech Cascades Version Manifest

This manifest pins the reproducible starting point for the speech Cascades migration.

| Component | Version |
|---|---|
| Megatron-Bridge | `v0.5.1` |
| Megatron-Bridge commit | `5cb3444c43f7499cf3872b2d46870cf8bc2e00ce` |
| Megatron Core commit | `458c8d0ecafdf6d9e36771600d62ade27f2a67b7` |
| MCore channel | `main` |
| Upstream Bridge base image | `nvcr.io/nvidia/pytorch:26.04-py3` |
| Speech baseline base image | `speechllmregistry.azurecr.io/jing-dev:mega_0812` |
| Speech baseline base digest | `sha256:c9fff4dd6bb38a14a9522f6b49b68c62140dd02eaea48ead979fca9906884268` |
| Speech baseline base ID | `sha256:760c51fa0bb8b6d6dc59b711bdb0687b0e0054bf6a734ffc835026d4aacb5843` |
| Python | `>=3.12,<3.13` |
| Dependency source | committed `uv.lock` |

The Megatron Core source under `3rdparty/Megatron-LM` is read-only. Speech-specific code belongs
under `src/megatron/bridge`, and dependencies must be installed with `uv` inside the pinned
container.

The speech baseline dependency image is built from `docker/Dockerfile.speech-baseline`. It keeps
the production CUDA/PyTorch/Transformer Engine stack and adds only the Python compatibility layer
needed to load Bridge v0.5.1. Source is mounted read-only at runtime.

```text
Development image: megatron-bridge-speech:mega_0812-v0.5.1
Image ID: sha256:37d2b11fc1dab2e2cd9dc3857214071a7a4670ba5c387f415b6e12f0f2840fff
Registry digest: local image; not pushed
Build date: 2026-08-17
Python: 3.12.3
PyTorch: 2.6.0+cu126
CUDA: 12.6
Transformer Engine: 2.4.0+b43596b8
Transformers overlay: 5.8.1
NumPy overlay: 1.26.4
```

The P1 parity image adds the repository-pinned ModelOpt version required by Transformers 5.8.1:

```text
P1 image: megatron-bridge-speech:mega_0812-v0.5.1-modelopt
Image ID: sha256:151b080d9010d655f93cdc912a321ed4876e2dd9ad9c9
ModelOpt overlay: 0.44.0rc5
```

The upstream `26.04` image remains the final native Bridge release gate. It could not be built on
this host because Docker's root filesystem had only 15 GB free. The baseline image is used first
to preserve the current repository's numerical environment.

The uncorrected baseline image's ModelOpt predates Transformers 5.8.1 and patches HF checkpoint
loading with an incompatible method signature. The P1 image overlays ModelOpt 0.44.0rc5 without
reinstalling Torch or CUDA. Importing Bridge followed by loading the real HF 4B model succeeds in
the corrected image, and the official HF-versus-MCore comparison runs end to end.

The production-configured Qwen3.5-4B safetensors index was inspected through read-only Azure Blob
metadata access. Its 32-layer language model uses the `model.language_model.` prefix; all expected
dense Qwen3.5 language tensors are present, and the checkpoint has no extra language tensors or
attention bias tensors. The checkpoint also contains `model.visual.` tensors, which are not part of
the dense language-model conversion gate. Public revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` has the same 738-tensor layout and supplied the P1
weight values.

Build and run the focused P0 gates from the repository root:

```bash
docker build -f docker/Dockerfile.speech-baseline \
	-t megatron-bridge-speech:mega_0812-v0.5.1 .

docker run --rm \
	-v "$PWD:/workspace/Megatron-Bridge:ro" \
	megatron-bridge-speech:mega_0812-v0.5.1 \
	python -m pytest -p no:cacheprovider \
	tests/unit_tests/models/test_qwen35_speech_baseline.py -q

docker run --rm --gpus all --ipc=host \
	-v "$PWD:/workspace/Megatron-Bridge:ro" \
	megatron-bridge-speech:mega_0812-v0.5.1 \
	python -m pytest -p no:cacheprovider \
	tests/unit_tests/models/test_qwen35_training_smoke.py -q
```

The GPU smoke passed on the H100 NVL with two optimizer steps through a native two-layer model:
one GDN layer followed by one full-attention layer. The upstream `26.04` image smoke remains a
separate release gate.
