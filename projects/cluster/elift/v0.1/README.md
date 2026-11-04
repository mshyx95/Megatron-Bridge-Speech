# ELIFT v0.1 training and data configurations

This inventory migrates the existing S2S Qwen development configurations at
revision `ef224cb1802ef298e010efbb4251a00adfcfe846`. The
[catalog](catalog.yaml) accounts for all 18 source YAML files by filename and
SHA-256, and maps 17 experiment stages to their data and training profiles.

## What is checked in

- [Reference configs](reference/): the 18 original source model, encoder, loss,
  optimizer, data and continuation configs, copied byte-for-byte (SHA-256 in the
  catalog). These retain the **source DeepSpeed schema**; they are not accepted
  directly by Megatron Bridge's fine-tuning CLI.
- [Native training profiles](train/): 15 flat YAML profiles accepted by
  [finetune.py](../../../../examples/models/speech_cascades/finetune.py).
  The two Flash-2609 encoder variants remain reference-only: the current
  SpeechCascades bridge builds a Conformer, not that Flash encoder.
- [Standalone data configs](data/): eight complete data snapshots with the
  source's absolute `/datablob1/...` paths and no environment variables. `b1`
  (Qwen3.5-4B tokenizer), `b2` (Qwen3.5-9B tokenizer: B2, C, E1-r, P2, P3, P4) and
  `b3` (Qwen3.8-27B tokenizer: B3, B4, P5, P6) share the same data specs and
  differ only in tokenizer. All 17 sampling slots, repeated spec references,
  ratios, filters, prompts and packing settings are retained. KD is explicitly
  enabled in the collator and audio transforms so the speech student receives
  text-only teacher inputs.
- [Data preparation](../../../../examples/models/speech_cascades/prepare_elift_data.py):
  nested 2k/5k/10k/20k sampling and per-language publication. This does not submit
  training/generation jobs, invoke a paid model, or download data.

No job IDs, checkpoints, manifests, audio, generated answers or experiment
results are checked in. Data and tokenizer paths are the original absolute
datablob paths; files inside this project use relative paths.

## Stage contracts

All stages freeze the language model, use main-path **pure KL** with weight 1,
temperature 2 and KL chunk size 128, and do not use LoRA. All encoders are frozen
except P3. "Random MLP" means a fresh projector, not a restored trained E1-r/P4
projector. LR warmup is 100 steps; global/micro batches are 512/1; context is 8192;
checkpoint cadence is 400. Refer to the catalog for the exact source overlays.

| Stage | LLM | Initial encoder / checkpoint | Steps | LR | Additional objective |
| --- | --- | --- | ---: | ---: | --- |
| B1 | Qwen3.5-4B | Phi-4 MM 24L, random MLP | 4000 | 1e-4 | None |
| B2 | Qwen3.5-9B | Phi-4 MM 24L, random MLP | 4000 | 1e-4 | None |
| B3 | Qwen3.8-27B | Phi-4 MM 24L, random MLP | 4000 | 1e-4 | None |
| B4 | Qwen3.8-27B | checkpoint-22L with 23 blocks, random MLP | 4000 | 1e-4 | None |
| C | Qwen3.5-9B | checkpoint-22L with 23 blocks, random MLP | 4000 | 1e-4 | None |
| E1-r | Qwen3.5-9B | S1-2605 encoder, random MLP | 4000 | 1e-4 | None |
| E1-r Hindi 2k | Qwen3.5-9B | S1-2605 encoder, random MLP; original + Hindi data | 4000 | 1e-4 | None |
| E1-r 2609 / Hindi 2k | Qwen3.5-9B | Flash-2609 encoder; reference-only | 4000 | 1e-4 | None |
| P2 | Qwen3.5-9B | checkpoint-22L, random MLP + native ExpandedMTP4 | 4000 | 1e-4 | MTP CE, weight 0.1 |
| P3 | Qwen3.5-9B | checkpoint-22L, train encoder + random MLP | 4000 | 1e-4 | None |
| P4 | Qwen3.5-9B | **Full trained E1-r step4000** + native ExpandedMTP4 | +4000 | 4e-5 | MTP CE, weight 0.1 |
| P5 | Qwen3.8-27B | **Full trained B3 step4000** + native ExpandedMTP4 | +4000 | 4e-5 | MTP CE, weight 0.1 |
| P6 | Qwen3.8-27B | **Full trained B3 step4000**, not P5, + ExpandedMTP7 | +4000 | 4e-5 | MTP CE, weight 0.1 |
| P4 Hindi 2k | Qwen3.5-9B | **Full trained P4 step4000**, preserving all four MTP layers | +4000 | 1e-5 | MTP CE, weight 0.1 |
| New Tier1 + Hindi 2k | Qwen3.5-9B | Clean LLM, frozen S1-2605 encoder, random MLP | 4000 | 1e-4 | None |
| New Tier1 + Hindi 5k | Qwen3.5-9B | Clean LLM, frozen S1-2605 encoder, random MLP | 8000 | 1e-4 | None |

The 2k new-data experiment uses English 5k and other Tier1 languages plus Hindi
2k. The 5k experiment uses English 10k and other Tier1 languages plus Hindi 5k,
with corrected Qwen-generated answer spec paths. Both preserve the source's
17 sampling slots and auxiliary AST entries. They are **not** P4 continuations.
Opaque immutable spec filenames are retained at their original absolute paths;
do not replace them with different manifests.

The four-layer MTP depth weights are `[0.30, 0.25, 0.25, 0.20]`. P6 retains seven
equal depth weights. MTP distillation weight is zero: these are main KL + MTP CE,
not the legacy 0.75-KL/0.25-CE recipe.

## Required runtime inputs and limitations

Use `speechllmregistry.azurecr.io/speechllm:megratron-latest` for training, not the
vLLM inference image. Supply a matching **prepared speech HF configuration** and,
where applicable, a complete Megatron checkpoint. Raw reference DeepSpeed
checkpoint paths are not Megatron checkpoints.
Mount or upload this repository checkout and run its entrypoint; the previously
published image does not automatically contain these new configs or CLI flags.

For B2/B3 and other no-MTP stages, preparation must preserve the exact clean LLM
and requested encoder/normalization, initialize a random MLP, and contain no
LoRA or MTP. An existing trained ASR/LoRA export is not equivalent.

For P2/P4/P5/P6, prepare native MTP layer 0 from the **matching LLM family/size**,
copy-initialize independent extra MTP layers, then restore the full learned
source weights where required. P4 Hindi restores **all** trained P4 MTP layers
without reinitializing them. The native profiles disable
`mtp_init_extra_layers_from_first`: SpeechCascadesModel's decoder-layer-copy
initializer is not the reference's native-MTP-layer-zero initializer.

These MTP profiles therefore require a separately validated complete native
Megatron checkpoint with the requested physical MTP layers. This migration
does **not** establish DeepSpeed-to-native-MTP weight conversion or add missing
HF MTP mappings. Do not cold-start these profiles from a plain HF LLM or an
incomplete checkpoint, and do not claim lossless conversion.

Native profiles retain source Adam betas `(0.9, 0.95)`, epsilon `1e-7`, weight
decay `0.01`, and linear warmup/decay through the existing recipe defaults.
Megatron distributed Adam and DeepSpeed ZeRO-1 execution are different;
configuration checks do not establish optimizer-state or numerical parity.

The data configs require the `/datablob1` mount used by the original training jobs.

## Launching a stage (native Megatron-Bridge, not DeepSpeed)

The original jobs used DeepSpeed `ds_train_advanced.py`. Use the native launcher instead:
[launch_elift_stage.py](../../../../examples/models/speech_cascades/launch_elift_stage.py) reads the stage's
training profile and data snapshot from [catalog.yaml](catalog.yaml), and the clean LLM, frozen encoder and
continuation checkpoints from [launch.yaml](launch.yaml). Rank 0 builds the prepared speech HF (clean LLM + frozen
encoder + random MLP) under `<output_root>/<stage>/initial_hf`; every rank then runs `finetune.py` with
`--train-config-yaml`, `--streaming-config-yaml` and a fresh output directory.

Check any stage without a GPU:

```bash
uv run python examples/models/speech_cascades/launch_elift_stage.py --stage p4 --dry-run
```

Submit one stage on 32 H200 GPUs (4 nodes x 8, `speechllm:megratron-latest`, global batch 512):

```bash
STAGE=b2 amlt run projects/cluster/amlt_elift_h200.yaml elift-b2 -d "ELIFT B2 native"
STAGE=p4 amlt run projects/cluster/amlt_elift_h200.yaml elift-p4 -d "ELIFT P4 native"
```

Valid stages are the 15 trainable catalog stages (all except the two reference-only Flash-2609 stages).
B1-B4, C, E1-r, P3 and the two new-data stages start from the clean LLM and need nothing else.
**P2, P4, P5, P6 and P4 Hindi 2k are continuation stages**: set `init_checkpoints.<stage>` in `launch.yaml` to an
existing, validated **complete native Megatron checkpoint** (with the physical MTP layers) before submitting; the
launcher refuses to start without it. The source DeepSpeed checkpoints are not Megatron checkpoints. These
launchers have not been run on GPUs for this migration.

Example B2 launch, inside a dependency-complete training container:

```bash
: "${MODEL_HF:?Set a prepared clean B2 speech HF directory}"
: "${OUTPUT:?Set a fresh experiment directory}"
uv run --no-project --python "$(command -v python)" python -m torch.distributed.run \
  --nproc_per_node=2 examples/models/speech_cascades/finetune.py \
  --hf-path "${MODEL_HF}" \
  --train-config-yaml projects/cluster/elift/v0.1/train/b2.yaml \
  --streaming-config-yaml projects/cluster/elift/v0.1/data/b2.yaml \
  --output-dir "${OUTPUT}" --name b2
```

Select data paths from the catalog rather than inferring them from stage names.
Use a fresh output for each source experiment; continuation initializes weights
but starts a fresh optimizer/scheduler. A preexisting output may trigger the
existing training loop's resume behavior.

## 2k / 5k / 10k / 20k data preparation

Run this only on the designated real-data preparation host with mounted input
data. `sample` retains the reference sampler's SHA-based chunk shuffle and
weighted fair prefix selection. Tier1 per-language sampling requires both
feature and transcription files. Hindi hours are measured after the source
filters: at most 4000 frames, nonempty transcription, at most 800 labels, and
`lid_mix_label` in `Mix`/`Native`. Hindi requires an explicit source spec and
matching tokenizer; no account-specific source is built in.

```bash
uv run python examples/models/speech_cascades/prepare_elift_data.py sample \
  --source-config "${TIER1_SOURCE_CONFIG:?Set an unsampled full Tier1 source config}" \
  --mode tier1_per_language --output-root "${SAMPLED_ROOT:?}" \
  --targets 2000 5000 10000 20000 --languages en

uv run python examples/models/speech_cascades/prepare_elift_data.py sample \
  --source-config "${HINDI_SOURCE_CONFIG:?}" \
  --mode hindi --hindi-spec "${HINDI_SOURCE_SPEC:?}" \
  --tokenizer /datablob1/users/ruchaofan/Phi-omni-checkpoints/language_ckpt/Qwen3.5-9B/ --output-root "${HINDI_SAMPLED_ROOT:?}"

uv run python examples/models/speech_cascades/prepare_elift_data.py publish \
  --source-root "${SAMPLED_ROOT:?}/en" --output-root "${PUBLISHED_ROOT:?}" \
  --mount-root "${PUBLISHED_MOUNT_ROOT:?}" --mode tier1_per_language --language en
```

Publication creates source-namespaced no-think and continuation answer spec
overlays. All four sizes reference the shared 20k answer superset, but their
sampled manifests stay distinct. A model-specific `--variant-suffix` adds
isolated answer paths without modifying base manifests. Actual answer generation
is a separate operation and must complete before those overlays are trainable.
The example selects English explicitly. Without `--languages`, per-language
sampling requires all nine Tier1 languages and fails if any is missing.

This port refuses to overwrite existing files. Insufficient hours retain partial
sampling evidence and raise an error; incomplete samples cannot be published as
full-size sets. Those stricter checks prevent a success-shaped undersized dataset.
Source namespace IDs remain stable when a smaller subset omits a source, so
shared 20k answers cannot be accidentally read from another source's directory.
Absolute source paths participate in the original shuffle hash, so relocation
does not guarantee the same chunk membership. Existing immutable 2k/5k specs
must be reused for exact experiment reproduction.

The source repo contained complete 2k/5k new-training mixtures, but no complete
10k/20k new-training mixture YAML. The checked-in sampling/publishing mechanism
supports all four sizes. **No 10k/20k training proportions or experiment results
have been invented.**

## Validation boundary

Focused checks cover the full inventory, source-to-native stage contracts,
CLI wiring, all data slots and teacher transforms, environment resolution,
nested sampling, partial-set failures, publication isolation, shared-20k
answer paths and Hindi filtering. They use synthetic local fixtures.

No ELIFT GPU training, checkpoint conversion/resume, full-data preparation,
quality evaluation or Flash encoder qualification was run for this migration.
Earlier vLLM/MTP inference results do not validate these training configurations.
