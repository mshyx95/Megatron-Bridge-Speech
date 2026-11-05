# ELIFT training, data and evaluation

## Start here

There are five independent training entries and one shared evaluation entry.
Edit inputs inside the selected YAML, then run from the repository root:

```bash
amlt run amulet/ELIFT/train/b2.yaml
amlt run amulet/ELIFT/train/p4.yaml
amlt run amulet/ELIFT/train/p4-hindi-2k.yaml
amlt run amulet/ELIFT/train/p5.yaml
amlt run amulet/ELIFT/train/new-hindi-2k.yaml
amlt run amulet/ELIFT/eval/eval.yaml
```

No external exports or stage overrides are required. Shared prefixes live in
[`paths.yaml`](../../amulet/ELIFT/paths.yaml): change output/model/encoder/data
roots once, rather than repeating complete paths in every entry.

| Entry | Training config | Data |
| --- | --- | --- |
| [B2](../../amulet/ELIFT/train/b2.yaml) | [train/b2.yaml](train/b2.yaml) | base |
| [P4](../../amulet/ELIFT/train/p4.yaml) | [train/p4.yaml](train/p4.yaml) | base |
| [P4-Hindi](../../amulet/ELIFT/train/p4-hindi-2k.yaml) | [train/p4-hindi-2k.yaml](train/p4-hindi-2k.yaml) | base + Hindi 2k |
| [P5](../../amulet/ELIFT/train/p5.yaml) | [train/p5.yaml](train/p5.yaml) | base |
| [New Hindi 2k](../../amulet/ELIFT/train/new-hindi-2k.yaml) | [train/new-tier1-hindi-2k-e1r.yaml](train/new-tier1-hindi-2k-e1r.yaml) | new-2k + Hindi 2k |

Each entry shows its training/data files, model name and encoder directories.
Encoder checkpoint, mean and inverse-standard-deviation filenames are derived
from their directory prefixes in the shared command template. P4/P5 still
require real complete native MTP initialization checkpoints; P4-Hindi follows
the completed P4 step4000 output. New Hindi 2k is a fresh projector run.

[`eval.yaml`](../../amulet/ELIFT/eval/eval.yaml) defaults to native OpenASR on this
B2 run's step-4000 checkpoint on 1 x 8 VC02 H100 GPUs (low priority, standard SLA).
Normally change only `RUN_NAME`, `CHECKPOINT_ITER` and `EVAL_SUITE`:
model/checkpoint/tokenizer, data, token budget, job name and results follow
automatically. The tokenizer comes from the prepared HF, including for P5/27B.
Results are separated by suite and checkpoint step. Judging credentials are
never stored in the YAML. See the [submission guide](../../amulet/README.md)
for all shared prefixes, overrides and checkpoint prerequisites.
H200 and VC02 use different storage accounts: synchronize the prepared HF
directory and checkpoint to VC02 before evaluating there, or change the eval
template include to H200 to read the training storage directly.

## Repository layout

- [`data/`](data/): shared catalog data recipes and legacy environment-driven streaming configs.
- [`train/`](train/): catalog training profiles, other training configs and training runners.
- [`reference/`](reference/): unchanged source snapshots.
- [`eval/`](eval/): evaluation and export runners.
- [`amulet/ELIFT/train/`](../../amulet/ELIFT/train/) and
  [`amulet/ELIFT/eval/`](../../amulet/ELIFT/eval/): submission YAMLs, using
  [`amulet/templates/`](../../amulet/templates/) for shared runtime settings.

The catalog below retains the original v0.1 experiment contracts. Additional
streaming and model-specific configs keep their existing filenames and behavior;
unlike the composed catalog recipes, legacy streaming configs can require environment variables.

## Catalog inventory

This inventory migrates the existing S2S Qwen development configurations at
revision `ef224cb1802ef298e010efbb4251a00adfcfe846`. The
[catalog](catalog.yaml) accounts for all 18 source YAML files by filename and
SHA-256, and maps 17 experiment stages to their data and training profiles.

## What is checked in

- [Reference configs](reference/): the 18 original source model, encoder, loss,
  optimizer, data and continuation configs, copied byte-for-byte (SHA-256 in the
  catalog). These retain the **source DeepSpeed schema**; they are not accepted
  directly by Megatron Bridge's fine-tuning CLI.
- [Native training profiles](train/): 15 catalog YAML profiles accepted by
  [finetune.py](../../examples/models/speech_cascades/finetune.py).
  The two Flash-2609 encoder variants remain reference-only: the current
  SpeechCascades bridge builds a Conformer, not that Flash encoder.
- [Shared mixture](data/mixture.yaml): the common 17 sampling slots, weights,
  filters, prompts and packing settings, without a model-specific tokenizer.
  [Source mappings](data/sources.yaml) independently select original/new inhouse
  data and optional Hindi. The launcher supplies the model tokenizer and renders
  a standalone config with absolute `/datablob1/...` paths and no environment
  dependencies. KD remains enabled in the collator and audio transforms.
- [Data preparation](../../examples/models/speech_cascades/prepare_elift_data.py):
  nested 2k/5k/10k/20k sampling and per-language publication. This does not submit
  training/generation jobs, invoke a paid model, or download data.

No job IDs, checkpoints, manifests, audio, generated answers or experiment
results are checked in. Data and tokenizer paths are the original absolute
datablob paths; files inside this project use relative paths.

## Data recipes

Data is selected by content, not by B1/B2/B3 model names:

1. `base`: original OpenASR, translation and inhouse sources. The ASR, rewrite and
   continuation branches retain their original 1:1:1 weights; translation slots
   remain separate and unchanged.
2. `base` plus `--hindi 2k` or `--hindi 5k`: append Hindi to the multilingual
   inhouse slot in **all three branches**, without changing their weights.
3. `new-2k` / `new-5k`: replace only inhouse source specs across the three
   branches. OpenASR, translation and the earnings spec remain unchanged.
   Hindi is optional and independent of the selected new-data recipe.

The existing mixed-language tier convention is retained: `new-2k` uses English
5k and other Tier1 languages 2k; `new-5k` uses English 10k and other Tier1
languages 5k. Historical stages retain their matching Hindi 2k/5k additions,
transport settings and encoder-specific streaming compression.

Sampling/publication supports 2k, 5k, 10k and 20k. Only the two historical new-data
training recipes currently have verified complete ASR/rewrite/continuation spec
mappings. New 10k/20k training recipes must supply real published spec mappings
and an explicit English tier; the composer rejects unmapped recipes instead of
guessing manifest filenames or falling back to another tier.

Render a standalone data config without preparing models or submitting jobs:

```bash
: "${TOKENIZER:?Set the tokenizer matching the training model}"
: "${OUTPUT:?Set a fresh experiment directory}"
uv run python examples/models/speech_cascades/prepare_elift_data.py config \
  --config-root projects/ELIFT/data --recipe new-2k --hindi 2k \
  --tokenizer "${TOKENIZER}" --output "${OUTPUT}/data.yaml"
```

Omit `--hindi` for a recipe without Hindi. The mixture template deliberately has
a mandatory tokenizer placeholder; do not pass it directly to the trainer.
Re-rendering the same output is allowed only when its config is unchanged.
Legacy `data_*.yaml` entrypoints remain for existing S1/KD scripts, including
different task subsets and environment overrides; this consolidation does not
silently replace those interfaces.

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

## How the training entry runs

The original jobs used DeepSpeed `ds_train_advanced.py`. Each of the five
[training entries](../../amulet/ELIFT/train/) passes its visible inputs through
the [shared execution template](../../amulet/templates/elift-train.yaml) directly to
[launch_elift_stage.py](../../examples/models/speech_cascades/launch_elift_stage.py),
without looking up a stage in the catalog or launch mapping:

1. Rank 0 composes `DATA_CONFIG` with `DATA_RECIPE`, optional `DATA_HINDI` and
   the selected base-model tokenizer into `OUTPUT_ROOT/RUN_NAME/data.yaml`.
2. Rank 0 prepares `OUTPUT_ROOT/RUN_NAME/initial_hf`: clean LLM + requested encoder and
   normalization + random MLP projector (seed 1234). Provenance is checked
   before reusing an existing prepared model.
3. Every rank runs the native fine-tuning CLI with `TRAIN_CONFIG`, the resolved
   data YAML and prepared model. Preparation failures are propagated to all ranks.

Edit `TRAIN_CONFIG` for optimizer, steps, batch, freeze and KD/MTP settings.
Data selection is independent: edit `DATA_RECIPE` and `DATA_HINDI`, not a model
or experiment name. `DATA_RECIPE: none` passes a standalone legacy data YAML
unchanged; its referenced environment variables must be defined in the same
submission YAML. The default composed recipe has no such external dependencies.

**P2, P4, P5, P6 and P4 Hindi 2k require a complete native Megatron checkpoint**
with the physical MTP layers. Select the corresponding training config and set
`INIT_CHECKPOINT` in the submission YAML; the launcher refuses a missing checkpoint.
For S1/LoRA, `PRETRAINED_CHECKPOINT` is the alternative native weight source.
DeepSpeed checkpoints are not valid substitutes. Use a fresh `OUTPUT_ROOT` or `RUN_NAME` for
each experiment; initialization starts a fresh optimizer/scheduler, while a
preexisting training output can trigger the existing loop's resume behavior.

The historical `--stage` planner and [launch.yaml](launch.yaml) remain only for
reproducing the catalog's original contracts, not as the public launch workflow.
They can still be inspected without running a job:

```bash
uv run python examples/models/speech_cascades/launch_elift_stage.py --stage b2 --dry-run
```

Only config/CPU checks have been performed for this refactor, not GPU training.

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
