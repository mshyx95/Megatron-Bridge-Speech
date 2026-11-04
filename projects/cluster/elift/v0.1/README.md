# ELIFT v0.1 training and data configurations

This inventory migrates the existing S2S Qwen development configurations at
revision `ef224cb1802ef298e010efbb4251a00adfcfe846`. The
[catalog](catalog.yaml) accounts for all 18 source YAML files by filename and
SHA-256, and maps 17 experiment stages to their data and training profiles.

## What is checked in

- [Reference configs](reference/): the original model, encoder, loss, optimizer,
  data and continuation contracts, with private paths replaced by required
  environment variables. These retain the **source DeepSpeed schema**; they are
  not accepted directly by Megatron Bridge's fine-tuning CLI.
- [Native training profiles](train/): 15 flat YAML profiles accepted by
  [finetune.py](../../../../examples/models/speech_cascades/finetune.py).
  The two Flash-2609 encoder variants remain reference-only: the current
  SpeechCascades bridge builds a Conformer, not that Flash encoder.
- [Standalone data configs](data/): six distinct, complete source data snapshots.
  All 17 sampling slots, repeated spec references, ratios, filters, prompts and
  packing settings are retained. KD is explicitly enabled in the collator and
  audio transforms so the speech student receives text-only teacher inputs.
- [Data preparation](../../../../examples/models/speech_cascades/prepare_elift_data.py):
  reusable explicit-path config materialization, nested 2k/5k/10k/20k sampling,
  and per-language publication. This does not submit training/generation jobs,
  invoke a paid model, or download data.

No source accounts, workspace bindings, job IDs, checkpoints, manifests, audio,
generated answers or experiment results are checked in. Set deployment-specific
paths outside this repository.

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
Opaque immutable spec filenames are retained under explicit 2k/5k spec-root
arguments; do not replace them with different manifests.

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

### Deployment paths

Deployment data configs use literal, existing absolute datablob paths, including
tokenizer directories. Files inside this project use paths relative to the
repository root. Do not invent a new mount alias or require environment variables
to locate remote data.

Run the checked-in `materialize` subcommand from the repository root. It selects
the stage's complete data snapshot from the catalog, takes explicit path
arguments, and writes a fully resolved YAML without environment variables or
other interpolation. No S2S source checkout, private binding file, or previously
generated local config is required. Use the tokenizer matching the selected
stage's LLM family and size; the script checks paths, not model-weight parity.

Only arguments used by the selected stage are required. Missing arguments are
reported by flag name even if the corresponding environment variable is set.

| Argument | Required contents |
| --- | --- |
| `--tokenizer-path` | Matching tokenizer directory; absolute if remote, or relative inside this project |
| `--data-spec-root` | DataSpecs root containing `mlang_s2/` |
| `--rewrite-root` | Parent of the original `asr_nothink/` and `asr_continuation/` directories |
| `--tier1-root` | Published per-language Tier1 root |
| `--hindi-root` | Published Hindi root containing `configs/` and answer overlays |
| `--new-data-2k-spec-root` | Exact immutable corrected 2k directory containing `963dfa3f0d6b8b28.json` |
| `--new-data-5k-spec-root` | Exact immutable corrected 5k directory containing `specs/fe910f419370af2f9a02.json` |
| `--data-mount-root` | Actual mounted data prefix used by storage-API specs |

The commands below are examples: replace every `/absolute/path/to/...` argument
with an existing directory on your mounted datablob. These are not proposed mount
aliases. B2/B3/P4/P5 need the tokenizer, DataSpecs and rewrite roots. The original
Hindi stages also need `--hindi-root`. The new-data stages additionally need the
Tier1 root, mounted data prefix, and their corresponding immutable spec root.

Example B2 materialization and launch, inside a dependency-complete training
container:

```bash
uv run python examples/models/speech_cascades/prepare_elift_data.py materialize \
  --stage b2 \
  --tokenizer-path /absolute/path/to/Qwen3.5-9B \
  --data-spec-root /absolute/path/to/DataSpecs \
  --rewrite-root /absolute/path/to/asr_rewrite \
  --output-config projects/cluster/.speech-elift-b2.local.yaml

uv run --no-project --python "$(command -v python)" python -m torch.distributed.run \
  --nproc_per_node=2 examples/models/speech_cascades/finetune.py \
  --hf-path /absolute/path/to/prepared-b2-speech-hf \
  --train-config-yaml projects/cluster/elift/v0.1/train/b2.yaml \
  --streaming-config-yaml projects/cluster/.speech-elift-b2.local.yaml \
  --output-dir runs/elift-b2 --name b2
```

For the new 5k mixture, use all required roots explicitly:

```bash
uv run python examples/models/speech_cascades/prepare_elift_data.py materialize \
  --stage new-tier1-hindi-5k-e1r-8k \
  --tokenizer-path /absolute/path/to/Qwen3.5-9B \
  --data-spec-root /absolute/path/to/DataSpecs \
  --rewrite-root /absolute/path/to/asr_rewrite \
  --tier1-root /absolute/path/to/tier1_per_language \
  --hindi-root /absolute/path/to/hindi \
  --new-data-5k-spec-root /absolute/path/to/immutable-corrected-5k-directory \
  --data-mount-root /absolute/path/to/mounted-datablob \
  --output-config projects/cluster/.speech-elift-new-5k.local.yaml
```

The script validates every referenced dataset-spec file before writing and
refuses to overwrite an existing output. It preserves all sampling slots,
repeated spec references, filters, weights and KD settings. It does not check
chunk/answer completeness or qualify checkpoints. Flash-encoder stages remain
reference-only, even when their data paths validate.

Generated `projects/cluster/.speech-*.local.yaml` files are Git-ignored because
they contain deployment-specific paths. Share the script and stage templates,
not personal account paths. The checked-in `data/` and `reference/` files remain
parameterized migration templates for source comparison; use materialized data
YAMLs for deployment, not those templates directly.

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
  --source-config projects/cluster/.speech-unsampled-tier1.local.yaml \
  --mode tier1_per_language --output-root runs/tier1-sampled \
  --targets 2000 5000 10000 20000 --languages en

uv run python examples/models/speech_cascades/prepare_elift_data.py sample \
  --source-config projects/cluster/.speech-unsampled-hindi.local.yaml \
  --mode hindi --hindi-spec /absolute/path/to/full-hindi-spec.json \
  --tokenizer /absolute/path/to/matching-tokenizer --output-root runs/hindi-sampled

uv run python examples/models/speech_cascades/prepare_elift_data.py publish \
  --source-root runs/tier1-sampled/en --output-root /absolute/path/to/fresh-published-root \
  --mount-root /absolute/path/to/mounted-datablob --mode tier1_per_language --language en
```

The sampling examples require separately supplied full, unsampled source configs;
materializing a stage's existing 2k/5k mixture does not reconstruct unsampled data.

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
CLI wiring, all data slots and teacher transforms, explicit-path materialization,
nested sampling, partial-set failures, publication isolation, shared-20k
answer paths and Hindi filtering. They use synthetic local fixtures.

No ELIFT GPU training, checkpoint conversion/resume, full-data preparation,
quality evaluation or Flash encoder qualification was run for this migration.
Earlier vLLM/MTP inference results do not validate these training configurations.
