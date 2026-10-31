# Speech Cascades Migration Execution Plan

The existing Megatron-LM training path remains the production fallback until all cutover gates pass.
Work proceeds in this repository without modifying `3rdparty/Megatron-LM`.

The machine-readable acceptance contract is
`tests/functional_tests/models/speech_cascades/acceptance_baseline.yaml`. Its M2-M6 thresholds
match the current Megatron-LM verification scripts; Bridge-specific TP and conversion checks are
additional gates and do not replace those baseline checks.

## Gates

| Gate | Deliverable | Acceptance |
|---|---|---|
| P0 | Pinned source, container, and upstream smoke run | Qwen3.5 provider builds and two iterations complete |
| P1 | Native Qwen3.5 text conversion and parity | Exact weight round trip; BF16 cosine similarity above `0.9999`; TP1/TP2 loss within `0.1%` |
| P2 | Legacy DeepSpeed checkpoint importer | Every active tensor mapped once; TP1 and TP2 load successfully |
| P3 | Cascades encoder package | Reference audio embeddings are bit-identical or have documented max error below `1e-3` |
| P4 | Reusable audio-language provider | Full step-zero loss within `1%` of the accepted baseline |
| P5 | Speech chunk data and storage API | Deterministic local tensors match; remote streaming produces non-empty batches |
| P6 | Recipe, short training, and resume | TP1, TP2, and TP2/PP2 train eight steps and resume correctly |
| P7 | Inference and HF checkpoint export | Existing five-sample inference bar and Bridge round trip pass |
| P8 | Cluster and larger-model cutover | Storage soak, checkpoint resume, and Qwen3.5-9B native parallel training pass |

## Native Integration Direction (supersedes DeepSpeed-at-runtime)

The audio-language model is built the **repo-native way**, following the existing
`src/megatron/bridge/models/qwen3_asr/` pattern (provider + model + bridge), not DeepSpeed runtime
loading:

- **Provider** (`SpeechCascadesModelProvider(GPTModelProvider)`): applies the Qwen3.5 GDN language
  configuration (via `qwen35_bridge._apply_qwen35_common_config`) and holds the Cascades audio
  encoder config plus `audio_token_id`. `provide()` builds a Megatron-Core `GPTModel` language model
  and attaches the audio module.
- **Model** (`SpeechCascadesModel(MegatronModule)`): composes the vendored
  `CascadesAudioEmbedding` (audio encoder + projection) with the Megatron Qwen3.5 language model,
  merging audio embeddings into the token stream at `audio_token_id` positions
  (`combined_embeddings[audio_mask] = audio_embeds`), then running
  `language_model(decoder_input=combined_embeddings, ...)` — the same mechanism Qwen3-ASR uses.
- **Bridge** (`@MegatronModelBridge.register_bridge(...)`): maps weights with `AutoMapping`,
  `QKVMapping`, `GatedMLPMapping`, GDN mappings for the LM and `ReplicatedMapping` for the audio
  module, so weights flow through `AutoBridge` and Megatron distributed checkpoints.

The legacy DeepSpeed checkpoint is only a **one-time import source**: it is converted once into the
native representation. The P2 importer and the `CascadesAudioEmbedding.load_full_model_weights`
helper are conversion/validation utilities, not the training runtime path.

The native provider, model, and bridge are implemented and validated for construction:
`src/megatron/bridge/models/speech_cascades/{provider,model,bridge}.py`. On GPU (a 4-layer,
real-dimension build) the provider builds a Qwen3.5 GDN hybrid `GPTModel` (layers use the
experimental gated-delta-net attention spec — layers 0–2 GDN with
`self_attention.{in_proj,conv1d,out_norm}`, layer 3 standard attention with `linear_qkv`) plus the
`CascadesAudioEmbedding` audio module, and every `megatron_param` target of the bridge mappings
resolves to a real parameter. Setting
`transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec` on the
provider is required; otherwise the language model is built with standard attention only. Remaining
P4 work: a one-time 40k→HF converter, then `AutoBridge.from_hf_pretrained` → native model →
step-zero loss versus the `0.2064` reference.

## Immediate P0/P1 Tasks

- [x] Clone Megatron-Bridge `v0.5.1` with its pinned MCore submodule.
- [x] Create the `speech-cascades-migration` development branch.
- [x] Record the source and base-image versions.
- [x] Build the speech baseline dependency image from `docker/Dockerfile.speech-baseline`.
- [x] Record the immutable base digest and local image ID.
- [x] Run a focused Qwen3.5 `AutoBridge` provider construction test.
- [x] Run two native Qwen3.5 GPU training iterations with one GDN and one full-attention layer.
- [x] Identify the Qwen3.5-4B HF checkpoint from the production runtime configuration.
- [x] Inventory the real 4B safetensors index without loading tensor values.
- [ ] Build the upstream `docker/Dockerfile.ci` image when Docker storage permits.
- [ ] Run the smallest upstream Qwen3.5 smoke recipe available in this release.
- [x] Run the single-GPU exact weight round-trip tool.
- [ ] Run the TP2 weight round-trip tool.
- [x] Run BF16 HF-versus-MCore forward parity.

The 4B checkpoint is a conditional-generation checkpoint whose language tensors use the
`model.language_model.` prefix and whose visual tensors use `model.visual.`. Metadata inventory
found 738 total tensors across 32 language layers, with no missing language-model mappings, no
unmapped language-model tensors, and no attention bias tensors. Visual tensors are outside the
dense language-model P1 mapping scope.

The pinned public checkpoint revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` was used to
materialize the same 738-tensor checkpoint and extract its 441 text tensors. The FP32 conversion
path round-tripped all 441 tensors exactly with no key or shape differences. The production BF16
path preserved 417 tensors exactly; the 24 FP32 GDN output-norm tensors were quantized to BF16 with
maximum absolute error `0.00390625` and minimum per-tensor cosine similarity `0.99999755`.

The official one-step comparison produced the same next token and Top-5 tokens for HF and MCore.
Last-token logits had cosine similarity `0.99994534`, maximum absolute difference `0.125`, and mean
absolute difference `0.0192648`. This passes the P1 cosine threshold. TP2 conversion and the TP1/TP2
loss comparison remain open because this node exposes only one H100 GPU.

## P3 Evidence

The Cascades audio encoder was vendored verbatim from the read-only Megatron-LM baseline into
`src/megatron/bridge/models/speech_cascades/encoder/audio_blocks/cascade_encoder/` (the
`ConformerEncoder`, its `utils`, and `activation_checkpointing`). Only two mechanical changes were
made: the two intra-package imports were made relative, and the optional `backoff` retry dependency
was guarded so the module imports without it. The MIT license headers are preserved, and the subtree
is excluded from ruff to keep it byte-identical to the baseline. The unused vendored logger and the
whisper/w2vbert/qformer siblings were not migrated; the audio projection/adapter belongs to gate P4.

The standalone initialization contract discovered from the baseline is: construct `ConformerEncoder`
(passing `encoder_embedding_config={"input_size": 80}`), then call `post_init({})` to build the
`MeanVarianceNormLayer`.

Parity was proven in two isolated processes in the pinned container by feeding the **same input tensor**
to both the baseline encoder (`PYTHONPATH=/baseline`) and the vendored encoder, loading the baseline's
exact state dict into the vendored copy. Two configurations were checked:

- The **real production config** from `projects/training/configs/qwen3-5-4b-s2-main-run-002.yaml`
  (`input_layer=conv2d`, streaming `chunk_size=10`, `relative_k=True`, `activation=swish`,
  `attention_dim=1280`, `num_blocks=12`, `time_reduction=8`; 502,733,080 parameters). Output shape
  `(2, 31, 1280)`, `torch.equal == True`, maximum absolute difference `0.0`.
- A small stand-in on the **same code path** (conv2d streaming + relative_k + swish) used by the
  committed regression test `tests/unit_tests/models/speech_cascades/test_encoder_parity.py`, which
  pins the deterministic output signature so future edits cannot silently change the numerics.

This bit-identical result meets the P3 gate for the encoder computation itself.

### Real-data parity (captured production batch)

The vendored encoder was also validated against the **real captured production batch**
`/nfs/yuxuanhu/tmp/m1_capture/batch.pt` (`inputs_embeds` = real log-mel of shape `(36, 1356, 80)`
with a real variable-length `audio_attention_mask`; 45,819 of 48,816 frames unmasked). Feeding this
real input through the baseline and the vendored encoder under the real production config, with the
same shared weights, produced output shape `(36, 168, 1280)` with `torch.equal == True`, maximum
absolute difference `0.0`.

### Real-weights + real-data parity

The strongest check loads the **real pretrained Cascades encoder** from
`/mnt/highperf01safn/data/users/ruchaofan/cascades_speech_encoder/S2S_streaming/ASR_499M_800ms_H200/model.checkpoint`
(plus `mean.vec` / `Invstd.vec`) via the production `post_init` path, confirms the weights actually
populate (a probe parameter sum changes from `-25.5278` to `-2.6725`), and feeds the real log-mel
batch. Baseline versus vendored gave output shape `(36, 168, 1280)`, `torch.equal == True`, maximum
absolute difference `0.0`. The vendored encoder therefore reproduces the baseline bit-for-bit with
**real trained weights, real production data, and the real production config**.

### What P3 has NOT yet validated

The captured reference `/nfs/yuxuanhu/tmp/m3_capture/ref_audio.pt` (`audio_features` `(36, 168, 2560)`)
was **reproduced end-to-end and bit-identically**. Using the real 40k full-model checkpoint
`/mnt/tsstd01safn/data/projects/phi-fastllm-2605/amlt-results/fast-llm-2605-qwen3-5-4b-500M-streaming-encoder-s1-data-v3.2-40k/40000/mp_rank_00_model_states.pt`,
the real encoder weights (`model.embed_tokens_extend.encoder.*`, 323 tensors, 0 missing / 0 unexpected)
plus the real MLP audio projection (`model.embed_tokens_extend.audio_projection.*` =
`Linear(1280→2560) → GELU → Linear(2560→2560)`) and the real CMVN were run on the real captured
log-mel batch in bfloat16. Output shape `(36, 168, 2560)`, maximum absolute difference `0.0`, mean
absolute difference `0.0`, with `0` of `15,482,880` elements exceeding `0.01`. Sample values match the
reference exactly (`3.67188 == 3.67188`). This meets the encoder acceptance gate
(`output_shape_exact` and `bf16_max_absolute_difference_max: 0.01`) with zero error.

The full audio projection module still needs to be migrated as a first-class P4 package (it was
validated here with an inline `nn.Sequential`); the LLM step-zero loss (P4) and data pipeline (P5)
remain. The `nemo_conv` `is_causal=True` subsampling path has a pre-existing baseline shape bug
(49 vs 51) unrelated to the port; the production run uses `input_layer=conv2d`, so it is off the
production critical path.

## P4 Evidence (audio embedding)

The audio projection was migrated as a first-class module
`src/megatron/bridge/models/speech_cascades/audio_embedding.py` (`CascadesAudioEmbedding`): the
vendored `ConformerEncoder` composed with the MLP/linear audio projection, faithful to the
production `get_audio_features` path (pre-encoder pad, optional linear downsample, projection).
Unsupported branches (QFormer, conv/temporal/bottleneck downsample, CTC) raise rather than silently
diverge. `load_full_model_weights` populates the encoder (with its nested CMVN buffers) and the
projection from the consolidated DeepSpeed `module` dict, asserting a one-to-one mapping.

The real 40k full-model checkpoint's encoder state (`model.embed_tokens_extend.encoder.*`, 323
tensors) already contains the CMVN under `encoder.encoder_embedding.*` (`global_mean` absmean
`14.18`), so loading the 40k alone (encoder + `audio_projection.*`) suffices — no separate CMVN
artifact is needed. The committed functional test
`tests/functional_tests/models/speech_cascades/test_audio_embedding_m3_parity.py` loads those real
weights, runs the real captured log-mel batch in bfloat16, and reproduces
`m3_capture/ref_audio.pt['audio_features']` `(36, 168, 2560)` with maximum absolute difference `0.0`
(gate `bf16_max_absolute_difference_max: 0.01`). The test skips cleanly when the artifacts are not
provided. Fast CPU unit tests
(`tests/unit_tests/models/speech_cascades/test_audio_embedding.py`) pin the projection wiring and a
deterministic output signature.

## P4 Evidence (native model + step-zero loss)

The native audio-language path works end-to-end the repo way. The one-time converter
`examples/conversion/convert_speech_cascades_ds_to_hf.py` turns the 40k DeepSpeed checkpoint into an
HF-format directory whose `architectures` dispatches to `SpeechCascadesBridge`. Then
`AutoBridge.from_hf_pretrained(...).to_megatron_provider(load_weights=True).provide_distributed_model(...)`
builds and loads the full 4.3B `SpeechCascadesModel` on a single GPU. Converted weights match the HF
source with maximum absolute difference `0.0` for the LM embedding, GDN `conv1d`, audio encoder
attention, audio projection, and the CMVN buffer.

Step-zero loss on the real captured batch `m1_capture/batch.pt` matches the baseline. The captured
labels are unshifted (`labels[i] == input_ids[i]`), so the causal next-token loss uses
`hidden[i-1] → labels[i]`. Over the 1,439 supervised positions the native model gives cross-entropy
`0.2068` versus the reference `0.20644` (relative difference `0.17%`, within the `1%` gate) with
teacher-forcing top-1 accuracy `0.9388`. Fixes required for a correct forward: GDN
`activation_func = F.silu`, `position_embedding_type = "rope"`, and `autocast_dtype = bfloat16`
(matching the shipped `Qwen35Bridge`); a buffer-aware fallback in
`ReplicatedMapping.hf_to_megatron` so the CMVN buffer-only module can be mapped. GPU memory is tight
(other services occupy most of the card), so the loss is measured by capturing the output-layer input
and running the output projection only on the supervised positions.

## P5 Evidence (ALM data pipeline)

The deterministic batch-assembly core of the ALM data pipeline was migrated to
`src/megatron/bridge/data/speech_chunk/`: `pad_features` and `ALMDataCollator` (bridge-native,
dependency-free) reproduce the exact batch schema consumed by `SpeechCascadesModel.forward`
(`input_ids`, `labels`, `inputs_embeds`, `audio_attention_mask`, `audio_embed_sizes`,
`audio_frames`, optional `attention_mask`), including the negative-id audio placeholder convention
(`-(k+1)` repeated `audio_embed_sizes[k]` times). `ALMLocalChunkDataset` is an `IterableDataset` that
reads local `.pt` example-record shards, splits them across data-parallel ranks, and yields collated
batches — the offline counterpart to Azure blob streaming (which shares the identical
example-record -> collator contract).

Parity was proven against the **real captured batch** `m1_capture/batch.pt`
(`input_ids/labels (1, 7778)`, `inputs_embeds (36, 1356, 80)`, 36 audio segments,
`audio_embed_sizes.sum() = 5680`). The batch is decomposed back into 36 per-audio un-padded features
and one ALM example, then re-collated; every produced key matches the captured tensors exactly
(`torch.equal`, gated functional test
`tests/functional_tests/data/speech_chunk/test_alm_collator_m1_parity.py`). Fast CPU unit tests
(`tests/unit_tests/data/speech_chunk/test_alm_collator.py`) pin the schema, audio-mask lengths,
cross-batch padding, and non-empty local-dataset batch production. `ruff check`/`ruff format` clean;
`7 passed`.

The full production streaming subsystem (`blob_chunk_alm`: manifest parsing, Azure/`blobfile`
download, filters, transforms, sample processors — ~18k lines) is deeply coupled to blob storage and
cannot be validated offline; it is intentionally out of scope. The migrated collator + local dataset
are the deterministic, real-data-verifiable heart of the pipeline and are drop-in compatible with a
future streaming reader that emits the same example records.

## P6 Evidence (training recipe + step)

The training path is wired the repo-native way. A custom forward step
`src/megatron/bridge/models/speech_cascades/step.py` (`forward_step` + `get_alm_batch`) consumes the
ALM collator batch and applies the **causal next-token shift** the data requires: the collator emits
unshifted labels (``labels[i]`` is the target *at* position ``i``, matching the captured batch), while
Megatron-Core computes ``CrossEntropy(logits[i], labels[i])`` with no internal shift, so the step sets
``shifted[i] = labels[i+1]``, masks the last position, and clamps ignored targets to ``0``.

The recipe `speech_cascades_finetune_config`
(`src/megatron/bridge/recipes/speech_cascades/speech_cascades.py`) returns a `ConfigContainer` built
from the converted HF directory (`AutoBridge.from_hf_pretrained(...).to_megatron_provider(load_weights=False)`)
wired to `ALMLocalDatasetProvider`. **Loading a previously trained model** is supported two ways: cold
start from the imported 40k weights via ``checkpoint.pretrained_checkpoint`` (a Megatron directory
produced by ``convert_checkpoints.py import``; weights loaded, optimizer/RNG/step reset), and resume
via ``checkpoint.load`` defaulting to ``checkpoint.save`` (weights + optimizer + RNG + step from the
latest ``iter_*``). The recipe is registered in `megatron.bridge.recipes` and the step in
`scripts/training/run_recipe.py` (`--step_func speech_cascades_step`).

The correctness-critical piece — the causal shift on the real 4.3B weights — was validated on the real
captured batch. `get_alm_batch` shifts the m1 labels, the model runs, and the masked cross-entropy over
the supervised positions is within `1%` of the reference `0.20644` (gated functional test
`tests/functional_tests/models/speech_cascades/test_step_m1_loss.py`, memory-safe supervised-position
projection). Fast CPU unit tests
(`tests/unit_tests/models/speech_cascades/test_step.py`) pin the shift, ignored-target masking, audio
passthrough, and the dataset-provider split wiring. The recipe constructs a valid resume-capable
`ConfigContainer` (`save == load`, `pretrained_checkpoint` passthrough, `seq_length` matched, batch
schema `input_ids / labels / inputs_embeds / audio_attention_mask / audio_embed_sizes / audio_frames`).
`ruff check` / `ruff format` clean.

Full-scale 8-step training + resume of the 4.3B model requires optimizer state that does not fit
alongside the co-located inference services on this single card; run it on a dedicated GPU/node with:

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
  --recipe speech_cascades_finetune_config --step_func speech_cascades_step \
  --hf_path <converted_hf_dir> \
  checkpoint.pretrained_checkpoint=<converted_megatron_dir> \
  dataset.shard_paths=[<train_shard.pt>] train.train_iters=8 checkpoint.save_interval=4
```

## P7 Evidence (HF export round trip + inference)

The bridge is bidirectional, so the native model exports back to HuggingFace with no export-only gaps.
`AutoBridge.export_hf_weights(model)` streams every weight through each mapping's ``megatron_to_hf``
(`ReplicatedMapping` covers the audio encoder / projection / CMVN buffers; the Qwen3.5 GDN LM reuses
the dense mappings). Validated on the real 4.3B weights: building the native `SpeechCascadesModel`
from the converted HF directory and re-exporting reproduces the source checkpoint — every one of the
>700 tensors matches within `1e-2`, with the audio encoder, audio projection, and CMVN buffers all
present in the export (gated functional test
`tests/functional_tests/models/speech_cascades/test_hf_roundtrip.py`). Example CLI:
`examples/models/speech_cascades/export_hf.py` (`AutoBridge.save_hf_pretrained`).

Inference is provided by `src/megatron/bridge/models/speech_cascades/generate.py::greedy_generate`, a
cache-free, memory-safe greedy decoder that captures the output-layer input via a forward pre-hook and
projects only the last position, so it never materializes a full-vocabulary logits tensor. Audio
features are supplied every step because the negative-id audio placeholders live in the prompt prefix.
On the real captured audio batch and real 4.3B weights, teacher-forced next-token top-1 accuracy over
the supervised positions is `>= 0.9` (measured `0.9388`), and `greedy_generate` decodes continuation
tokens without error (gated functional test
`tests/functional_tests/models/speech_cascades/test_inference_m1.py`; example CLI
`examples/models/speech_cascades/generate.py`). `ruff check` / `ruff format` clean.

## P8 Evidence (VC02 cluster submission)

The native path was submitted to the production Singularity cluster (VC02,
`spch-train-h200-safn`, amlt project `yuxuan-phi3mm`), modeled on the team's proven baseline config. A
CLI entry `examples/models/speech_cascades/finetune.py` builds the recipe and calls `finetune(...)`;
the recipe gained a `load_hf_weights` option so a single self-contained job can cold-start directly
from the converted HF weights across TP/PP (no separate Megatron conversion phase). The launch wrapper
`projects/cluster/run_speech_cascades_cluster.sh` bootstraps the `/opt/bridge-overlay` deps if absent,
converts the real 40k checkpoint to HF on the primary rank, then runs the native training and a resume
phase; the amlt config `projects/cluster/amlt_speech_cascades_vc02.yaml` requests 2 nodes of
`141G8-H200-IB-NvLink` under `mpi: True`.

Submitted as experiment `speech_cascades_p8` (job `speech-cascades-bridge-tp2-2node`,
`2x141G8-H200-IB-NvLink`). This job validates native parallel training (TP2) plus distributed
checkpoint save and resume at scale, cold-starting from the real 40k weights. The production
storage-API streaming data pipeline (`blob_chunk_alm`) is not yet ported, so the training loop uses a
small deterministic local shard; full production training on VC02 additionally requires porting the
streaming reader (which shares the ALM example-record -> collator contract).

The run **passed** end to end: the 40k checkpoint converts to HF on the cluster, the native model
builds and loads across TP2, phase 1 trains and saves `iter_0000008`, and phase 2 resumes from that
distributed checkpoint and trains through `iter_0000012`, saving again ("after training is done").
Getting there required an overlay/training image with `nvidia-resiliency-ext`
(`speechllmregistry.azurecr.io/jing-dev:mega_0812-bridge-v0.5.1-train`), a recipe-aggregator bypass in
the CLI entry (optional deps like `diffusers`/`megatron.energon` are absent from the minimal image),
bridge self-registration in the recipe, and a `torch.distributed.checkpoint.load` compatibility shim
(`src/megatron/bridge/models/speech_cascades/dcp_compat.py`) because PyTorch 2.6 dropped the `no_dist`
argument the pinned Megatron-Core loader passes.

## Target Extension Layout

```text
src/megatron/bridge/models/speech_cascades/
  bridge.py
  model.py
  provider.py
  step.py
  encoder/
src/megatron/bridge/data/speech_chunk/
src/megatron/bridge/recipes/speech/
examples/models/speech_cascades/
tests/unit_tests/models/speech_cascades/
tests/unit_tests/data/speech_chunk/
```

Do not create these implementation packages before P1 proves that the native Qwen3.5 bridge can
represent and convert the target language model. This keeps model-conversion risk separate from
speech integration risk.

## Validation Commands

Run all commands inside the pinned container:

```bash
uv run python examples/conversion/hf_megatron_roundtrip.py \
  --hf-model-id <local-or-remote-model>

uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
  --hf-model-id <local-or-remote-model> --tp 2

uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/compare_hf_and_megatron/compare.py \
  --hf_model_path <local-or-remote-model> --tp 2 \
  --prompt "The capital of France is"
```

## Stop Conditions

- If P1 exact round-trip fails, fix or extend `Qwen35Bridge` before moving speech code.
- If exact round-trip passes but forward parity fails, inspect provider configuration before weight
  mappings.
- If a gate fails, retain the current production launcher and checkpoint unchanged.
- Do not commit, push, or switch the production launcher without explicit approval.