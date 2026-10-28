# Verified speech training and evaluation changes

This guide covers the portable fixes collected in October 2026. Unit and CPU
distributed checks are not evidence of completed GPU training, benchmark scores,
convergence or performance gains.

## Streaming data

- `streaming_use_storage_api=None` preserves each native dataset entry's
  transport setting. The fine-tuning CLI exposes `--streaming-use-storage-api`
  and `--no-streaming-use-storage-api`; `--streaming-force-local` explicitly
  selects mounted files. Combining forced-local reads with an enabled storage
  API override is rejected.
- Audio, transcription and answer overlays retain independent storage roots.
  HTTPS and `az://` roots preserve their account/container rather than being
  interpreted as mounted filesystem paths. Container clients are reused.
- Successful slow SDK reads count as progress. Transport errors resume from the
  last appended byte; request timeouts bound stalled network reads.
- Set `BLOBCHUNK_SPILL_DIR` to a writable node-local directory to spill decoded
  shuffle-buffer samples. Sample order and resume position are preserved,
  consumed samples are released, and spill files are reclaimed with their final
  reference. Cache write failures propagate rather than silently dropping data.
  Unset the variable to retain in-memory behavior.
- Abnormal transcriptions are rejected with a warning before teacher/target
  encoding. They do not become silently substituted training targets.
- `SpeechChunkStreamingDatasetProvider.timeout` must be nonnegative; a positive
  value requires worker processes.

The storage override changes runtime copies of native entries, not published
data specifications or sampling weights.

## Tensor parallel speech and distillation

Only the TP source rank opens the streaming iterator. The forward step broadcasts
dynamic text, audio and teacher tensors to its TP peers.

Audio is merged before sequence-parallel scattering. Odd sequence lengths are
padded for TP alignment, excluded from router auxiliary loss through a padding
mask, and trimmed from batch-first losses and inference logits. Explicit attention
masks combined with automatic alignment padding are rejected.

The KD head uses the backbone's actual tied or untied output weights and logit
scaling. TP vocabulary projection and SP hidden-state gathering preserve
full-vocabulary loss/gradient semantics without a second SP token gather. Frozen
KD routers do not add a separate auxiliary training objective. Speech MTP with SP
is explicitly unsupported.

Audio attention maps remain enabled by default for compatibility. Set the model
provider's `retain_audio_attention_maps=False` to release the encoder's diagnostic
attention references after each forward; output and gradient behavior is tested.
No measured memory or throughput improvement is claimed.

Fine-tuning also exposes `--overlap-grad-reduce` / `--no-overlap-grad-reduce`.
`--exit-interval` requests a save/exit boundary without shortening `train_iters`
or its learning-rate schedule; nonpositive values are rejected.

## Export trained checkpoints

Use the trained Megatron iteration directory, not an initial HF round-trip:

```bash
uv run python examples/models/speech_cascades/export_hf.py \
  --hf-dir /path/to/initial-hf \
  --checkpoint /path/to/checkpoints/iter_0004000 \
  --vllm --output /path/to/fresh-export
```

The command requires iteration metadata and run configuration, uses strict
checkpoint export, and refuses nonempty output directories or overwriting the HF
source. `--vllm` requires a trained checkpoint. Runtime metadata preserves the
audio configuration and disables MTP metadata when no MTP tensors were exported.

The portable export wrappers under `projects/cluster/` take explicit source,
checkpoint and output inputs. Export receipts bind source metadata, profile,
iteration, tokenizer and shard hashes. Interrupted exports and existing results
are retained; reuse requires matching receipts.

## Evaluation

The repository-owned
[speech evaluation package](../../examples/models/speech_cascades/speech_eval/README.md)
provides OpenASR short/multilingual/long, BBH, VoiceBench, BFCL and Fleurs
entrypoints, HTTP audio transport and checkpoint/runtime inspection. Its migrated
reference sources retain their original licenses and provenance manifest.
No external source checkout is imported at runtime.

The seven-suite dispatcher requires eight MPI ranks, one service per assigned
GPU, explicit dataset mounts and fresh outputs. Completion receipts validate rank
coverage, source order, run identity, request audits and output hashes before
global scoring. WER/BLEU are recomputed from merged predictions, not rank averages.
Worker failures prevent a global completion summary. Short local scratch paths
avoid Unix socket path limits, and cleanup retains user output/logs.

VoiceBench generation does not invoke local scoring or paid judges. The result
collector verifies retained evidence and preserves smoke denominators, capped
responses, precision and the distinction between short and long datasets.
Small smoke runs must not be reported as complete benchmark results.

Serving/scoring libraries and datasets must already be available in the chosen
environment. Optional speculative runtime modes are retained for compatibility
but are not GPU-qualified by these checks. This change does not add container
dependency manifests, cluster/account bindings, job submissions or paid requests.

## Reproduce focused checks

The separate [Hindi evaluation guide](hindi-eval.md) covers the confirmed text,
recorded-audio, Realtime protocol and configurable-judge tools. Its offline checks
mock services and do not claim new live benchmark results.

Use an existing, dependency-complete speech training container and the pinned
Megatron-Core checkout. Do not synchronize or upgrade dependencies during these
checks:

```bash
UV_NO_SYNC=1 uv run python -m pytest \
  tests/unit_tests/data/speech_chunk_streaming \
  tests/unit_tests/models/speech_cascades \
  --ignore=tests/unit_tests/models/speech_cascades/test_vllm_eval.py \
  --ignore=tests/unit_tests/models/speech_cascades/test_eval_artifacts.py -q

UV_NO_SYNC=1 uv run python -m pytest \
  tests/unit_tests/models/speech_cascades/test_eval_artifacts.py \
  tests/unit_tests/models/speech_cascades/test_vllm_eval.py \
  --confcutdir=tests/unit_tests/models/speech_cascades -p no:cacheprovider -q

CUDA_VISIBLE_DEVICES="" UV_NO_SYNC=1 uv run python -m pytest \
  tests/unit_tests/models/qwen/test_qwen35_bridge.py \
  tests/unit_tests/models/qwen_vl/test_qwen35_vl_bridge.py \
  tests/unit_tests/models/qwen4_exp \
  tests/unit_tests/models/test_conversion_utils.py \
  tests/unit_tests/peft/test_canonical_lora.py \
  -k "not TestCanonicalLoRAMegatronIntegration" -p no:cacheprovider -q
```

The second command intentionally excludes unrelated global model fixtures for
standalone HTTP/scorer/CPU-Gloo checks. The eight-process cases need `mpirun`
and a CPU PyTorch Gloo runtime. Container-root MPI launches need the runtime's
explicit root authorization; do not enable that on a shared bare-metal host.

The two-GPU sequence-parallel checkpoint test and encoder parity test are separate
functional gates; CPU success must not be presented as their GPU qualification.

## CI on public forks

The October 2026 upstream forward port retains the official replacement for the
retired Claude auto-review workflow: review-command guidance and the `/review`
command. It does not restore the former private SSO pre-flight or inference
integration. The secrets detector and other validation workflows remain enabled.

The secrets baseline records reviewed false positives for source SHA-256 hashes,
pinned upstream revisions and a mock credential-bearing URL used to test endpoint
rejection. These entries are bound to their file and value; detector rules and
file exclusions are unchanged. Scanning the complete forward-ported history also
audited 12 older false positives: 11 output/checkpoint path literals and one
vendored image-substitution marker. Their exact file/value hashes are recorded
without changing the original files or adding detector exclusions.

## Upstream forward port: 2026-10-27

### Baselines and preservation

- Old official base: `2eae3c971dd6971beb403e6536987427b7e8a3b2`.
- Merged downstream source: `688d3ae6d4bf96d2ec3dcd0bef4398efd89d29a3`.
- New official base: `eb09bfc06933e59615ce46b32c30a67b5a60a71e`
  (2026-10-27, Muon distributed setup and checkpointing).
- Replayed source tip: `36f38e605862216b50f519b4bfd9a3ce8ab67f1a`;
  compatibility tests and this report follow in a separate commit.
- Rollback refs:
  `backup/main-pre-upstream-20261027-6ceaa450` and
  `backup/speech-merged-pre-upstream-20261027-688d3ae6`.

The official range contains 377 commits and 1123 changed files. The fork's older
main was 478 commits behind the selected official tip. All 38 linear downstream
commits were replayed onto that exact tip, without rewriting the existing speech
branch. Uncommitted experiments were excluded and left in their original worktree.

Range-diff preserves 35 patches unchanged and accounts for three adaptations:

1. `0ea2a0c8`: retain both new upstream BAGEL and downstream speech-step
   registrations; combine upstream pickle-preservation tests with downstream
   mixed-dtype conversion tests.
2. `a01c1f0b`: apply `router_aux_loss_coef` through the new shared Qwen3.5 MoE
   config-kwargs helper, so the builder and provider paths both retain the value,
   including explicit zero.
3. `688d3ae6`: retain upstream's replacement review workflow and current
   documentation fixes instead of restoring retired integrations or model pages.
   Keep the audited secrets-baseline entries and release-pinned historical links.

Of the original 335 downstream delta paths, 319 remain in the replayed delta.
Fourteen other paths are already covered by upstream's workflow/documentation
changes; two retired Gemma/Llama documentation files stay deleted. No downstream
source, test, recipe, Dockerfile or launcher was lost. The follow-up also adds
regression coverage to the current upstream Qwen3.5 test file.

### Upstream impact

The new baseline moves Qwen3.5 and Qwen3-Next text models to builder-backed configs,
changes shared training/checkpoint/conversion code, and advances MCore to
`3ddfe0fe07efc73dbdc40d4a02acbc1c1f8a9f08`. CPU checkpoint test doubles now supply
the model config required by topology validation; router test groups supply the
new expert-TP group. Neither change bypasses production validation.

The manifest, lockfile and MCore gitlink are byte-identical to the selected
official tree. No dependency was added or downgraded. The official CI Docker base
is now 26.09. Effective locked versions differ from the existing diagnostic image:

| Component | Official lock | Diagnostic image |
| --- | --- | --- |
| Transformers | 5.20.2 | 5.15.0 |
| Transformer Engine | 2.20.2+44f31bbe | 2.18.0+27486e03 |
| nvidia-resiliency-ext | 0.7.0 | 0.6.0 |
| MLflow | 3.17.2 | 3.16.0 |
| Hydra | 1.3.8 | 1.3.2 |

### Validation and remaining gates

- `uv lock --check`: passed, 349 packages; no full dependency sync was performed.
- All 252 replayed Python paths parse. Changed compatibility Python files pass
  Ruff correctness/import checks and formatting checks.
- Candidate source and the new MCore checkout import with the diagnostic image,
  including speech, streaming-data and fine-tuning modules.
- Focused speech/data/conversion/Qwen/PEFT checks: 392 passed, 22 skipped.
  Two CUDA-only canonical-LoRA integration cases were explicitly deselected
  after confirming they cannot run in the CPU-only diagnostic configuration.
- Standalone evaluation/receipt/HTTP and MPI/Gloo checks: 98 passed.
  These include eight-worker success and failure-propagation cases.
- All 57 previously audited scanner entries are retained. The complete migration
  delta passes the CI scanner after auditing the 12 legacy false positives.
  A new unregistered credential fixture is still rejected. Upstream detector
  plugins, thresholds and filters are unchanged.
- CPU native checkpoint restore passes with real distributed-checkpoint
  save/load. This is not GPU optimizer/RNG/sampler resume qualification.
- Required all-files pre-commit was run in a disposable checkout. Existing EOF,
  trailing-whitespace, formatting and two Markdown-filename violations remain;
  unrelated automatic rewrites were not imported. The replay's whitespace
  warnings match the old downstream baseline.
- Strict Mypy did not pass in the diagnostic image: the normal run hit a checker
  internal error, while an import-isolated diagnostic reported missing typing
  metadata and untyped tests/decorators. Type checking is not a passed gate.
- Official and active speech images were not rebuilt or published; no candidate
  shipping image digest exists. Installation against the complete new lock,
  image/ABI checks and the approved image pipeline remain required.
- Real-data two-step training, production-layout distributed save/resume, full
  training and matched OpenASR quality evaluation were not run.

This is a source-migration candidate, not a production-ready runtime or a new
training/quality baseline. Existing launcher image references are retained for
review and must not be interpreted as qualification for the new dependency stack.
