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
```

The second command intentionally excludes unrelated global model fixtures for
standalone HTTP/scorer/CPU-Gloo checks. The eight-process cases need `mpirun`
and a CPU PyTorch Gloo runtime. Container-root MPI launches need the runtime's
explicit root authorization; do not enable that on a shared bare-metal host.

The two-GPU sequence-parallel checkpoint test and encoder parity test are separate
functional gates; CPU success must not be presented as their GPU qualification.
