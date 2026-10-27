# Repository-Owned Speech Evaluation

This package contains the source needed by the seven-suite vLLM entrypoint.
It does not import `phyagi`, an external checkout's `projects.inference_vllm`,
or the image's `/opt/speech-src`. The image supplies installed runtime packages
only (vLLM, PyTorch, datasets, audio libraries and metric libraries).

For the reusable export -> serve/evaluate -> collect workflow and its validation
boundaries, see the [speech changes guide](../../../../docs/speech_cascades/verified-changes.md).
`export_receipt.py` validates safe export reuse; `collect_results.py` validates
retained MPI8 artifacts and aggregates existing metrics without invoking judges.
Cluster/account bindings and experiment templates are intentionally not included.
The portable suite dispatcher starts serving automatically after explicit model,
dataset and output inputs are supplied. Standard decoding is the supported
evaluation path; speculative decoding needs independent GPU qualification.

## Job Layout

Each cluster job runs one suite with `mpi: true` and
`process_count_per_node: 8`, requesting one 8-H100 node. Each scheduler rank
starts one TP1 vLLM server on its assigned GPU and processes disjoint strided
whole-recording shards. Set batching, per-rank concurrency and GPU memory utilization explicitly for the
selected model and hardware. Evaluation uses standard decoding, not ExpandedMTP4.

The repository-owned `mpi.py` checks all eight rank/device assignments through
Gloo before workers start. Shards and request audits live under
`<suite>/ranks/<rank>/results/`; completion receipts bind source order, run
identity, output hashes and request counts. Rank zero validates all shards,
restores source order and computes global WER/BLEU from merged predictions,
never averaging rank scores. BFCL reporting and BBH formatting run after merge.
VoiceBench only merges predictions/audits; its scoring stays local. Worker errors
propagate to peers, and no global completion summary is written on failure.
Use a fresh output directory; stale or partial directories are not resumed.

## Migration Scope

- Reference CLI, prompts, HF/AA/JSONL/TSV readers and long-form segmentation.
- HTTP audio client, native batch handling, segment merging, WER and BLEU.
- Original runner argument positions and selected ASR subsets (7/14/4),
  56 Fleurs directions, BBH/VoiceBench conversion, and five-category BFCL AST checks.
- vLLM audio adapter, checkpoint inspector, launch overlay and stable MTP proposer.
- VoiceBench evaluator sources pinned at `3c3b0d3a7a956f745305eb348f5e03ce7ec73dad`;
  no runtime clone, fetch or checkout. Evaluators are imported per task.

`migration_sources.json` records original relative paths and SHA256 values for
copied files and extracted functions. Imports and paths were localized; unused
model loaders and unrelated suite branches were excluded. `speech_api.py`
preserves the reference HTTP request fields; `speech_task.py` implements the
selected reference batching/merging/scoring path without local model generation.
The existing repository `eval_metrics` normalizers are reused, not duplicated.

Reference speech code is covered by `LICENSE.MIT`; the pinned VoiceBench code
is covered by `voicebench/LICENSE`. Existing source notices are retained.

## Assets And Optional Scoring

Dataset roots are explicit `OPENASR_ESB_ROOT`, `OPENASR_ML_ROOT`,
`OPENASR_LONG_ROOT`, `OPENASR_AA_ROOT`, and `FLEURS_ROOT` parameters.
Long-form SVAD needs the separately mounted reference ONNX asset through
`SPEECH_SVAD_MODEL`. Assets are not downloaded or replaced with other algorithms.

Cluster VoiceBench jobs only generate predictions and request audits. The cluster
entrypoint forces `RUN_EVAL=0`, `RUN_JUDGE=0` and `POSTPROCESS=0`; conversion and
scoring take place locally after generation. Scoring dependencies, language
resources and judge credentials therefore do not block cluster inference.
The unchanged image's missing `loguru` and `qa_metrics` concern optional local
scoring only; no packages were installed and no replacement metrics were used.
Local ASR and BFCL inference do not establish full VoiceBench scores. This PR
does not add dependency manifests or build a scoring image.

## Verification

The [evaluation regressions](../../../../tests/unit_tests/models/speech_cascades/test_vllm_eval.py)
check blocked external imports, HTTP fields/audits, long-form merging, runner
subsets, eight-process MPI/Gloo coordination on CPU, whole-recording partitioning,
empty rank shards, global scoring, tamper rejection, failure propagation, BFCL
coverage, read-only overlays and cleanup.
The [artifact regressions](../../../../tests/unit_tests/models/speech_cascades/test_eval_artifacts.py)
check checkpoint identity, export reuse, retained evidence and scoring denominators.
These checks do not certify GPU serving, speculative decoding, complete benchmark
scores or throughput. See the speech changes guide for reproducible commands.
