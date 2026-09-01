# S2T + KL Distillation — Feasibility & Execution Plan

> Scope: implement the phyagi stage-2 speech->text (S2T) knowledge-distillation training and the
> S2T eval suite in Megatron-Bridge, on top of the already-shipped stage-1 EN-ASR encoder+projector
> path. Structured as independent **work packages (WP)** so multiple subagents can run in parallel,
> each with concrete **acceptance criteria**.
>
> Reference (READ-ONLY): `/home/yuxuanhu/remote_code/phi-3-moe/omni_worktree/s2s_qwen_dev/` (`phyagi`).
> Branch: `speech-cascades-on-main`. Image: `megatron-bridge-speech:torch210`
> (registry `speechllmregistry.azurecr.io/jing-dev:mega_torch210-bridge-speech-train2`).

## 1. Architecture (ground truth from the reference)

**KLD framework** (`phyagi/models/qwen3_5/modeling_qwen3_5.py` forward, lines ~1610-1663):
- The **student** is the speech-conditioned pathway (audio encoder -> projector -> LLM); the
  **teacher** is a text-only forward of the **same frozen LLM** with each audio span replaced by its
  aligned transcript tokens. Both views keep identical supervised **response** tokens.
- Per forward: `response_hidden = hidden[..., :-1, :][label_mask]` (student, shifted, response tokens
  only). Teacher runs under `torch.no_grad()` -> `teacher_response_hidden = teacher_last_hidden[..., :-1, :][teacher_label_mask]`.
  A hard assert requires `response_labels == teacher_response_labels` (token alignment).
- `kl = chunked_kl_divergence(student_resp_hidden, teacher_resp_hidden, lm_head, T=2.0, chunk=128)`
  then `loss = w*kl + (1-w)*ce` with `distillation_weight w = 0.75` (`w == 1.0` -> pure KL).
- KL loss + mixing already ported: `src/megatron/bridge/models/speech_cascades/distillation.py`
  (`chunked_kl_divergence`, `distillation_loss`), unit-tested.

**Training stages**:
- **S1 (DONE)** — encoder+projector align, frozen LLM, CE. (`en-asr-s1-torch210-full-v2`, prod-matched.)
- **S2 KD (this plan)** — freeze encoder (from S1), train the projector (Proj20M or MLP), loss = KL+CE.
  Config keys: `knowledge_distillation: true`, `distillation_weight: 0.75`. Data mixture spans
  transcription / direct-response / speech-continuation formats.
- **MTP (REQUIRED)** — the target S2 config is `proj20m-p1-mtp4-kd`: the model carries
  `mtp_num_hidden_layers` (=4) multi-token-prediction depth heads (extra decoder layers, init-from-first).
  The loss adds, over depths k=1..D, `depth_loss = (1-mtp_w)*depth_ce + mtp_w*depth_kl*T^2` weighted by
  `mtp_loss_depth_weights` ([0.50,0.25,0.15,0.10]) and scaled by `mtp_loss_weight` (=1.0), with
  `mtp_distillation_weight (mtp_w) = 0.75`. Training is 2-phase: warm (~500 steps) then full (~4000
  steps), MTP heads initialized from the first layer (`mtp_init_extra_layers_from_first`).

**Evals**: OpenASR WER (DONE). Remaining: BigBench-Audio (BBH, GPT-4.1 judge), VoiceBench (9-task
macro), BFCL (function-call micro), Fleurs (translation BLEU).

## 2. Reference source map (read these to implement)

| Topic | Reference file |
| --- | --- |
| KL + CE forward, teacher path | `phyagi/models/qwen3_5/modeling_qwen3_5.py` (`~1556-1663` main; `~2814-2960` MTP) |
| KL / CE loss fns | same file `1228 fused_cross_entropy_loss`, `1245 chunked_kl_divergence` |
| MTP heads + per-depth loss | same file `1681 Qwen3_5MultiTokenPredictor`, `1814 Qwen3_5MTP`, forward `~2930-3025`; config keys `mtp_*` at `configuration_qwen3_5.py:107-123` |
| MTP4 KD config (warm->full) | `.../cascade_encoder_stage2/qwen3-5-4b-s2-proj20m-p1-mtp4-kd-{warm500,full4000}.yaml` |
| S2 KD config | `scripts/train/configs/phi3_audio/cascade_encoder_stage2/qwen3-5-4b-s2-projmlp-phi4mm-full-kl075-ce025.yaml` (and `*-proj20m-*-kd-*.yaml`) |
| Teacher-text data build | `phyagi/datasets/**` (blob_chunk_alm ALM collator; transcript span alignment) |
| OpenASR eval (done) | `scripts/audio_eval/run_openasr_text_qwen35.py` |
| BBH judge | `scripts/audio_eval/bbh_gpt_judge.py`, `format_bigbench_audio_judge_input.py` |
| VoiceBench | `scripts/audio_eval/run_voicebench_cascade_text_qwen35.py`, `prepare_voicebench_latest.py` |
| BFCL | `scripts/audio_eval/run_bfcl_v3.sh` |
| Fleurs (MT) | `scripts/audio_eval/run_fleurs_cascade_qwen35.py` |
| Results targets | `projects/jobs/S2T_COMPARE_RESULTS.md` |

## 3. Shared conventions (every WP)

- **Code**: `src/megatron/bridge/models/speech_cascades/**`, recipes under `recipes/speech_cascades/`,
  cluster yamls under `projects/cluster/`, evals under `examples/models/speech_cascades/`.
- **Tests**: unit -> `tests/unit_tests/models/speech_cascades/`; gated functional (real weights,
  env-guarded, skip w/o artifacts) -> `tests/functional_tests/models/speech_cascades/`.
- **Lint/test in image**: `docker run --rm -v "$PWD:/repo:ro" megatron-bridge-speech:torch210 bash -lc
  'PYTHONPATH=/repo/src python -m pytest <test> -q -p no:cacheprovider; ruff check <file> --no-cache'`.
- **Local real-data smoke**: SAFN mount `/mnt/highperf01safn/data` -> `/datablob1` (eus2 mount is
  EIO-broken). stage1_hf at `/nfs/yuxuanhu/tmp/stage1_hf`, ckpt at `/nfs/yuxuanhu/tmp/v2_local`.
- **Cluster**: target `aiservices-vc02-eu2`, UAI `aiservices-training-c02-uai-eus2`, image tag
  `jing-dev:mega_torch210-bridge-speech-train2`. Loss on rank N-1 stdout; mlflow via obs_logging.
- **Do NOT** modify `3rdparty/Megatron-LM`; sign off commits (`-s`); no copilot co-author; NVIDIA
  header on new `src/` files; run under the image (not bare pytest/torchrun).

## 4. Work packages

Status legend: TODO / WIP / DONE / BLOCKED. "Parallel group" = WPs with no unmet deps can run together.

### WP0 — KL distillation loss (DONE)
- **Deps**: none. **Files**: `distillation.py`, `test_distillation.py`.
- **Acceptance (met)**: 6 unit tests pass (identical->0, chunk-invariance, student-only grad, empty/shape raise, mixing); ruff clean.

### WP-A1 — Teacher text-forward data path
- **Deps**: none (can start now). **Owner**: subagent A.
- **Scope**: extend the ALM collator / `step.py::get_alm_batch` so each training example yields, next to
  the speech batch, a **teacher** text batch: `teacher_input_ids` (audio placeholder span replaced by
  the aligned transcript tokens, rest identical), `teacher_attention_mask`, `teacher_labels` (== the
  student response labels). Must guarantee token-aligned response positions.
- **Files**: `src/megatron/bridge/data/speech_chunk/collator.py` (+ streaming collator path),
  `src/megatron/bridge/models/speech_cascades/step.py`.
- **Impl notes**: mirror phyagi — student input has audio placeholders (negative ids in this repo);
  teacher input replaces that span with the transcript token ids from the same example; response
  (answer) tokens are identical, so `label_mask` picks the same count on both sides.
- **Acceptance**:
  1. Unit test: given a synthetic example (audio span of size K + known transcript of T tokens + answer),
     the collated `teacher_input_ids` has the K placeholders replaced by the T transcript tokens and the
     answer segment byte-identical; `teacher_labels` response positions equal the student response labels
     (`torch.equal`).
  2. `get_alm_batch` returns the teacher tensors on cuda with correct dtypes; existing ASR (non-KD) path
     unchanged when KD disabled.

### WP-A2 — Dual-forward KD loss in the training step
- **Deps**: WP0 + WP-A1. **Owner**: subagent A (after A1).
- **Scope**: in `step.py`, run student forward (speech) -> response hidden; run teacher forward (frozen
  LLM, `torch.no_grad()`, text `decoder_input`) -> response hidden; assert response-label alignment;
  main `loss = distillation_loss(chunked_kl_divergence(...), ce, w)`. Reuse the memory-safe output-head
  projection (`shared_embedding_or_output_weight`) as `logits_fn`. The **total** training loss also adds
  the MTP term from WP-C2 (`loss += mtp_loss_weight * mtp_loss`).
- **Files**: `step.py`, small helper in `model.py` to expose a text-only forward returning hidden states.
- **Impl notes**: teacher forward must NOT build gradients; both forwards share the frozen LM weights;
  student gradients flow to the projector/encoder only. Keep the CE identical to the current ASR CE
  (causal shift already handled).
- **Acceptance**:
  1. Unit test (CPU, tiny dims, mocked LM forward): dual-forward path returns a finite scalar; grad flows
     to student trainables and not to the teacher/frozen weights.
  2. Gated functional (real 4.3B, env-guarded): KD step-zero loss is finite and `~= CE` when
     `w -> 0`; `~= pure-KL` when `w -> 1`; teacher path detached (no grad on LM).

### WP-A3 — S2 KD+MTP recipe + CLI + cluster yaml + local smoke
- **Deps**: WP-A2 + WP-C2 + WP-B1. **Owner**: subagent A (after A2/C2/B1).
- **Scope**: new recipe `speech_cascades_s2_kd_config` (freeze encoder+LLM, train projector + MTP heads;
  loss = KD + MTP) with `distillation_weight`, `distillation_temperature`, `distillation_kl_chunk_size`,
  `mtp_num_hidden_layers=4`, `mtp_loss_weight=1.0`, `mtp_loss_depth_weights=[0.50,0.25,0.15,0.10]`,
  `mtp_distillation_weight=0.75`; **2-phase** warm (~500) -> full (~4000, init from the warm ckpt);
  `finetune.py` flags; cluster wrapper + amlt yaml pointing at the S2 data (WP-B1) and the S1 checkpoint.
- **Files**: `recipes/speech_cascades/speech_cascades.py`, `examples/.../finetune.py`,
  `projects/cluster/{run_speech_cascades_s2_kd.sh, amlt_speech_cascades_s2_kd_vc02.yaml}`.
- **Acceptance**:
  1. Recipe builds a valid `ConfigContainer` (freeze flags correct: encoder+LLM frozen, projector
     trainable; KD keys set); ruff clean; recipe unit test.
  2. Local 2-step real-data smoke (SAFN mount) green: KD loss finite/decreasing, 0 nan, dist-ckpt saves,
     EXIT=0.
  3. (Optional) short cluster run (e.g. 400 steps) shows KD loss decreasing on rank N-1.

### WP-B1 — S2 KD data spec
- **Deps**: none (parallel with A). **Owner**: subagent B.
- **Scope**: `projects/cluster/data_en_s2_kd.yaml` mirroring the reference S2 mixture (in-house+earnings
  ASR, multilingual ASR, open-source ASR: MLS/GigaSpeech/CommonVoice/VoxPopuli/Fleurs) with the
  transcript-alignment fields the teacher path (WP-A1) consumes; per-group filters per the ref config.
- **Files**: `projects/cluster/data_en_s2_kd.yaml`.
- **Acceptance**:
  1. yaml parses; streaming pipeline `build_speech_chunk_dataset` builds it on the SAFN mount; one batch
     has the expected student + teacher keys.
  2. Group weights/filters match the reference S2 config (documented diff table in the PR).

### WP-C1 — MTP depth heads on the model (REQUIRED)
- **Deps**: none (parallel with A1/B1). **Owner**: subagent C.
- **Scope**: add `mtp_num_hidden_layers` (=4) multi-token-prediction depth heads to the
  SpeechCascadesModel/provider — extra decoder layers predicting tokens at depths k=1..D, each
  producing a per-depth hidden state; init from the first layer (`mtp_init_extra_layers_from_first`).
  Config keys on the provider: `mtp_num_hidden_layers`, `mtp_loss_weight`, `mtp_loss_depth_weights`,
  `mtp_distillation_weight`, `mtp_init_extra_layers_from_first` (all default off/0 so the ASR path is
  unaffected when MTP is disabled).
- **Files**: `provider.py`, `model.py`, unit tests. Mirror `Qwen3_5MultiTokenPredictor` (ref 1681) +
  the depth-block construction in `Qwen3_5MTP.__init__` (ref 1843-1885).
- **Acceptance**:
  1. Model builds with 4 MTP depth heads; total param count = base + D extra depth blocks (documented).
  2. Model forward returns the D per-depth hidden states with correct shapes; ASR path byte-identical
     when `mtp_num_hidden_layers == 0`.
  3. Weights round-trip: MTP heads init-from-first matches the reference init; unit test on shapes.

### WP-C2 — MTP per-depth KL+CE loss (REQUIRED)
- **Deps**: WP0 + WP-C1 + WP-A1 (teacher). **Owner**: subagent C (after C1).
- **Scope**: in `step.py`, aggregate the MTP loss over depths: `depth_loss = (1-mtp_w)*depth_ce +
  mtp_w*depth_kl*T^2` (note the `T^2` on the MTP KL, unlike the main-path KL), weighted by
  `mtp_loss_depth_weights` and scaled by `mtp_loss_weight`; add to the main KD loss (WP-A2).
- **Files**: `step.py`, `distillation.py` (add an `mtp_depth_loss` helper), tests.
- **Acceptance**: unit test — per-depth loss matches the reference formula (mocked hidden/logits);
  depth weighting + `T^2` correct; gated functional step-zero total loss (main + MTP) finite; grad
  flows to the trainable projector + MTP heads only.

### WP-D1 — BBH (BigBench-Audio) eval + GPT-4.1 judge
- **Deps**: none (parallel). **Owner**: subagent D1.
- **Scope**: `examples/models/speech_cascades/eval_bbh.py` (audio -> model answer) + judge harness
  mirroring `bbh_gpt_judge.py` (GPT-4.1 CORRECT/INCORRECT accuracy). Reuse the load-once + greedy infra
  from `eval_openasr.py`; `TORCHDYNAMO_DISABLE=1`, 30-min pg timeout.
- **Files**: `examples/.../eval_bbh.py`, `projects/cluster/{run_,amlt_}...bbh...`.
- **Acceptance**: local dry-run on a few BBH samples produces judge inputs; judge computes accuracy on a
  tiny mocked set; cluster runner parity vs the reference format.

### WP-D2 — VoiceBench eval (9-task macro)
- **Deps**: none (parallel). **Owner**: subagent D2.
- **Scope**: port `run_voicebench_cascade_text_qwen35.py` + `prepare_voicebench_latest.py`; macro over
  the 9 tasks. **Acceptance**: runner produces per-task outputs; macro scorer matches the reference
  aggregation on a small sample.

### WP-D3 — BFCL eval (function-call micro)
- **Deps**: none (parallel). **Owner**: subagent D3.
- **Scope**: port `run_bfcl_v3.sh` flow. **Acceptance**: micro accuracy computed on a small sample.

### WP-D4 — Fleurs translation (BLEU)
- **Deps**: none (parallel). **Owner**: subagent D4.
- **Scope**: port `run_fleurs_cascade_qwen35.py`. **Acceptance**: BLEU computed on a small sample.

## 5. Parallelization plan (dependency graph)

```
WP0 (done)
  ├── WP-A1 (teacher data) ─┐
  ├── WP-C1 (MTP heads) ─────┼─► WP-A2 (main KD loss) + WP-C2 (MTP loss) ─► WP-A3 (S2 KD: proj20m+mtp4, warm->full)
  │                          │                                                ▲
  WP-B1 (S2 data) ───────────┴────────────────────────────────────────────────┘
  └── WP-D1 | WP-D2 | WP-D3 | WP-D4 (all independent, fully parallel)
```

- **Wave 1 (now, parallel — 7 subagents)**: WP-A1, WP-C1, WP-B1, WP-D1, WP-D2, WP-D3, WP-D4.
- **Wave 2**: WP-A2 (needs A1) + WP-C2 (needs C1 + A1), once teacher path lands.
- **Wave 3**: WP-A3 (needs A2 + C2 + B1) -> the end-to-end KD+MTP training smoke.
- MTP (C1/C2) is on the **critical path** for A3 (the target config is `proj20m-p1-mtp4-kd`).

## 6. Milestones / acceptance gates

- **M1 — KD+MTP trains**: WP-A1+C1+A2+C2+A3+B1 -> local 2-step KD+MTP smoke green (main + per-depth
  MTP loss finite/decreasing, 0 nan) + short cluster warm run with loss decreasing.
- **M2 — KD quality**: full 2-phase (warm->full) S2 KD run + OpenASR (done infra) + WP-D1/D2 evals;
  compare to `S2T_COMPARE_RESULTS.md` targets.
- **M3 — Full eval suite**: WP-D1..D4 all green on cluster.

## 7. Status

| WP | Title | Status |
| --- | --- | --- |
| WP0 | KL distillation loss | DONE |
| WP-A1 | Teacher text-forward data path | DONE (baee76825) |
| WP-A2 | Dual-forward KD loss in step | TODO |
| WP-A3 | S2 KD recipe + smoke | TODO |
| WP-B1 | S2 KD data spec | DONE (2caccc3c8) |
| WP-C1 | MTP depth heads on the model (REQUIRED) | DONE (eefa9f759) |
| WP-C2 | MTP per-depth KL+CE loss (REQUIRED) | TODO |
| WP-D1 | BBH judge eval | TODO |
| WP-D2 | VoiceBench eval | TODO |
| WP-D3 | BFCL eval | TODO |
| WP-D4 | Fleurs MT eval | TODO |
