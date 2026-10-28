---
name: upstream-forward-port
description: "Forward-port a Megatron Bridge fork or long-lived feature branch onto the latest official NVIDIA upstream commit while preserving every local commit and working-tree change. Use for: sync official main, update to latest upstream, rebase or merge upstream, forward version, replay speech-cascades changes, 更新官方版本, 前进版本, 合并最新官方代码, and validating Docker, uv.lock, MCore, Transformer Engine, Transformers, checkpoint, training, and evaluation compatibility after the move."
argument-hint: "Optional upstream ref and branch, for example: upstream/main speech-cascades-on-main"
---

# Upstream Forward Port

Move a downstream Megatron Bridge branch to an exact, freshly fetched official
NVIDIA commit, replay the complete downstream change set, and prove that the
result still builds, trains, resumes, and evaluates correctly.

This is a forward-port workflow, not a blind merge. The invariant is:

```text
new branch = exact official upstream tree + every intentional downstream change
```

Never declare success from a clean rebase alone. Upstream changes to MCore,
Transformer Engine, Transformers, Docker bases, checkpointing, conversion,
training, and data loading can leave the Git operation green while making the
runtime unusable.

## Required Context

Read these before acting:

- `AGENTS.md`, especially the VC02 submission gate and repository boundaries.
- `skills/build-and-dependency/SKILL.md` for container and `uv` rules.
- `skills/testing/SKILL.md` for focused test selection.
- `skills/cicd/SKILL.md` before opening a PR or triggering CI.
- The AMLT CLI skill before any AMLT command or job submission.

Also read the current branch, official commit, dependency files, active speech
Dockerfiles, active job configs, and the previous migration report. Do not form
a migration plan from commit subjects alone.

## Non-Negotiable Safety Rules

1. Do not merge, rebase, checkout, reset, clean, or force-push while intentional
   working-tree changes are unrecorded.
2. Never lose untracked files. Inventory them and either commit the intentional
   files with user approval or explicitly exclude generated artifacts.
3. Create and verify a backup ref before rewriting history.
4. Never modify `3rdparty/Megatron-LM/`; update only its gitlink to the official
   pin unless a separately reviewed downstream pin is intentional.
5. Do not hand-resolve `uv.lock`. Resolve `pyproject.toml` intent, then regenerate
   the lock inside the supported Linux container.
6. Do not run the full test suite. Select focused unit, functional, build, and
   real-workload checks based on the touched shared paths.
7. Do not force-push over the last known-good remote branch before validation.
   Push a new candidate branch, or use `--force-with-lease` only after explicit
   approval and after preserving a remote backup.
8. Any AMLT submission must first pass `amlt run --dump`. If the resolved target
   is VC02, every job must have `priority: low`; abort otherwise. Verify accepted
   priority after submission.

## Phase 1: Freeze and Inventory the Old State

Fetch only after confirming the official remote URL:

```bash
git remote get-url upstream
git ls-remote upstream HEAD
git fetch upstream --prune
```

The official remote must be:

```text
https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
```

Record immutable inputs:

```bash
BRANCH=$(git branch --show-current)
OLD_HEAD=$(git rev-parse HEAD)
UPSTREAM_REF=${UPSTREAM_REF:-upstream/main}
UPSTREAM_SHA=$(git rev-parse "$UPSTREAM_REF")
OLD_BASE=$(git merge-base "$OLD_HEAD" "$UPSTREAM_SHA")

git status --short
git submodule status 3rdparty/Megatron-LM
git log --reverse --format='%H %s' "$OLD_BASE".."$OLD_HEAD"
git diff --name-status "$OLD_BASE".."$OLD_HEAD"
git diff --stat "$OLD_BASE".."$OLD_HEAD"
git rev-list --count "$OLD_BASE".."$OLD_HEAD"
git rev-list --count "$OLD_BASE".."$UPSTREAM_SHA"
git rev-list --merges "$OLD_BASE".."$OLD_HEAD"
```

Save the output in the migration report. The report must distinguish:

- committed downstream changes;
- tracked but uncommitted changes;
- untracked intentional files;
- generated or local-only artifacts;
- upstream-only commits and changed shared paths.

If the worktree is dirty, stop the rewrite. Review every path with the user and
create signed, signed-off snapshot commits for intentional work. Do not hide the
only copy in a stash. Continue only when `git status --short` is empty.

Create the backup after the final snapshot commit:

```bash
OLD_HEAD=$(git rev-parse HEAD)
BACKUP="backup/${BRANCH}-pre-upstream-$(date -u +%Y%m%d)-${OLD_HEAD:0:9}"
git branch "$BACKUP" "$OLD_HEAD"
test "$(git rev-parse "$BACKUP")" = "$OLD_HEAD"
```

Push the backup if the rewrite will later update a shared remote branch.

## Phase 2: Understand the Official Delta

Review the exact official range before replaying downstream changes:

```bash
git log --format='%h|%cI|%s' "$OLD_BASE".."$UPSTREAM_SHA"
git diff --shortstat "$OLD_BASE".."$UPSTREAM_SHA"
git diff --name-status "$OLD_BASE".."$UPSTREAM_SHA"
git diff "$OLD_BASE".."$UPSTREAM_SHA" -- \
  pyproject.toml uv.lock 3rdparty/Megatron-LM docker/ \
  src/megatron/bridge/models/conversion/ \
  src/megatron/bridge/training/ \
  src/megatron/bridge/models/qwen/ \
  src/megatron/bridge/models/qwen_vl/
```

Classify impact by behavior, not directory count:

| Area | Required question |
|---|---|
| MCore/TE/PyTorch | Does the current base image satisfy newly imported symbols, CUDA kernels, and ABI requirements? |
| Transformers/vLLM | Did supported ranges, architecture registration, lazy imports, or generation APIs change? |
| Conversion/checkpointing | Did load, tracker, RNG, pipeline layout, BF16 export, or mapping behavior change? |
| Training/PEFT | Did setup, recompute, signal handling, evaluation rank behavior, or adapter state change? |
| Data | Did sampler resume, worker handling, canonical-grid, packing, or output-directory behavior change? |
| Recipes/providers | Did field names, defaults, pipeline layouts, or model registration move? |
| Docker/CI | Did the base image, extras, patches, or build stages change? |

Search explicitly for deleted or moved APIs imported by downstream code. Use
language-server references where available, then verify imports in the rebuilt
container.

## Phase 3: Replay the Complete Downstream Series

The default for a linear downstream series is:

```bash
git rebase --onto "$UPSTREAM_SHA" "$OLD_BASE" "$BRANCH"
```

This replays exactly `OLD_BASE..OLD_HEAD` onto the exact official commit. Do not
replace it with a hand-picked subset. If the downstream range contains merge
commits, inspect them first and use `--rebase-merges` only when preserving their
topology is meaningful.

Use a normal merge only when published merge topology is an explicit project
requirement. An off-main release branch can make a merge replay an entire release
line and create broad, misleading conflicts.

For each conflict:

1. Read the upstream implementation and the downstream intent.
2. Keep the new upstream abstraction, file location, and public API.
3. Re-express the downstream behavior against that abstraction.
4. Preserve both registrations or tests when both remain valid.
5. For moved files, port the behavior to the new owner instead of reviving the
   deleted file.
6. For `pyproject.toml`/`uv.lock`, resolve only `pyproject.toml` now and relock in
   Phase 5.
7. For the MCore gitlink, start from the official pin. Never edit MCore sources.

After every conflict batch, run the cheapest focused check that can disprove the
resolution. Abort safely with `git rebase --abort` if the chosen base or replay
range is wrong.

## Phase 4: Prove Every Local Change Survived

Immediately after replay:

```bash
NEW_HEAD=$(git rev-parse HEAD)
git range-diff "$OLD_BASE"..."$OLD_HEAD" "$UPSTREAM_SHA"..."$NEW_HEAD"
git log --reverse --format='%H %s' "$UPSTREAM_SHA".."$NEW_HEAD"
git diff --name-status "$UPSTREAM_SHA".."$NEW_HEAD"
git diff --stat "$UPSTREAM_SHA".."$NEW_HEAD"
git diff --check "$UPSTREAM_SHA".."$NEW_HEAD"
```

Review every `!` or unmatched entry in `git range-diff`. Conflict-resolution
changes are acceptable only when explained in the migration report. Compare the
old and new downstream path inventories and account for every removed or renamed
file. Confirm that:

- all intended downstream commits are represented;
- registrations moved to current entry points;
- no test was silently dropped;
- no active recipe, Dockerfile, job config, or documentation path disappeared;
- the working tree is clean before dependency and runtime validation.

## Phase 5: Reconcile Dependencies and Images

Treat the exact official tree as the dependency baseline. Reapply only intentional
downstream additions.

### Dependency gate

Audit at least:

```text
pyproject.toml
uv.lock
3rdparty/Megatron-LM gitlink
Docker base image tags
Transformer Engine source or system package
PyTorch, CUDA, NCCL, Triton, flash-attn, tilelang
Transformers, vLLM, modelopt
nvidia-resiliency-ext and torch-memory-saver
speech/audio/streaming dependencies
MLflow and AzureML tracking plugin compatibility
```

Then, inside the supported Linux build container:

```bash
git submodule update --init 3rdparty/Megatron-LM
uv lock
uv sync --locked --all-extras --all-groups
```

Review `git diff -- pyproject.toml uv.lock 3rdparty/Megatron-LM` and explain every
transitive movement. A lockfile that parses is not enough; it must match the MCore
gitlink and the Docker install mode.

### Docker gate

Inventory all `docker/Dockerfile.speech-*` files and all active job `IMAGE`
references. Build the official CI image and every custom speech image used by an
active job. Example commands, adjusted to the current stage names:

```bash
docker build -f docker/Dockerfile.ci --target megatron_bridge \
  -t megatron-bridge:forward-port .

docker build -f docker/Dockerfile.speech-torch210 --target speech_torch210 \
  -t megatron-bridge-speech:forward-port .

docker build -f docker/Dockerfile.speech-speechllm --target speech_speechllm \
  -t megatron-bridge-speech:speechllm-forward-port .
```

Do not build an obsolete image merely because its file exists. Build every image
that an active launch config can select. Check Docker comments and bootstrap
scripts for stale version bounds as well as executable instructions.

Run a version/import probe in each candidate runtime image:

```bash
uv run python - <<'PY'
import importlib.metadata as md
import torch
import transformers
import megatron.core
import megatron.bridge
import transformer_engine.pytorch

for package in ("torch", "transformers", "megatron-core", "transformer-engine", "mlflow"):
    try:
        print(package, md.version(package))
    except md.PackageNotFoundError:
        print(package, "system-or-missing")
print("cuda", torch.version.cuda)
PY
```

Also import the speech bridge, streaming data provider, training entry point,
checkpoint loader, inference, and evaluation modules. For AzureML images, verify
the `azureml://` MLflow tracking-store plugin without downgrading the Bridge MLflow
security floor.

## Phase 6: Validation Ladder

Run these gates in order. Stop at the first failure and fix the same layer before
moving on.

### A. Static and focused unit checks

```bash
uv run python -m pytest tests/unit_tests/models/speech_cascades/ -x -v
uv run python -m pytest tests/unit_tests/data/speech_chunk/ -x -v
uv run pre-commit run --all-files
```

Add focused tests for changed shared conversion, checkpoint, recipe, or Qwen
behavior. Do not use unrelated failures as a reason to broaden the task.

### B. Model and checkpoint compatibility

Verify the production model can:

1. register through `AutoBridge` without manual import ordering;
2. load HF/source weights with complete mappings;
3. build under the required TP/PP layout;
4. save a distributed checkpoint;
5. resume weights, optimizer, RNG, sampler, and iteration;
6. export back to HF and pass the existing tensor/parity tolerances.

### C. Real-data two-step training

Use real speech data, the production collator, frozen/unfrozen policy, activation
recompute, and the actual image. Run at least two optimizer steps and one save.
Confirm finite loss, finite grad norm, no skipped/NaN iterations, audio reaches
the encoder in the expected dtype, and the produced checkpoint can be loaded.

Data preparation, AzCopy, recovery, and cleanup must run on the designated H200
host, never on a local/H100 machine. Do not install `phyagi`; load it from source
with `PYTHONPATH` if a validation path requires it.

### D. Distributed save/resume

Run the production distributed layout through train, save, exit, resume, and at
least one additional optimizer step. This catches failures that a single-process
two-step run cannot, including DCP API drift and sampler exhaustion.

### E. Full production and quality gate

Because an upstream forward port changes shared runtime dependencies, production
readiness requires the full representative training job and the established
OpenASR evaluation, not only a smoke test. Compare loss, throughput, peak memory,
checkpoint cadence, and per-subset WER against the last known-good run under the
same data mixture and scorer. Resolve any regression before changing the
baseline.

AMLT jobs follow the review and VC02 low-priority gate in `AGENTS.md`. Record the
resolved config, image digest, experiment/job status, checkpoint, and evaluation
artifact. A submitted or running job is not a pass.

## Phase 7: Check In and Deliver

Keep history reviewable:

- rebased downstream commits preserve their original intent;
- compatibility fixes discovered after replay are separate signed, signed-off
  commits;
- dependency/image changes are isolated where practical;
- generated artifacts, credentials, local paths, account names, and `.env` files
  are never committed.

Before publishing:

```bash
git status --short
git log --oneline "$UPSTREAM_SHA"..HEAD
git diff --check "$UPSTREAM_SHA"..HEAD
git range-diff "$OLD_BASE"..."$OLD_HEAD" "$UPSTREAM_SHA"...HEAD
```

Push a candidate branch first. Update an existing remote branch only with user
approval and `git push --force-with-lease`; never use an unconditional force push.

## Required Migration Report

Report all of the following:

```markdown
## Baselines
- Old downstream HEAD:
- Old official base:
- New official SHA/date/subject:
- New downstream HEAD:
- Backup ref:

## Preservation
- Downstream commits before/after:
- Dirty files captured:
- range-diff exceptions and reasons:
- Renamed/deleted downstream paths accounted for:

## Upstream impact
- Models/recipes:
- Training/data/checkpoint/conversion:
- Dependencies/submodule:
- Docker/CI:

## Validation
- Lock/sync:
- Docker builds and image digests:
- Import/version probes:
- Focused unit tests:
- Real-data two-step:
- Distributed save/resume:
- Full training:
- Quality evaluation:

## Remaining risk
- Blockers:
- Deferred checks:
- Rollback ref:
```

Do not say "fully updated" or "production ready" when Docker, real-data training,
resume, or quality gates remain pending.

## Validated Lessons from This Repository

These are historical examples, not hardcoded future baselines:

- In August 2026, the working branch was based on the off-main `v0.5.1` release.
  Merging official `main` pulled in the release-line divergence and caused broad
  conflicts. Replaying only downstream commits with
  `git rebase --onto upstream/main v0.5.1 <branch>` produced the correct history.
- The first upstream-adapted source tree still failed at runtime because the old
  Torch 2.6 image lacked `torch.float8_e8m0fnu`, required by the newer MCore.
  Rebuilding on the Torch 2.10 stack removed that import blocker.
- Passing speech unit tests was necessary but insufficient. Real-data training
  exposed a changed finetune checkpoint contract and a GDN backward backend
  requirement; the full distributed run later exposed MLflow plugin and audio
  decoding environment drift.
- Production equivalence was established only after a full 12k-step run and
  matched OpenASR evaluation, not after the rebase or image import check.
- On 2026-09-11, the next official range after `084d35688` contained 97 commits
  and changed 382 files. It made no direct `speech_cascades` edits, but changed
  MCore, Transformer Engine, Transformers, checkpointing, PEFT, training, data,
  Qwen3.5 recipes/providers, Docker, and vLLM. This is precisely why indirect
  compatibility gates are mandatory.

## Common Failure Patterns

| Symptom | Likely cause | Correct response |
|---|---|---|
| Hundreds of release-wide conflicts | Merge base is an off-main release line | Abort, inventory branch-only commits, use `rebase --onto` |
| Clean rebase but MCore import fails | Old image does not satisfy new MCore/PyTorch/TE requirements | Reconcile and rebuild the image before code debugging |
| `uv.lock` conflict | Lockfile was merged by hand or MCore pin changed | Resolve project intent and regenerate in Linux container |
| Tests pass outside image but job fails | Runtime image or bootstrap changes package versions | Probe and test inside the exact candidate image |
| Cold start works, resume fails | Checkpoint/DCP/tracker/RNG API drift | Add an explicit distributed save-resume gate |
| Metrics absent in AzureML | Tracking-store plugin missing or incompatible | Verify plugin registration and preserve MLflow security floor |
| Evaluation fails after dependency bump | Dataset audio backend or codec behavior changed | Use explicit decode path and rerun matched quality evaluation |
| Good smoke loss but WER regresses | Data mixture, prompt, scorer, or training config differs | Compare actual manifests/configs and run matched OpenASR |
