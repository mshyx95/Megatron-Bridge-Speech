# Amulet submissions

Submission YAMLs live in `<project>/train/` or `<project>/eval/`. Runtime scripts,
training profiles and data configs live under [`projects/<project>/`](../projects/).

## ELIFT: five training entries, one shared evaluation entry

Select one training entry, edit its inputs if needed, then run from the repository root:

```bash
amlt run amulet/ELIFT/train/b2.yaml
amlt run amulet/ELIFT/train/p4.yaml
amlt run amulet/ELIFT/train/p4-hindi-2k.yaml
amlt run amulet/ELIFT/train/p5.yaml
amlt run amulet/ELIFT/train/new-hindi-2k.yaml
amlt run amulet/ELIFT/eval/eval.yaml
```

No external exports, stage selection, private includes or extra CLI inputs are
required. Amulet authentication/project setup and the actual input artifacts
are still required. Training defaults to 4 x 8 H200 GPUs.

| Entry | Training config | Data | Model / encoder | Initialization |
| --- | --- | --- | --- | --- |
| [B2](ELIFT/train/b2.yaml) | [b2.yaml](../projects/ELIFT/train/b2.yaml) | base | 9B / Phi4-MM-24L | Clean LLM + random MLP |
| [P4](ELIFT/train/p4.yaml) | [p4.yaml](../projects/ELIFT/train/p4.yaml) | base | 9B / S1-2605 | Complete E1-r + native MTP4 |
| [P4-Hindi](ELIFT/train/p4-hindi-2k.yaml) | [p4-hindi-2k.yaml](../projects/ELIFT/train/p4-hindi-2k.yaml) | base + Hindi 2k | 9B / S1-2605 | Complete P4 step4000 |
| [P5](ELIFT/train/p5.yaml) | [p5.yaml](../projects/ELIFT/train/p5.yaml) | base | 27B / Phi4-MM-24L | Complete B3 + native MTP4 |
| [New Hindi 2k](ELIFT/train/new-hindi-2k.yaml) | [new-tier1-hindi-2k-e1r.yaml](../projects/ELIFT/train/new-tier1-hindi-2k-e1r.yaml) | new-2k + Hindi 2k | 9B / S1-2605 | Clean LLM + random MLP |

P4/P5 intentionally leave `INIT_CHECKPOINT: none`: fill a real, validated complete
native MTP checkpoint in the selected file before launching. Plain E1-r/B3
checkpoints without the four physical MTP layers are not substitutes.
P4-Hindi derives its input from the completed P4 under the shared output root;
it does not automatically launch P4 or wait for it. New Hindi 2k is **not**
a P4 continuation; its existing English-5k / other-Tier1-2k convention is preserved.

### Shared path prefixes

Edit [`ELIFT/paths.yaml`](ELIFT/paths.yaml) once to change shared output, model,
encoder, normalization or evaluation-data directories. Each entry names its
training/data YAML and selects the corresponding directory anchors directly:

- `BASE_MODEL_ROOT` + `BASE_MODEL_NAME` select the clean LLM and training tokenizer.
- `ENCODER_DIR/model.checkpoint` selects the encoder checkpoint.
- `NORMALIZATION_DIR/mean.vec` and `NORMALIZATION_DIR/Invstd.vec` share one prefix.
  B2/P5 use the same directory anchor for encoder and normalization; S1-2605
  genuinely stores its normalization files elsewhere.
- `OUTPUT_ROOT/RUN_NAME` contains `initial_hf` and generated `data.yaml`.
  The existing recipe saves checkpoints beneath
  `OUTPUT_ROOT/RUN_NAME/RUN_NAME/checkpoints/`.

The shared [training command template](templates/elift-train.yaml) contains
execution boilerplate only; it does not select an experiment from a catalog.
Use a fresh `RUN_NAME` or output root when changing a training experiment.

### Switching evaluation

Normally edit only these fields in [`eval.yaml`](ELIFT/eval/eval.yaml):

```yaml
RUN_NAME: p5
CHECKPOINT_ITER: 4000
EVAL_SUITE: voicebench # openasr | bbh | voicebench
```

The HF/checkpoint paths, matching tokenizer, data manifest, token budget, job
name and result directory follow automatically. The prepared HF contains its
tokenizer files, so switching to P5 does not leave evaluation using a 9B tokenizer.
Results are isolated at
`OUTPUT_ROOT/RUN_NAME/eval_output/<suite>_iter<step>/`; this replaces the earlier
unqualified `openasr` output directory. Data/path overrides remain visible in
the evaluation YAML. VoiceBench scoring/judging is opt-in; BBH paid judging
still requires separately supplied secrets. Default resources: 1 x 8 VC02 H100 GPUs.

H200 and VC02 mount different storage accounts. Before evaluating on VC02,
synchronize both the prepared HF directory and native checkpoint to its
model storage at the configured paths. Alternatively, change the evaluation
`requires` include to the H200 template to read the training storage directly;
no synchronization is performed by either entry.

Optional validation only (does not submit):

```bash
amlt run --dump amulet/ELIFT/train/p4-hindi-2k.yaml
amlt run --dump amulet/ELIFT/eval/eval.yaml
```

Other historical runtime profiles and scripts remain in
[`projects/ELIFT/`](../projects/ELIFT/), but are not additional active submission
entries. Standalone legacy data YAMLs
can be selected with `DATA_RECIPE: none`; any environment variables referenced
by those legacy configs must also be defined inside the submission YAML.

## Reusing templates

[`templates/vc02.yaml`](templates/vc02.yaml) and
[`templates/h200.yaml`](templates/h200.yaml) define common environment defaults,
targets, images, mounts, repository upload roots and job resources. Concrete
configs retain their experiment-specific inputs. The five training entries
also include the shared directory anchors and training command template.

Use Amulet's native `requires` / `!include` and YAML merge anchors:

```yaml
requires:
  - !include ../../templates/vc02.yaml

env_defaults:
  <<: *elift_env
  NODES: 1
  JOB_NAME: example-eval

<<: *elift_config

jobs:
  - <<: *elift_job
    command:
      - bash projects/ELIFT/eval/run_speech_cascades_openasr.sh
```

VC02 keeps low priority and standard SLA; H200 keeps its existing high priority
and premium SLA. Both expose the same anchors: change only the `requires`
include to switch infrastructure, then review the node count and runtime inputs.
All configs upload the
repository root, not the submission directory. Plain `yaml.safe_load` cannot
resolve cross-file aliases or `!include`; use `amlt run --dump`.
Amulet substitutes variables only once. Directory anchors therefore hold literal
paths; the job exports shared prefixes before expanding derived paths in the shell.
No recursive Amulet substitution or caller-side exports are assumed.
