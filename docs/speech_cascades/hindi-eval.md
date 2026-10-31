# Hindi text and audio evaluation

These portable tools separate dataset preparation, prediction generation,
deterministic diagnostics and optional paid judging. They do not contain endpoint
bindings, credentials, downloaded datasets, predictions or historical job records.
CPU regressions mock network calls; they do not establish current live-service
availability or benchmark scores.

## Prepare inputs

Run the [asset preparer](../../examples/models/speech_cascades/prepare_hindi_eval_assets.py)
in an environment with Hugging Face Hub and PyArrow already installed:

```bash
uv run python examples/models/speech_cascades/prepare_hindi_eval_assets.py --help
```

Preparation records the resolved source revision, validates the requested Hindi
inventory and exports normalized test rows without gold-answer leakage into
prompts. Conflicting answer labels are retained as unscored rather than silently
corrected. VoiceAgentBench audio references must resolve inside the downloaded
root. MILU requires separately accepted gated access; no credentials are supplied
by the repository.

## Generate predictions

- [Text evaluation](../../examples/models/speech_cascades/eval_hindi_text.py)
  uses an existing loopback OpenAI-compatible endpoint, strict choice-letter
  parsing, source/model/prompt-bound resume checks and ordered checkpointing.
- [Voice evaluation](../../examples/models/speech_cascades/eval_voiceagentbench_hindi.py)
  preserves recorded audio and assistant history. Supply the official
  VoiceAgentBench checkout explicitly for its parser.
- [Subset runner](../../examples/models/speech_cascades/run_voiceagentbench_hindi.py)
  assigns one distinct existing local server port per requested subset and
  propagates subprocess failures.
- [Realtime evaluation](../../examples/models/speech_cascades/eval_hindi_realtime.py)
  reads an explicit endpoint config and credential file, validates a GA Realtime
  WebSocket deployment, and converts recordings to 24 kHz mono PCM16. It preserves
  assistant messages, collects complete responses and distinguishes explicitly
  retryable failures from invalid/incomplete output. Low-effort text results are
  a separate protocol and cannot resume default-effort results.

Run each entrypoint with `--help` for its required inputs. Realtime additionally
requires compatible `websockets`, `soxr`, NumPy and SoundFile packages in the
chosen environment. No project dependency manifests are changed by these tools.
Live Realtime and judge calls can incur charges; do not invoke them as a smoke
test without separate authorization.

## Score and judge

The [deterministic scorer](../../examples/models/speech_cascades/score_voiceagentbench_hindi.py)
requires complete, ordered source/model/hash coverage and reports tool-name and
strict-argument diagnostics. Safety and dependent-call semantics remain unscored
without the official judge. These diagnostics are not replacements for the
official LLM evaluation.

The [official-tool judge adapter](../../examples/models/speech_cascades/judge_voiceagentbench_hindi_tools.py)
uses a pinned upstream evaluator and configurable Azure deployment. Both a resource
endpoint and an `/openai/v1` API base are supported; credential-bearing or
non-Azure endpoint URLs are rejected. Replies must match the requested deployment
and rubric format. Content-filtered calls are distinguished from scored failures.
The [safety adapter](../../examples/models/speech_cascades/judge_voiceagentbench_hindi_safety.py)
loads the upstream refusal rubric and requires an explicit, unambiguous verdict.
Judge authentication is acquired at runtime through the Azure CLI; tokens are not
stored in results.

## Offline validation

In the existing speech training environment, with the Realtime example's optional
WebSocket dependency available:

```bash
UV_NO_SYNC=1 uv run python -m pytest \
  tests/unit_tests/examples/test_hindi_eval_assets.py \
  tests/unit_tests/examples/test_hindi_judge_endpoint.py \
  tests/unit_tests/examples/test_hindi_text_workers.py -q
```

These checks use fixtures and mocked HTTP/WebSocket/Azure requests, not paid APIs.
