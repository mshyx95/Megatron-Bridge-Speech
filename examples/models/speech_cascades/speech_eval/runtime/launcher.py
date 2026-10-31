"""Compose and optionally exec the existing OpenAI serving entry point."""

import argparse
import json
import os
import sys
from pathlib import Path

from .checkpoint import CompatibilityError, inspect_checkpoint, positive_integer
from .source_overlay import validate_sources


REPO_ROOT = Path(__file__).resolve().parents[2]
MODES = ("standard", "NativeMTP1", "ExpandedMTP4")
ENV_PREFIXES = ("MTP_BATCHED_", "SPEECH_RUNTIME_")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="CPU-only inspection and explicit Qwen3.5 speech serving")
    commands = root.add_subparsers(dest="action", required=True)
    inspect = commands.add_parser("inspect", help="Validate checkpoint config, index, and shard headers")
    inspect.add_argument("checkpoint", type=Path)
    inspect.add_argument("--index-only", action="store_true", help="Do not read shards; not launch validation")
    launch = commands.add_parser("launch", help="Plan or execute the existing vLLM OpenAI server")
    launch.add_argument("checkpoint", type=Path)
    launch.add_argument("--dry-run", action="store_true", help="Print JSON only; never import vLLM or use a GPU")
    launch.add_argument("--mode", choices=MODES, default="standard")
    launch.add_argument("--speculation", choices=("sequential", "parallel"), default="sequential")
    launch.add_argument("--num-speculative-tokens", type=int)
    launch.add_argument("--proposer", choices=("stable", "batched"), default="stable")
    launch.add_argument("--allow-experimental-batched", action="store_true")
    launch.add_argument("--batched-use-graphs", action="store_true")
    launch.add_argument("--batched-sdpa", choices=("auto", "math", "flash"), default="auto")
    launch.add_argument("--python", default=sys.executable, help="Target vLLM interpreter, e.g. /opt/vllm/bin/python")
    launch.add_argument(
        "--vllm-root",
        type=Path,
        default=Path("/opt/vllm-dev"),
        help="Directory containing the target vllm source package",
    )
    launch.add_argument("--model-path", type=Path, help="Target mount path when planning for another environment")
    launch.add_argument("--repo-path", type=Path, help="Target source bind mount; defaults to this repository")
    launch.add_argument("--tokenizer", type=Path, help="Target tokenizer directory; defaults to target model path")
    launch.add_argument(
        "--chat-template", type=Path, help="Target template; defaults to the repository speech template"
    )
    launch.add_argument("--served-model-name", default="qwen35-speech")
    launch.add_argument("--host", default="127.0.0.1")
    launch.add_argument("--port", type=int, default=8000)
    launch.add_argument("--tensor-parallel-size", type=int, default=1)
    launch.add_argument("--max-model-len", type=int, default=8192)
    launch.add_argument("--max-num-seqs", type=int, default=2)
    launch.add_argument("--max-num-batched-tokens", type=int)
    launch.add_argument("--max-audio-per-prompt", type=int, default=128)
    launch.add_argument("--gpu-memory-utilization", type=float, default=0.42)
    launch.add_argument("--reasoning-parser", choices=("qwen3",))
    launch.add_argument(
        "--evaluation-mode",
        action="store_true",
        help="Allow trusted native benchmark request templates and disable output parsers",
    )
    launch.add_argument(
        "--native-tools",
        action="store_true",
        help="Parse native tool calls for turn-based evaluation instead of preserving raw output",
    )
    launch.add_argument(
        "--expect-encoder",
        choices=("Phi4MM", "Checkpoint22L"),
        help="Assert a structural profile, not encoder weight provenance",
    )
    launch.add_argument("--expect-projector", choices=("GatedProj", "MLP"))
    return root


def build_plan(args: argparse.Namespace) -> dict:
    if args.proposer != "stable":
        raise CompatibilityError("Only the repository-owned stable proposer is supported")
    report = inspect_checkpoint(args.checkpoint)
    if args.evaluation_mode and args.reasoning_parser:
        raise CompatibilityError("evaluation-mode preserves raw native output and cannot use a reasoning parser")
    if args.native_tools and not args.evaluation_mode:
        raise CompatibilityError("--native-tools requires --evaluation-mode")
    if args.speculation == "parallel":
        raise CompatibilityError(
            "Parallel drafting is not implemented by the checked-in speech MTP proposers; request batching is not parallel depth speculation"
        )
    if args.expect_projector and report["projector"]["name"] != args.expect_projector:
        raise CompatibilityError("Projector does not match --expect-projector")
    if args.expect_encoder and report["encoder"]["structural_profile"] != args.expect_encoder:
        raise CompatibilityError("Encoder structure does not match --expect-encoder")
    retained = report["mtp"]["retained_physical_layers"]
    if report["encoder"]["structural_profile"] is None:
        raise CompatibilityError("Encoder does not match the Phi4MM or Checkpoint22L structural profiles")
    expected = {"standard": 0, "NativeMTP1": 1, "ExpandedMTP4": 4}[args.mode]
    if expected and retained != expected:
        raise CompatibilityError(f"{args.mode} requires {expected} retained physical MTP layers; found {retained}")
    tokens = args.num_speculative_tokens if args.num_speculative_tokens is not None else expected
    if args.mode == "standard" and tokens != 0:
        raise CompatibilityError("standard mode cannot request speculative tokens")
    if expected:
        positive_integer(tokens, "num_speculative_tokens")
    if args.mode == "ExpandedMTP4" and tokens != 4:
        raise CompatibilityError("The existing ExpandedMTP4 proposer requires exactly four speculative tokens")
    if args.mode == "ExpandedMTP4" and args.tensor_parallel_size != 1:
        raise CompatibilityError("The existing ExpandedMTP4 proposer requires tensor_parallel_size=1")
    if args.proposer == "batched" and (args.mode != "ExpandedMTP4" or not args.allow_experimental_batched):
        raise CompatibilityError(
            "batched requires ExpandedMTP4 and --allow-experimental-batched; illegal CUDA access remains unresolved"
        )
    if args.proposer != "batched" and (
        args.allow_experimental_batched or args.batched_use_graphs or args.batched_sdpa != "auto"
    ):
        raise CompatibilityError("Batched options require --proposer batched")
    for field in ("tensor_parallel_size", "max_model_len", "max_num_seqs", "max_audio_per_prompt"):
        positive_integer(getattr(args, field), field)
    if not 1 <= args.port <= 65535 or not 0 < args.gpu_memory_utilization <= 1:
        raise CompatibilityError("port must be 1..65535 and gpu-memory-utilization must be in (0, 1]")
    budget = args.max_num_batched_tokens if args.max_num_batched_tokens is not None else args.max_model_len
    positive_integer(budget, "max_num_batched_tokens")
    repository = (args.repo_path or REPO_ROOT).absolute()
    model = (args.model_path or args.checkpoint.resolve()).absolute()
    vllm_root = args.vllm_root.absolute()
    tokenizer = (args.tokenizer or model).absolute()
    template = (args.chat_template or repository / "speech_eval/runtime/audio_chat_template.jinja").absolute()
    runtime = repository / "speech_eval/runtime"
    paths = [str(runtime / "overlay"), str(repository)]
    if args.mode == "ExpandedMTP4":
        paths.append(str(repository / "speech_eval/mtp4"))
    paths.append(str(vllm_root))
    environment = {
        "PYTHONPATH": os.pathsep.join(paths),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "SPEECH_RUNTIME_MODE": args.mode,
        "SPEECH_RUNTIME_PROPOSER": args.proposer,
        "SPEECH_RUNTIME_VLLM_ROOT": str(vllm_root),
    }
    if args.proposer == "batched":
        environment.update(
            {
                "MTP_BATCHED_MAX_BATCH": str(args.max_num_seqs),
                "MTP_BATCHED_USE_GRAPHS": str(int(args.batched_use_graphs)),
                "MTP_BATCHED_VERIFY": "0",
                "MTP_BATCHED_SDPA": args.batched_sdpa,
                "MTP_BATCHED_VERIFY_MODE": "elementwise",
                "MTP_BATCHED_MAX_SUFFIX": "16",
                "MTP_BATCHED_MAX_GRAPHS": "8",
                "MTP_BATCHED_GRAPH_MB": "512",
                "MTP_BATCHED_CACHE_MB": "2048",
                "MTP_BATCHED_EAGER_SUFFIX": "256",
                "MTP_BATCHED_PREPARE_CONTEXT": "0",
            }
        )
    command = [args.python, "-m", "vllm.entrypoints.openai.api_server"]
    for flag, value in (
        ("model", model),
        ("tokenizer", tokenizer),
        ("chat-template", template),
        ("served-model-name", args.served_model_name),
        ("host", args.host),
        ("port", args.port),
        ("tensor-parallel-size", args.tensor_parallel_size),
        ("max-model-len", args.max_model_len),
        ("max-num-seqs", args.max_num_seqs),
        ("max-num-batched-tokens", budget),
        ("gpu-memory-utilization", args.gpu_memory_utilization),
        ("limit-mm-per-prompt", json.dumps({"audio": args.max_audio_per_prompt})),
    ):
        command.extend(["--" + flag, str(value)])
    command.extend(["--trust-remote-code", "--enforce-eager"])
    if args.evaluation_mode:
        command.extend(["--trust-request-chat-template", "--generation-config", "vllm"])
    if not args.evaluation_mode or args.native_tools:
        command.extend(["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"])
    if args.reasoning_parser:
        command.extend(["--reasoning-parser", args.reasoning_parser])
    if expected:
        speculative = {"method": "mtp", "num_speculative_tokens": tokens}
        if args.mode == "ExpandedMTP4":
            speculative["disable_padded_drafter_batch"] = True
            command.extend(["--override-generation-config", '{"temperature":0}'])
        command.extend(
            [
                "--no-enable-prefix-caching",
                "--mamba-cache-mode",
                "align",
                "--speculative-config",
                json.dumps(speculative),
            ]
        )
    else:
        command.append("--enable-prefix-caching")
    source_evidence = validate_sources(vllm_root, args.mode, args.proposer) if vllm_root.exists() else None
    missing = [
        str(path)
        for path in (
            model / "config.json",
            tokenizer / "tokenizer_config.json",
            template,
            runtime / "overlay/sitecustomize.py",
        )
        if not path.is_file()
    ]
    if args.mode == "ExpandedMTP4":
        required_proposer_files = [
            repository / "speech_eval/mtp4/mtp_model.py",
            repository / "speech_eval/mtp4/proposer.py",
        ]
        if args.proposer == "batched":
            required_proposer_files.extend(
                repository / "speech_eval/mtp4_batched" / name for name in ("proposer.py", "batched_model.py")
            )
        missing.extend(str(path) for path in required_proposer_files if not path.is_file())
    if source_evidence is None:
        missing.append(str(vllm_root / "vllm"))
    warnings = list(report["warnings"])
    if args.evaluation_mode:
        output_mode = "parse native tool calls" if args.native_tools else "preserve raw outputs"
        warnings.append(
            f"Evaluation mode accepts trusted per-request templates; expose only to trusted benchmark clients and {output_mode}."
        )
    if args.mode == "NativeMTP1" and tokens > 1:
        warnings.append(
            "NativeMTP1 repeats the same physical layer for multiple sequential draft tokens; this is not ExpandedMTP4."
        )
    if args.mode == "ExpandedMTP4":
        warnings.append(
            "ExpandedMTP4 accepts greedy requests only (temperature=0); four independent physical depths are loaded without copying weights."
        )
    if args.proposer == "batched":
        warnings.append("EXPERIMENTAL: unresolved illegal CUDA access; disabling graphs is not a verified fix.")
    if missing:
        warnings.append(
            "Target paths are not all available here. Dry-run is a composition plan, not a startup validation."
        )
    return {
        "schema_version": 1,
        "command": command,
        "env": environment,
        "environment_policy": {
            "inherit": "parent environment, including scheduler-selected devices; never printed",
            "remove_prefixes": list(ENV_PREFIXES),
            "replace": ["PYTHONPATH"],
        },
        "checkpoint": report,
        "mode": args.mode,
        "speculation": args.speculation,
        "evaluation_mode": args.evaluation_mode,
        "native_tools": args.native_tools,
        "proposer": args.proposer,
        "num_speculative_tokens": tokens,
        "source_sha256": source_evidence,
        "missing_target_paths": missing,
        "warnings": warnings,
    }


def launch_environment(plan: dict, parent: dict) -> dict:
    environment = {name: value for name, value in parent.items() if not name.startswith(ENV_PREFIXES)}
    environment.update(plan["env"])
    return environment


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "inspect":
            result = inspect_checkpoint(args.checkpoint, index_only=args.index_only)
        else:
            result = build_plan(args)
            if not args.dry_run:
                if result["missing_target_paths"]:
                    raise CompatibilityError(
                        "Target launch files unavailable: " + ", ".join(result["missing_target_paths"])
                    )
                model_argument = result["command"][result["command"].index("--model") + 1]
                if Path(result["checkpoint"]["checkpoint"]) != Path(model_argument).resolve():
                    raise CompatibilityError(
                        "--model-path remapping is dry-run only; inspect and launch the target checkpoint in its own environment"
                    )
                os.execvpe(result["command"][0], result["command"], launch_environment(result, os.environ))
                return 0
        print(json.dumps(result, indent=2))
        return 0
    except (CompatibilityError, OSError, SyntaxError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
