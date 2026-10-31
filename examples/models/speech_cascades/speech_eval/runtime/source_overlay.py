"""Guarded, in-memory equivalents of the existing MTP installer changes."""

import ast
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

from .checkpoint import CompatibilityError


ADAPTER_MODULE = "vllm.model_executor.models.qwen3_5_audio"
ADAPTER_PATH = Path(__file__).with_name("qwen3_5_audio.py")
MODULES = {
    ADAPTER_MODULE: "model_executor/models/qwen3_5_audio.py",
    "vllm.model_executor.models.phi4mm_audio": "model_executor/models/phi4mm_audio.py",
    "vllm.config.speculative": "config/speculative.py",
    "vllm.v1.worker.gpu_model_runner": "v1/worker/gpu_model_runner.py",
}
EXPANDED_TEST = (
    'self.model_config.hf_config.model_type == "qwen3_5_text" and '
    'getattr(self.model_config.hf_config, "mtp_num_hidden_layers", 0) == 4 and '
    '"Qwen3_5AudioForCausalLM" in (self.model_config.hf_config.architectures or [])'
)


def transform_source(module: str, source: str, mode: str, proposer: str) -> ast.Module:
    tree = ast.parse(source)
    if module == ADAPTER_MODULE:
        return tree
    matches = 0
    if module.endswith("phi4mm_audio"):

        class EncoderTransformer(ast.NodeTransformer):
            def visit_Assign(self, node):
                nonlocal matches
                if ast.unparse(node) == "self.encoder = ConformerEncoder(**encoder_config)":
                    matches += 1
                    return ast.parse(
                        "encoder_config = dict(encoder_config)\n"
                        'encoder_config.pop("use_flashattention_t5bias", None)\n'
                        "self.encoder = ConformerEncoder(**encoder_config)"
                    ).body
                return node

        tree = EncoderTransformer().visit(tree)
    elif module.endswith("speculative"):

        class SpeculativeTransformer(ast.NodeTransformer):
            def visit_If(self, node):
                nonlocal matches
                self.generic_visit(node)
                is_qwen_branch = "qwen3_5" in ast.unparse(node.test)
                converts_mtp = any(
                    isinstance(statement, ast.Assign)
                    and ast.unparse(statement) == "hf_config.model_type = 'qwen3_5_mtp'"
                    for statement in node.body
                )
                if is_qwen_branch and converts_mtp:
                    matches += 1
                    node.test = ast.parse(
                        "(" + ast.unparse(node.test) + ") or "
                        '(hf_config.model_type == "qwen3_5_text" and '
                        'getattr(hf_config, "mtp_num_hidden_layers", 0) in (1, 4) and '
                        '"Qwen3_5AudioForCausalLM" in (hf_config.architectures or []))',
                        mode="eval",
                    ).body
                return node

        tree = SpeculativeTransformer().visit(tree)
    elif module.endswith("gpu_model_runner") and mode == "ExpandedMTP4":
        proposer_module = "speech_eval.mtp4.proposer" if proposer == "stable" else "speech_eval.mtp4_batched.proposer"
        branch = (
            f"if {EXPANDED_TEST}:\n"
            f"    from {proposer_module} import ExpandedMTP4Proposer\n"
            "    self.drafter = ExpandedMTP4Proposer(self.vllm_config, self.device, self)\n"
            "else:\n"
            "    self.drafter = EagleProposer(self.vllm_config, self.device, self)"
        )
        has_embeddings = "hasattr(self.drafter, 'set_target_embeddings')" in ast.unparse(tree)
        embedding_matches = 0

        class RunnerTransformer(ast.NodeTransformer):
            def visit_If(self, node):
                nonlocal matches
                if any(
                    isinstance(statement, ast.ImportFrom)
                    and any(alias.name == "ExpandedMTP4Proposer" for alias in statement.names)
                    for statement in node.body
                ):
                    matches += 1
                    return ast.parse(branch).body[0]
                return self.generic_visit(node)

            def visit_Assign(self, node):
                nonlocal matches, embedding_matches
                text = ast.unparse(node)
                if text == "self.drafter = EagleProposer(self.vllm_config, self.device, self)":
                    matches += 1
                    return ast.parse(branch).body[0]
                if (
                    text.startswith(
                        "draft_token_ids = self.drafter.propose(num_speculative_tokens=num_spec_tokens_to_schedule,"
                    )
                    and isinstance(node.value, ast.Call)
                    and any(keyword.arg == "target_token_ids" for keyword in node.value.keywords)
                ):
                    embedding_matches += 1
                    if not has_embeddings:
                        before = ast.parse(
                            'if hasattr(self.drafter, "set_target_embeddings"):\n'
                            "    self.drafter.set_target_embeddings(self.inputs_embeds.gpu[:num_scheduled_tokens] "
                            "if spec_decode_metadata is None else self.inputs_embeds.gpu[token_indices])"
                        ).body
                        return before + [node]
                return node

        tree = RunnerTransformer().visit(tree)
        if embedding_matches != 1:
            raise CompatibilityError(
                "Unsupported vLLM runner: accepted-embedding proposal anchor missing or ambiguous"
            )
    else:
        raise CompatibilityError(f"Unexpected overlay module: {module}")
    if matches != 1:
        raise CompatibilityError(f"Unsupported vLLM source: {module}: expected one anchor, found {matches}")
    return ast.fix_missing_locations(tree)


def active_modules(mode: str) -> dict:
    return {
        module: relative
        for module, relative in MODULES.items()
        if module == ADAPTER_MODULE
        or module.endswith("phi4mm_audio")
        or mode != "standard"
        and (module.endswith("speculative") or mode == "ExpandedMTP4")
    }


def validate_sources(root: Path, mode: str, proposer: str) -> dict:
    evidence = {}
    required_symbols = {
        "model_executor/models/qwen3_5_audio.py": "Qwen3_5AudioForCausalLM",
        "model_executor/models/registry.py": "Qwen3_5AudioForCausalLM",
    }
    if mode == "NativeMTP1":
        required_symbols["model_executor/models/qwen3_5_mtp.py"] = "Qwen3_5MTP"
    for relative, symbol in required_symbols.items():
        path = ADAPTER_PATH if relative == MODULES[ADAPTER_MODULE] else root / "vllm" / relative
        try:
            source = path.read_text()
        except OSError as error:
            raise CompatibilityError(f"Required serving source unavailable: {path}") from error
        tree = ast.parse(source)
        if not any(
            isinstance(node, ast.ClassDef)
            and node.name == symbol
            or isinstance(node, ast.Constant)
            and node.value == symbol
            for node in ast.walk(tree)
        ):
            raise CompatibilityError(f"Required serving symbol {symbol} missing: {path}")
        evidence[relative] = hashlib.sha256(source.encode()).hexdigest()
    for module, relative in active_modules(mode).items():
        path = ADAPTER_PATH if module == ADAPTER_MODULE else root / "vllm" / relative
        try:
            source = path.read_text()
        except OSError as error:
            raise CompatibilityError(f"Required serving source unavailable: {path}") from error
        compile(transform_source(module, source, mode, proposer), str(path), "exec")
        evidence[relative] = hashlib.sha256(source.encode()).hexdigest()
    return evidence


class OverlayLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode("utf-8")
        tree = transform_source(
            fullname, source, os.environ["SPEECH_RUNTIME_MODE"], os.environ["SPEECH_RUNTIME_PROPOSER"]
        )
        return compile(tree, self.path, "exec")


class OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in active_modules(os.environ["SPEECH_RUNTIME_MODE"]):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ImportError(f"Speech runtime needs Python source for {fullname}")
        expected = Path(os.environ["SPEECH_RUNTIME_VLLM_ROOT"]) / "vllm" / MODULES[fullname]
        if Path(spec.origin).resolve() != expected.resolve():
            raise ImportError(f"Speech runtime source shadowed: {fullname}")
        if fullname == ADAPTER_MODULE:
            return importlib.util.spec_from_file_location(
                fullname, ADAPTER_PATH, loader=OverlayLoader(fullname, str(ADAPTER_PATH))
            )
        spec.loader = OverlayLoader(fullname, spec.origin)
        return spec


def install() -> None:
    sys.meta_path.insert(0, OverlayFinder())
