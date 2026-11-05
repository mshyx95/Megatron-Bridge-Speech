import json
import os
import re
import subprocess
import sys
from pathlib import Path
from string import Template

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects/ELIFT"
SUBMISSIONS = ROOT / "amulet/ELIFT"
PUBLIC_SUBMISSIONS = (
    "train/b2.yaml",
    "train/p4.yaml",
    "train/p4-hindi-2k.yaml",
    "train/p5.yaml",
    "train/new-hindi-2k.yaml",
    "eval/eval.yaml",
)


def _submission(relative_path, family=None, paths_text=None):
    text = (SUBMISSIONS / relative_path).read_text()
    includes = re.findall(r"^  - !include (\S+)", text, re.MULTILINE)
    sources = []
    for index, include in enumerate(includes):
        path = (SUBMISSIONS / relative_path).parent / include
        if index == 0 and family is not None:
            path = ROOT / "amulet/templates" / f"{family}.yaml"
        sources.append(paths_text if path.resolve() == SUBMISSIONS / "paths.yaml" and paths_text else path.read_text())
    body = re.sub(r"^requires:\n(?:  - !include [^\n]+\n)+\n", "", text, flags=re.MULTILINE)
    return yaml.safe_load("\n".join([*sources, body]))


@pytest.mark.unit
def test_project_layout_separates_submission_configs_and_runtime_files():
    assert {path.name for path in PROJECT.iterdir() if path.is_dir()} == {"data", "train", "reference", "eval"}
    assert {path.name for path in PROJECT.glob("*.yaml")} == {"catalog.yaml", "launch.yaml"}
    assert not list(PROJECT.glob("*.sh"))
    assert {str(path.relative_to(SUBMISSIONS)) for path in SUBMISSIONS.rglob("*.yaml")} == {
        *PUBLIC_SUBMISSIONS,
        "paths.yaml",
    }
    assert not list(PROJECT.rglob("amlt_*.yaml"))


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", PUBLIC_SUBMISSIONS)
def test_shared_templates_preserve_resource_and_upload_contracts(relative_path):
    family = "h200" if relative_path.startswith("train/") else "vc02"
    text = (SUBMISSIONS / relative_path).read_text()
    assert f"!include ../../templates/{family}.yaml" in text
    config = _submission(relative_path)
    job = config["jobs"][0]
    assert job["mpi"] is True
    assert job["process_count_per_node"] == 8
    assert job["sku"] == "${NODES}x${SKU}"
    assert job["priority"] == ("high" if family == "h200" else "low")
    assert config["env_defaults"]["sla_tier"] == ("premium" if family == "h200" else "standard")
    assert config["env_defaults"]["NODES"] == (4 if family == "h200" else 1)
    assert (SUBMISSIONS / relative_path).parent.joinpath(
        config["code"]["local_dir"].replace("$CONFIG_DIR/", "")
    ).resolve() == ROOT


@pytest.mark.unit
def test_all_submission_and_runner_references_resolve_in_the_new_layout():
    for path in (
        *SUBMISSIONS.rglob("*.yaml"),
        ROOT / "amulet/templates/elift-train.yaml",
        *PROJECT.rglob("*.sh"),
    ):
        text = path.read_text()
        assert "projects/cluster/" not in text, path
        for reference in re.findall(r"projects/ELIFT/[\w./-]+\.(?:yaml|sh)", text):
            assert (ROOT / reference).is_file(), (path, reference)
        if path in [SUBMISSIONS / relative for relative in PUBLIC_SUBMISSIONS]:
            includes = re.findall(r"!include (\S+\.yaml)", text)
            assert includes, path
            assert all((path.parent / include).is_file() for include in includes), path
            assert ".local.yaml" not in text
            assert "STAGE" not in text


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", PUBLIC_SUBMISSIONS)
@pytest.mark.parametrize("family", ["h200", "vc02"])
def test_either_entry_can_reuse_either_infrastructure_template(relative_path, family):
    path = SUBMISSIONS / relative_path
    config = _submission(relative_path, family)
    assert config["jobs"][0]["priority"] == ("high" if family == "h200" else "low")
    assert config["env_defaults"]["sla_tier"] == ("premium" if family == "h200" else "standard")
    upload_root = config["code"]["local_dir"].replace("$CONFIG_DIR/", "")
    assert (path.parent / upload_root).resolve() == ROOT


@pytest.mark.unit
def test_entrypoints_define_every_submission_variable_without_external_exports():
    for relative_path in PUBLIC_SUBMISSIONS:
        config = _submission(relative_path)
        commands = json.dumps([config["jobs"], config["env_defaults"]])
        variables = set(re.findall(r"(?<!\$)\$\{(\w+)\}", commands))
        assert variables <= config["env_defaults"].keys()
        assert all(config["env_defaults"][name] is not None for name in variables)


@pytest.mark.unit
def test_evaluation_defaults_match_the_actual_training_output_layout():
    train = _submission("train/b2.yaml")["env_defaults"]
    evaluate = _submission("eval/eval.yaml")["env_defaults"]
    assert evaluate["RUN_NAME"] == train["RUN_NAME"] == "b2"
    assert evaluate["OUTPUT_ROOT"] == train["OUTPUT_ROOT"]
    assert evaluate["MODEL_HF"] == "${OUTPUT_ROOT}/${RUN_NAME}/initial_hf"
    assert evaluate["CHECKPOINT_DIR"] == "${OUTPUT_ROOT}/${RUN_NAME}/${RUN_NAME}/checkpoints"
    assert evaluate["TOKENIZER"] == "${MODEL_HF}"
    assert evaluate["RUN_EVAL"] == evaluate["RUN_JUDGE"] == 0


def _run_eval(tmp_path, values):
    config = _submission("eval/eval.yaml")
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    python = stub_bin / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['-m', 'pip']:\n"
        "    sys.exit(0)\n"
        "with open(os.environ['CAPTURE_EVAL'], 'a') as output:\n"
        "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    python.chmod(0o755)
    capture = tmp_path / "commands.jsonl"
    env = {
        **os.environ,
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
        "CAPTURE_EVAL": str(capture),
        "OVERLAY_DIR": str(tmp_path / "overlay"),
    }
    env.pop("AZURE_OPENAI_ENDPOINT", None)
    commands = [Template(command).substitute(values) for command in config["jobs"][0]["command"][2:]]
    result = subprocess.run(
        ["bash", "-e", "-c", "\n".join(commands)], cwd=ROOT, env=env, check=False, capture_output=True, text=True
    )
    calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
    return result, calls


@pytest.mark.unit
@pytest.mark.parametrize("entry", PUBLIC_SUBMISSIONS[:-1])
@pytest.mark.parametrize("suite", ["openasr", "bbh", "voicebench"])
def test_eval_switches_every_run_and_suite_without_editing_model_data_or_output_paths(tmp_path, entry, suite):
    evaluate = _submission("eval/eval.yaml")["env_defaults"]
    run_name = _submission(entry)["env_defaults"]["RUN_NAME"]
    values = {
        **evaluate,
        "RUN_NAME": run_name,
        "EVAL_SUITE": suite,
        "OUTPUT_ROOT": str(tmp_path / "outputs with spaces"),
        "EVAL_DATA_ROOT": str(tmp_path / "eval data"),
        "VOICEBENCH_ROOT": str(tmp_path / "voicebench data"),
        "AUDIO_ROOT": str(tmp_path / "audio"),
    }
    result, calls = _run_eval(tmp_path, values)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    command = calls[0]
    assert command[0] == f"examples/models/speech_cascades/eval_{suite}.py"
    output = f"{values['OUTPUT_ROOT']}/{run_name}"
    for argument, expected in (
        ("--hf-path", f"{output}/initial_hf"),
        ("--checkpoint", f"{output}/{run_name}/checkpoints/iter_0004000"),
        ("--tokenizer", f"{output}/initial_hf"),
        ("--output-dir", f"{output}/eval_output/{suite}_iter4000"),
        ("--max-new-tokens", "128" if suite == "openasr" else "2048"),
    ):
        assert command[command.index(argument) + 1] == expected
    data_argument = "--data-dir" if suite == "openasr" else "--manifest"
    data_path = {
        "openasr": f"{values['EVAL_DATA_ROOT']}/OpenASR/esb-datasets-test-only-sorted",
        "bbh": f"{values['EVAL_DATA_ROOT']}/BigBenchAudio/manifest.jsonl",
        "voicebench": f"{values['VOICEBENCH_ROOT']}/voicebench.jsonl",
    }[suite]
    assert command[command.index(data_argument) + 1] == data_path
    if suite != "openasr":
        assert command[command.index("--audio-root") + 1] == values["AUDIO_ROOT"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"EVAL_SUITE": "unsupported"}, "Unsupported EVAL_SUITE"),
        ({"CHECKPOINT_ITER": "bad-step"}, "CHECKPOINT_ITER must be a nonnegative decimal integer"),
    ],
)
def test_invalid_eval_selections_fail_before_invoking_any_model_runner(tmp_path, override, message):
    values = {
        **_submission("eval/eval.yaml")["env_defaults"],
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        **override,
    }
    result, calls = _run_eval(tmp_path, values)
    assert result.returncode == 2
    assert message in result.stderr
    assert calls == []


@pytest.mark.unit
def test_checkpoint_step_strings_are_decimal_and_zero_padded(tmp_path):
    values = {
        **_submission("eval/eval.yaml")["env_defaults"],
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "CHECKPOINT_ITER": "0004000",
    }
    result, calls = _run_eval(tmp_path, values)
    assert result.returncode == 0, result.stderr
    command = calls[0]
    assert command[command.index("--checkpoint") + 1].endswith("/iter_0004000")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("anchor", "consumers"),
    [
        ("output_root", {path: ["OUTPUT_ROOT"] for path in PUBLIC_SUBMISSIONS}),
        ("qwen35_model_root", {path: ["BASE_MODEL_ROOT"] for path in PUBLIC_SUBMISSIONS[:5] if "p5." not in path}),
        ("public_model_root", {"train/p5.yaml": ["BASE_MODEL_ROOT"]}),
        (
            "phi4_encoder_dir",
            {path: ["ENCODER_DIR", "NORMALIZATION_DIR"] for path in ("train/b2.yaml", "train/p5.yaml")},
        ),
        (
            "s1_encoder_dir",
            {path: ["ENCODER_DIR"] for path in ("train/p4.yaml", "train/p4-hindi-2k.yaml", "train/new-hindi-2k.yaml")},
        ),
        (
            "s1_normalization_dir",
            {
                path: ["NORMALIZATION_DIR"]
                for path in ("train/p4.yaml", "train/p4-hindi-2k.yaml", "train/new-hindi-2k.yaml")
            },
        ),
        ("evaluation_root", {"eval/eval.yaml": ["EVAL_DATA_ROOT"]}),
        ("voicebench_root", {"eval/eval.yaml": ["VOICEBENCH_ROOT"]}),
    ],
)
def test_changing_one_shared_prefix_updates_all_its_consumers(anchor, consumers):
    text = (SUBMISSIONS / "paths.yaml").read_text()
    old = yaml.safe_load(text)[".paths"][anchor]
    replacement = f"/new root/{anchor}"
    assert text.count(old) == 1
    updated = text.replace(old, replacement)
    for path, keys in consumers.items():
        values = _submission(path, paths_text=updated)["env_defaults"]
        for key in keys:
            assert values[key] == replacement, (path, key)


@pytest.mark.unit
def test_secret_scanner_metadata_does_not_reference_retired_submissions():
    baseline = json.loads((ROOT / ".github/workflows/config/.secrets.baseline").read_text())
    for filename in baseline["results"]:
        if filename.startswith("amulet/ELIFT/"):
            assert (ROOT / filename).is_file(), filename


@pytest.mark.unit
def test_relocated_shell_entrypoints_find_the_repository_root():
    for path in PROJECT.rglob("*.sh"):
        text = path.read_text()
        root = re.search(r'cd "\$\(dirname "(?:\$0|\$\{BASH_SOURCE\[0\]\})"\)(/[^"]+)"', text)
        if root is not None:
            assert (path.parent / root[1].lstrip("/")).resolve() == ROOT, path
        else:
            assert 'exec bash "$(dirname "${BASH_SOURCE[0]}")/run_speech_cascades_vllm_export.sh"' in text, path
