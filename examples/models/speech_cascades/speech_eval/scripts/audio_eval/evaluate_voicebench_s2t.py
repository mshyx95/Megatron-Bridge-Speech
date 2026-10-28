#!/usr/bin/env python3

import argparse
import ast
import importlib
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OFFICIAL_REPO_URL = "https://github.com/MatthewCYM/VoiceBench.git"
OFFICIAL_COMMIT = "3c3b0d3a7a956f745305eb348f5e03ce7ec73dad"
DEFAULT_AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
DEFAULT_AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
SUBSET_EVALUATORS = {
    "alpacaeval": "open",
    "commoneval": "open",
    "wildvoice": "open",
    "sd-qa": "qa",
    "mmsu": "mcq",
    "openbookqa": "mcq",
    "ifeval": "ifeval",
    "advbench": "harm",
    "bbh": "bbh",
}
JUDGE_SUBSETS = {"alpacaeval", "commoneval", "wildvoice", "sd-qa"}

OPEN_JUDGE_PROMPT = """I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model's responses based on the provided user input transcription [Instruction] and the model's output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user's query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user's question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don't contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user's question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user's query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user's instruction and models' response:
### [Instruction]: {prompt}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don't need to provide any explanations."""

QA_JUDGE_PROMPT = """### Question
{prompt}

### Reference answer
{reference}

### Candidate answer
{response}

Is the candidate answer correct based on the question and reference answer?
Please only output a single "Yes" or "No". Do not output anything else."""


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_command(command, cwd, log_file):
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    Path(log_file).write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed; see {log_file}: {' '.join(command)}")


def ensure_official_repo(repo_dir, log_dir):
    import speech_eval.voicebench

    return Path(speech_eval.voicebench.__file__).parent


def load_official_class(repo_dir, module_name, class_name):
    module = importlib.import_module(f"speech_eval.voicebench.{module_name}")
    return getattr(module, class_name)


def parse_serialized_kwargs(value):
    class NumpyArrayNormalizer(ast.NodeTransformer):
        def visit_Call(self, node):
            valid_dtype = (
                len(node.keywords) == 1
                and node.keywords[0].arg == "dtype"
                and isinstance(node.keywords[0].value, ast.Name)
                and node.keywords[0].value.id == "object"
            )
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id != "array"
                or len(node.args) != 1
                or not valid_dtype
            ):
                raise ValueError("Unsupported expression in serialized kwargs")
            return self.visit(node.args[0])

    expression = ast.parse(value, mode="eval")
    normalized = NumpyArrayNormalizer().visit(expression)
    return ast.literal_eval(ast.fix_missing_locations(normalized))


def normalize_kwargs_value(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: normalize_kwargs_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_kwargs_value(item) for item in value]
    return value


def normalize_records(records):
    normalized = []
    for record in records:
        record = dict(record)
        record["response"] = str(record.get("response", ""))
        if isinstance(record.get("kwargs"), str):
            record["kwargs"] = parse_serialized_kwargs(record["kwargs"])
        if record.get("kwargs") is not None:
            record["kwargs"] = normalize_kwargs_value(record["kwargs"])
        normalized.append(record)
    return normalized


def judge_prompt(item, subset):
    values = {
        "prompt": str(item.get("prompt", item.get("question", ""))),
        "reference": str(item.get("reference", "")),
        "response": str(item.get("response", "")),
    }
    template = QA_JUDGE_PROMPT if subset == "sd-qa" else OPEN_JUDGE_PROMPT
    return template.format(**values)


def create_judge_client(backend, endpoint, api_version):
    if backend == "azure-cli":
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(
            AzureCliCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_provider,
        )

    from openai import OpenAI

    return OpenAI()


def normalize_yes_no(value):
    match = re.fullmatch(r"\s*(yes|no)\s*[.!]?\s*", str(value), flags=re.IGNORECASE)
    return match.group(1).casefold() if match else None


def judge_scores_are_valid(scores, subset):
    if not isinstance(scores, list) or len(scores) != 3:
        return False
    if subset == "sd-qa":
        return all(normalize_yes_no(score) is not None for score in scores)
    for score in scores:
        value = str(score).strip()
        try:
            rating = float(value)
        except ValueError:
            match = re.fullmatch(r"\[\[(\d+)\]\]", value)
            if not match:
                return False
            rating = float(match.group(1))
        if not 1 <= rating <= 5:
            return False
    return True


def judge_one(client, model, prompt, retries, subset, use_max_completion_tokens=False):
    last_error = None
    for attempt in range(retries):
        try:
            request = dict(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who tries to help answer the user's question.",
                    },
                    {"role": "user", "content": prompt},
                ],
                frequency_penalty=0,
                presence_penalty=0,
                temperature=0.5,
                top_p=0.95,
                n=3,
            )
            token_parameter = "max_completion_tokens" if use_max_completion_tokens else "max_tokens"
            request[token_parameter] = 1024
            response = client.chat.completions.create(**request)
            scores = [choice.message.content.strip() for choice in response.choices]
            if not judge_scores_are_valid(scores, subset):
                raise ValueError(f"Invalid {subset} judge scores: {scores}")
            return scores
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Judge request failed after {retries} attempts: {last_error}")


def run_judge(
    records,
    subset,
    judge_dir,
    model,
    workers,
    retries,
    round_retries,
    backend,
    endpoint,
    api_version,
):
    judge_dir = Path(judge_dir)
    judge_dir.mkdir(parents=True, exist_ok=True)
    request_file = judge_dir / f"{subset}_judge_requests.jsonl"
    result_file = judge_dir / f"{subset}_judge_results.jsonl"
    requests = []
    for index, item in enumerate(records):
        requests.append(
            {
                "index": index,
                "utt_id": item.get("utt_id", index),
                "model": model,
                "prompt": judge_prompt(item, subset),
            }
        )
    write_jsonl(request_file, requests)

    completed = {}
    if result_file.exists():
        completed = {
            int(item["index"]): item
            for item in read_jsonl(result_file)
            if judge_scores_are_valid(item.get("score"), subset)
        }
    client = create_judge_client(backend, endpoint, api_version)
    pending = [request for request in requests if request["index"] not in completed]
    for round_index in range(round_retries + 1):
        if not pending:
            break
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    judge_one,
                    client,
                    model,
                    request["prompt"],
                    retries,
                    subset,
                    backend == "azure-cli",
                ): request
                for request in pending
            }
            for future in as_completed(futures):
                request = futures[future]
                try:
                    score = future.result()
                except Exception as error:
                    failed.append(request)
                    print(
                        f"Judge retry pending: subset={subset} index={request['index']} "
                        f"round={round_index + 1}/{round_retries + 1} error={error}",
                        file=sys.stderr,
                    )
                    continue
                completed[request["index"]] = {**request, "score": score}
                write_jsonl(result_file, [completed[index] for index in sorted(completed)])
        pending = failed
        if pending and round_index < round_retries:
            wait_seconds = min(2**round_index, 30)
            print(
                f"Retrying {len(pending)} incomplete {subset} judge requests after {wait_seconds}s",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    if pending:
        raise RuntimeError(
            f"{len(pending)} {subset} judge requests remain incomplete after "
            f"{round_retries + 1} rounds; no metrics were generated"
        )

    judged = []
    for index, item in enumerate(records):
        judged_item = dict(item)
        judged_item["score"] = completed[index]["score"]
        judged.append(judged_item)
    judged_file = judge_dir / f"{subset}_judged.jsonl"
    write_jsonl(judged_file, judged)
    return judged


def majority_vote(scores):
    normalized = [normalize_yes_no(score) for score in scores]
    if any(score is None for score in normalized):
        raise ValueError(f"Invalid Yes/No judge scores: {scores}")
    return max(set(normalized), key=normalized.count) == "yes"


def evaluate_qa(records):
    gpt_results = [majority_vote(item["score"]) for item in records]
    metrics = {"gpt": 100.0 * sum(gpt_results) / len(gpt_results)}
    try:
        from qa_metrics.pedant import PEDANT

        pedant = PEDANT()
        panda_results = [
            pedant.evaluate(
                [str(item["reference"]).lower()],
                item["response"].lower(),
                item["prompt"].lower(),
            )
            for item in records
        ]
        metrics["panda"] = 100.0 * sum(panda_results) / len(panda_results)
    except ImportError:
        metrics["panda"] = None
        metrics["panda_status"] = "qa_metrics is not installed"
    return metrics


def evaluate_official(records, evaluator_name, repo_dir, seed):
    class_names = {
        "open": ("open", "OpenEvaluator"),
        "mcq": ("mcq", "MCQEvaluator"),
        "ifeval": ("ifeval", "IFEvaluator"),
        "harm": ("harm", "HarmEvaluator"),
        "bbh": ("bbh", "BBHEvaluator"),
    }
    module_name, class_name = class_names[evaluator_name]
    evaluator_class = load_official_class(repo_dir, module_name, class_name)
    random.seed(seed)
    if evaluator_name == "ifeval":
        from langdetect import DetectorFactory

        DetectorFactory.seed = seed
    return evaluator_class().evaluate(records)


# The official extractor substring-matches its templates against the WHOLE
# response with `a` tried before `b`/`c`/`d`, so a reply that walks through
# "A. ... B. ... C. ..." and only then concludes is scored as "A". These
# patterns read the conclusion instead. A bare "**D:**" is NOT accepted unless
# it ends the response, because models also bold options while rejecting them.
FINAL_CHOICE_PATTERNS = (
    re.compile(
        r"(?:the\s+)?(?:correct\s+|final\s+|best\s+)?answer\s*(?:is|:)?\s*"
        r"[:\-]?\s*\**\s*\(?\s*([ABCD])\b",
        re.IGNORECASE,
    ),
    # LaTeX \boxed{X} / \boxed{\text{X}} answer format (used by reasoning models).
    re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\**\s*\(?([ABCD])\)?\s*\**\s*[\.\!]?\s*\**\s*$"),
)


def extract_final_choice(response, tail_chars=400):
    tail = response[-tail_chars:]
    for pattern in FINAL_CHOICE_PATTERNS:
        matches = pattern.findall(tail)
        if matches:
            return matches[-1].upper()
    return None


def evaluate_mcq(records, repo_dir, seed):
    """Same protocol as the official evaluator, but read the response's final answer first."""
    evaluator = load_official_class(repo_dir, "mcq", "MCQEvaluator")()
    random.seed(seed)
    official = evaluator.evaluate(records)

    random.seed(seed)
    correct = 0
    fell_back = 0
    guessed = 0
    for item in records:
        choice = extract_final_choice(item["response"])
        if choice is None:
            official_choice = evaluator.extract_answer(item["response"])
            if official_choice is not None:
                choice = str(official_choice).upper()
                fell_back += 1
        if choice is None:
            choice = random.choice(["A", "B", "C", "D"])
            guessed += 1
        if choice == str(item["reference"]).strip().upper():
            correct += 1

    total = len(records)
    return {
        "acc": 100 * correct / total,
        "fail": 100 * guessed / total,
        "acc_official": official["acc"],
        "fail_official": official["fail"],
        "official_fallback": 100 * fell_back / total,
    }


def evaluate_subset(subset, records, repo_dir, seed):
    evaluator_name = SUBSET_EVALUATORS[subset]
    if evaluator_name == "qa":
        return evaluate_qa(records)
    if evaluator_name == "mcq":
        return evaluate_mcq(records, repo_dir, seed)
    return evaluate_official(records, evaluator_name, repo_dir, seed)


def main():
    parser = argparse.ArgumentParser(description="Run the official VoiceBench evaluators.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--official-repo")
    parser.add_argument("--run-judge", action="store_true")
    parser.add_argument("--judge-backend", choices=("azure-cli", "openai"), default="azure-cli")
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--judge-endpoint", default=DEFAULT_AZURE_OPENAI_ENDPOINT)
    parser.add_argument("--judge-api-version", default=DEFAULT_AZURE_OPENAI_API_VERSION)
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--judge-retries", type=int, default=5)
    parser.add_argument("--judge-round-retries", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subsets", nargs="*", choices=sorted(SUBSET_EVALUATORS))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    metrics_dir = output_dir / "metrics"
    judge_dir = output_dir / "judge"
    log_dir = output_dir / "logs"
    official_dir = output_dir / "official"
    for directory in (metrics_dir, judge_dir, log_dir, official_dir):
        directory.mkdir(parents=True, exist_ok=True)

    requested = args.subsets or list(SUBSET_EVALUATORS)
    available = [subset for subset in requested if (results_dir / f"{subset}_results.jsonl").exists()]
    if not available:
        raise FileNotFoundError(f"No converted VoiceBench result files found in {results_dir}")

    repo_dir = Path(args.official_repo).resolve() if args.official_repo else official_dir / "VoiceBench"
    repo_dir = ensure_official_repo(repo_dir, log_dir)
    summary = {
        "official_repo": OFFICIAL_REPO_URL,
        "official_commit": OFFICIAL_COMMIT,
        "seed": args.seed,
        "subsets": {},
    }
    for subset in available:
        source_file = results_dir / f"{subset}_results.jsonl"
        records = normalize_records(read_jsonl(source_file))
        if subset in JUDGE_SUBSETS and not all("score" in item for item in records):
            if not args.run_judge:
                summary["subsets"][subset] = {
                    "status": "judge_required",
                    "samples": len(records),
                }
                continue
            records = run_judge(
                records,
                subset,
                judge_dir,
                args.judge_model,
                args.judge_workers,
                args.judge_retries,
                args.judge_round_retries,
                args.judge_backend,
                args.judge_endpoint,
                args.judge_api_version,
            )

        metrics = evaluate_subset(subset, records, repo_dir, args.seed)
        metrics["samples"] = len(records)
        metrics["status"] = "complete"
        with (metrics_dir / f"{subset}_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
            file.write("\n")
        summary["subsets"][subset] = metrics

    with (output_dir / "voicebench_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
