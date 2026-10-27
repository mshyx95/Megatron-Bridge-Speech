import json
import os
from pathlib import Path

import fire
import pandas as pd
from bfcl.ast_eval.ast_checker import ast_checker
from bfcl.bfcl_utils import *
from report_utils import save_task_gen_result


def load_answer(json_file):
    results = json.loads(Path(json_file).read_text())
    answer_by_category = {}
    for result in results:
        audio_id = result["audio_ids"][0]
        test_category = audio_id.rsplit("_", 1)[0]
        answer_by_category.setdefault(test_category, []).append(
            {"id": audio_id, "result": result["generated_texts"][0]}
        )
    return answer_by_category


def relevance_file_runner(model_result, prompt, test_category):
    correct_count = 0
    for i in range(len(model_result)):
        model_result_item = model_result[i]["result"]
        contain_func_call = False
        decoded_result = None
        try:
            decoded_result = default_decode_ast_prompting(model_result_item, language="Python")
            contain_func_call = True
            if is_empty_output(decoded_result):
                contain_func_call = False
        except Exception:
            contain_func_call = False
        if "irrelevance" in test_category:
            success = not contain_func_call
        else:
            success = contain_func_call
        if success:
            correct_count += 1
    accuracy = correct_count / len(model_result)
    return (accuracy, len(model_result))


def ast_file_runner(model_result, prompt, possible_answer, language, test_category, model_name):
    assert len(model_result) == len(prompt) == len(possible_answer), (
        f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."
    )
    correct_count = 0
    for i in range(len(model_result)):
        index = model_result[i]["id"]
        model_result_item = model_result[i]["result"]
        prompt_item = prompt[i]["function"]
        possible_answer_item = possible_answer[i]["ground_truth"]
        assert index == prompt[i]["id"] == possible_answer[i]["id"], (
            f"Index mismatch: {index} {prompt[i]['id']} {possible_answer[i]['id']}"
        )
        try:
            model_result_item_raw = model_result_item
            model_result_item = default_decode_ast_prompting(model_result_item, language)
        except Exception as e:
            print(f"Invalid syntax. Failed to decode AST. {str(e)}")
            print(f"Model result: {model_result_item_raw}")
            print(f"Possibel answer: {possible_answer_item}")
            continue
        decoder_output_valid = is_function_calling_format_output(model_result_item)
        if not decoder_output_valid:
            print(
                "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
            )
            continue
        checker_result = ast_checker(
            prompt_item, model_result_item, possible_answer_item, language, test_category, model_name
        )
        if checker_result["valid"]:
            correct_count += 1
    accuracy = correct_count / len(model_result)
    return (accuracy, len(model_result))


def main(hyp_json, meta_dir, output_dir=None):
    """Generate report from response and request files."""
    output_dir = Path(output_dir or Path(hyp_json).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_by_category = load_answer(hyp_json)
    model_name = "phi"
    acc_df = pd.DataFrame(columns=["category", "sub_category", "acc"])
    for test_category in answer_by_category:
        if is_chatable(test_category) or is_sql(test_category):
            continue
        if "multi-turn" in test_category:
            category = "multi-turn"
        elif "live" in test_category:
            category = "live-ast"
        elif "exec" in test_category:
            category = "non-live-exec"
        else:
            category = "non-live-ast"
        language = "Python"
        if is_java(test_category):
            language = "Java"
            continue
        if is_js(test_category):
            language = "JavaScript"
            continue
        print(f"🔍 Running test: {test_category}")
        prompt_file = os.path.join(meta_dir, f"BFCL_v3_{test_category}.json")
        prompt = load_file(prompt_file, sort_by_id=True)
        model_result = answer_by_category[test_category]
        model_result.sort(key=sort_key)
        if is_relevance_or_irrelevance(test_category):
            accuracy, total_count = relevance_file_runner(model_result, prompt, test_category)
            print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")
            new_row = pd.DataFrame([{"category": "hallucination", "sub_category": test_category, "acc": accuracy}])
            acc_df = pd.concat([acc_df, new_row], ignore_index=True)
            continue
        if is_executable(test_category):
            raise ValueError("Executable BFCL categories are outside the five-suite contract")
        possible_answer_file = os.path.join(meta_dir, "possible_answer", f"BFCL_v3_{test_category}.json")
        possible_answer = load_file(possible_answer_file, sort_by_id=True)
        if is_multi_turn(test_category):
            raise NotImplementedError("Multi-turn is not supported in this dataset yet.")
        else:
            accuracy, total_count = ast_file_runner(
                model_result, prompt, possible_answer, language, test_category, model_name
            )
            print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")
            new_row = pd.DataFrame([{"category": category, "sub_category": test_category, "acc": accuracy}])
            acc_df = pd.concat([acc_df, new_row], ignore_index=True)
    avg = acc_df.groupby("category")["acc"].mean().reset_index()
    table_path = output_dir / "bfcl_table.tsv"
    print(f"Saving pivot table to: {table_path}")
    print(avg.to_csv(sep="\t", index=True))
    avg.to_csv(table_path, sep="\t", index=True)
    acc_path = output_dir / "bfcl_acc.tsv"
    print(f"Saving accuracy to: {acc_path}")
    acc_df = acc_df.sort_values(by=["category", "sub_category"])
    print(acc_df.T.to_csv(sep="\t", index=False))
    acc_df.to_csv(acc_path, sep="\t", index=False)
    save_task_gen_result(acc_df, output_dir, "category", "acc", score="ACC")
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
