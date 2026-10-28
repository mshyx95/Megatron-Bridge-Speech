import json
import os

import pandas as pd
from fire import Fire


def to_json_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def load_s2t_results(results_file):
    results = {}
    for item in json.load(open(results_file, "r")):
        if "utt_id" in item and "hyp" in item:
            results[item["utt_id"]] = item["hyp"]
        elif "audio_ids" in item and "generated_texts" in item:
            results[item["audio_ids"][0]] = item["generated_texts"][0]
    return results


def write_res_jsonl(df, test_name, s2t_results, s2s_results, out_file):
    converted_res = []
    for idx, item in enumerate(df.iloc()):
        if "key" in item:
            utt_id = f"{test_name}-f{item['key']}"
        elif "question_id" in item:
            utt_id = f"{test_name}-f{item['question_id']}"
        else:
            utt_id = f"{test_name}_{idx}"
        new_item = {key: to_json_value(item[key]) for key in item.keys() if key != "audio"}
        new_item["utt_id"] = utt_id
        new_item["response"] = s2t_results.get(utt_id, "")
        if utt_id in s2s_results:
            new_item["s2s_response"] = s2s_results[utt_id]
        converted_res.append(new_item)
    with open(out_file, "w") as f:
        for item in converted_res:
            f.write(json.dumps(item) + "\n")
    return converted_res


def main(root_dir, results_dir, s2t_results_file=None):
    if s2t_results_file is None:
        s2t_results_file = os.path.join(results_dir, "generate_voice_bench.txt")
    # if not os.path.exists(s2t_results_file):
    #     rank = 8
    #     sharded_files = [open(f"{s2t_results_file}.{s}", "r") for s in range(rank)]
    #     outputs = []
    #     for i in range(rank):
    #         outputs.extend(json.load(sharded_files[i]))

    #     for file in sharded_files:
    #         file.close()

    #     for s in range(rank):
    #         os.unlink(f"{s2t_results_file}.{s}")

    #     with open(s2t_results_file, "w") as f:
    #         json.dump(outputs, f, indent=2)

    s2t_results = load_s2t_results(s2t_results_file)
    s2s_results = {}
    test_configs = [
        ("AlpacaEval", "alpacaeval_full"),
        ("CommonEval", "commoneval"),
        ("SD-QA", "sd-qa"),
        ("MMSU", "mmsu"),
        ("OpenBookQA", "openbookqa"),
        ("IFEval", "ifeval"),
        ("AdvBench", "advbench"),
        ("bbh", "bbh"),
        ("WildVoice", "wildvoice"),
    ]
    all_out_file = os.path.join(results_dir, "voicebench_results.json")
    final_output = {}
    for test_name, source_name in test_configs:
        print(f"Processing {test_name}...")
        if test_name == "SD-QA":
            df = pd.read_parquet(f"{root_dir}/{source_name}/usa-00000-of-00001.parquet")
        else:
            df = pd.read_parquet(f"{root_dir}/{source_name}/")
        out_file = os.path.join(results_dir, f"{test_name.lower()}_results.jsonl")
        converted_res = write_res_jsonl(df, test_name, s2t_results, s2s_results, out_file)
        final_output[test_name] = converted_res
    with open(all_out_file, "w") as f:
        json.dump(final_output, f)


if __name__ == "__main__":
    Fire(main)
