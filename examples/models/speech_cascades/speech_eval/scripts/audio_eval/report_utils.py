import json


def save_task_gen_result(df, output_dir, by_col, score_col, exclude=None, score="LLM"):
    """Save task result."""
    exclude = exclude or []
    tasks = [t for t in df[by_col].unique() if t not in exclude]
    for task in tasks:
        task_df = df[df[by_col] == task]
        task_output_path = output_dir / f"generate_{task}.txt"
        task_result = task_df.to_dict(orient="records")
        avg_score = task_df[score_col].mean()
        task_result.append({"final_score": f"{score}: {avg_score:.2f}"})
        print(f"Saving [{task}]: {task_output_path}")
        with task_output_path.open("w") as f:
            json.dump(task_result, f, indent=4)
