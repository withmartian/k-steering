from __future__ import annotations

from typing import List, Tuple

from datasets import load_dataset

from .tasks import tones_prompts, debates_prompts


def load_task(task: str):
    if task == "tones":
        ds = load_dataset("Narmeen07/tone_agnostic_questions", split="train")
        steered_prompts = tones_prompts()
        unique_labels = sorted({t for t, _ in steered_prompts})
        dataset = []
        for row in ds:
            q_text, q_id = row["text"], row["id"]
            for lbl, sys_prompt in steered_prompts:
                dataset.append(
                    {
                        "id": f"{q_id}_{lbl}",
                        "original_question": q_text,
                        "text": f"{sys_prompt}\n{q_text}",
                        "label": lbl,
                    }
                )
        eval_prompts = list(ds["text"])
        max_samples = 10
        return dataset[:max_samples], unique_labels, eval_prompts[:max_samples]
    if task == "debates":
        ds = load_dataset("Narmeen07/debate_style_agnostic_questions", split="train")
        steered_prompts = debates_prompts()
        unique_labels = sorted({t for t, _ in steered_prompts})
        dataset = []
        for row in ds:
            q_text, q_id = row["text"], row.get("id", None) or str(hash(row["text"]))
            for lbl, sys_prompt in steered_prompts:
                dataset.append(
                    {
                        "id": f"{q_id}_{lbl}",
                        "original_question": q_text,
                        "text": f"{sys_prompt}\n{q_text}",
                        "label": lbl,
                    }
                )
        eval_prompts = list(ds["text"])
        return dataset, unique_labels, eval_prompts
    raise ValueError(f"Unknown task {task}")



