import json
import re
from src.k_steering.evals.others.base_eval import BaseEvaluator
from datasets import load_dataset

class TinyMMLUEvaluator(BaseEvaluator):

    def __init__(self, path: str, **kwargs):
        super().__init__(name="tiny_mmlu", **kwargs)
        self.path = "tinyBenchmarks/tinyMMLU"

    def load_dataset(self, split="test"):
        dataset = load_dataset(self.path)[split]
        return dataset

    def format_prompt(self, ex):
        choices = ex["choices"]
        return (
            f"{ex['question']}\n\n"
            f"A. {choices['A']}\n"
            f"B. {choices['B']}\n"
            f"C. {choices['C']}\n"
            f"D. {choices['D']}\n\n"
            f"Answer with a single letter (A, B, C, or D)."
        )

    def parse_output(self, output, ex):
        m = re.search(r"\b([ABCD])\b", output.upper())
        return m.group(1) if m else None

    def score_prediction(self, pred, ex):
        return float(pred == ex["answer"])
