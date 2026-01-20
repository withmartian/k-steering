from k_steering.evals.others.base_eval import BaseEvaluator
from datasets import load_dataset
from src.k_steering.evals.judges.alpaca_judge import AlpacaJudge


class TinyAlpacaEvaluator(BaseEvaluator):

    def __init__(self, path: str, judge_fn = AlpacaJudge, **kwargs):
        super().__init__(name="tiny_alpaca", **kwargs)
        self.path = "tinyBenchmarks/tinyAlpacaEval"
        self.judge_fn = judge_fn  # external LLM or heuristic

    def load_dataset(self,split="test"):
        dataset = load_dataset(self.path)[split]
        return dataset

    def format_prompt(self, ex):
        return ex["instruction"]

    def parse_output(self, output, ex):
        return output.strip()

    def score_prediction(self, pred, ex):
        """
        judge_fn returns score in [0, 1]
        """
        return float(
            self.judge_fn(
                prediction=pred,
                reference=ex.get("output", None),
                instruction=ex["instruction"],
            )
        )
