from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseEvaluator(ABC):
    """
    Benchmark-agnostic evaluation interface.
    """

    def __init__(
        self,
        name: str,
        batch_size: int = 8,
    ):
        self.name = name
        self.batch_size = batch_size

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def format_prompt(self, example: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def parse_output(self, output: str, example: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def score_prediction(
        self,
        prediction: Any,
        example: Dict[str, Any],
    ) -> float:
        pass

    def evaluate(
        self,
        *,
        model,
        tokenizer,
        generate_fn,
        generation_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        dataset = self.load_dataset()

        total_score = 0.0
        n = 0
        failures = 0

        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            prompts = [self.format_prompt(ex) for ex in batch]

            outputs = generate_fn(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                generation_config=generation_config,
            )

            for ex, out in zip(batch, outputs):
                pred = self.parse_output(out, ex)
                if pred is None:
                    failures += 1
                    continue

                total_score += self.score_prediction(pred, ex)
                n += 1

        return {
            "benchmark": self.name,
            "score": total_score / max(n, 1),
            "n_eval": n,
            "n_failures": failures,
        }
