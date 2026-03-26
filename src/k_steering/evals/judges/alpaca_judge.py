import pandas as pd
from jinja2 import Template
from pydantic import BaseModel

from k_steering.data.eval_prompt_templates import ALPACA_EVAL_PROMPT_TEMPLATE_STR
from k_steering.data.task_constants import ALPACA_JUDGE_SYSTEM_PROMPT
from k_steering.evals.judges.base_judge import BaseLLMJudge


class AlpacaJudge(BaseLLMJudge):
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini"
    ):
        """
        Initialize Alpaca Judge class
        
        Args:
            model_name: Judge Model Name
        """
        super().__init__(model_name)
        self.task = None
        self.system_prompt = ALPACA_JUDGE_SYSTEM_PROMPT
        self.style_descriptions = None
        
    def evaluate_sample(
        self,
        dataset_instruction: str,
        model_output: str,
        dataset_output: str,
        response_format: type[BaseModel],
    ) -> dict:
        """
        Evaluate a single (baseline, steered) pair.
        """
        prompt = self._create_prompt(
            dataset_instruction = dataset_instruction,
            model_output= model_output,
            dataset_output= dataset_output,
        )

        raw_output = self._run_model(
            prompt=prompt,
            response_format=response_format,
        )

        parsed = self._parse_json_from_llm_output(raw_output)
        return parsed
    

    def evaluate_batch(
        self,
        model_outputs: list[str],
        benchmark_dataset: pd.DataFrame,
        response_format: type[BaseModel]
    ) -> dict:
        """
        Evaluate a batch and return aggregate statistics.
        """
        if len(model_outputs) != benchmark_dataset.shape[0]:
            raise ValueError("Model output and benchmark dataset length must match.")

        results = [
            self.evaluate_sample(
                dataset_instruction=benchmark_dataset['instruction'][i],
                model_output=model_outputs[i],
                dataset_output=benchmark_dataset['output'][i],
                response_format=response_format,
            )
            for i in range(len(model_outputs))
        ]

        return self._aggregate_results(results)
    
    def _create_prompt(
        self,
        dataset_instruction: str,
        model_output: str,
        dataset_output: str,
        target_style: str = None,
    ) -> str:
        template_str = self._select_prompt_template(
            target_style=target_style,
        )

        context = self._build_prompt_context(
            dataset_instruction = dataset_instruction,
            model_output= model_output,
            dataset_output= dataset_output,
        )

        return Template(template_str).render(context)
        
    def _select_prompt_template(
        self,
        target_style: None,
    ) -> str:
        """
        Return the correct prompt template string.
        """
        return ALPACA_EVAL_PROMPT_TEMPLATE_STR
    
    def _build_prompt_context(
        self,
        dataset_instruction: str,
        model_output: str,
        dataset_output: str,
    ) -> dict:
        """
        Default context builder. Subclasses may extend or override.
        """
        context = {
            "dataset_instruction": dataset_instruction,
            "model_output": model_output,
            "dataset_output": dataset_output,
        }

        return context
    
    def _aggregate_results(self, results: list[dict]) -> dict:
        """
        Default aggregation logic.
        """
        n = len(results)
        if n == 0:
            return {"success_rate": 0.0, "average_strength": 0.0}

        is_acceptable = sum(r.get("is_acceptable", False) for r in results)
        overall_quality = sum(r.get("overall_quality", 0) for r in results)
        coherence = sum(r.get("coherence", 0) for r in results)
        relevance = sum(r.get("relevance", 0) for r in results)
        fluency = sum(r.get("fluency", 0) for r in results)
        instruction_adherence = sum(r.get("instruction_adherence", 0) for r in results)
        factual_consistency = sum(r.get("factual_consistency", 0) for r in results)

        return {
            "acceptance_rate": is_acceptable / n,
            "overall_quality": overall_quality / n,
            "coherence": coherence / n,
            "relevance": relevance / n,
            "fluency": fluency / n,
            "instruction_adherence": instruction_adherence / n,
            "factual_consistency": factual_consistency / n,
        }
