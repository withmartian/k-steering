import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from jinja2 import Template
from pydantic import BaseModel

from src.k_steering.utils.io import openai_api_call, anthropic_api_call
from src.k_steering.utils.prompt_templates import AVOID_AND_TOWARDS_EVALUATION_PROMPT_TEMPLATE_STR, AVOID_ONLY_EVALUATION_PROMPT_TEMPLATE_STR

class BaseLLMJudge(ABC):
    """
    Abstract base class for LLM-based judges.

    Subclasses should define:
    - task
    - system_prompt
    - style_descriptions
    - prompt template selection logic
    - result postprocessing logic
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.judge_name = "BaseLLMJudge"

        # Must be set by subclasses
        self.task: Optional[str] = None
        self.system_prompt: Optional[str] = None
        self.style_descriptions: Optional[Dict[str, str]] = None

    def evaluate_sample(
        self,
        baseline_text: str,
        steered_text: str,
        avoid_style: str,
        response_format: Type[BaseModel],
        target_style: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a single (baseline, steered) pair.
        """
        prompt = self._create_prompt(
            baseline_text=baseline_text,
            steered_text=steered_text,
            avoid_style=avoid_style,
            target_style=target_style,
        )

        raw_output = self._run_model(
            prompt=prompt,
            response_format=response_format,
        )

        parsed = self._parse_json_from_llm_output(raw_output)
        return self._postprocess_result(parsed)

    def evaluate_batch(
        self,
        baseline_texts: List[str],
        steered_texts: List[str],
        avoid_style: str,
        response_format: Type[BaseModel],
        target_style: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a batch and return aggregate statistics.
        """
        if len(baseline_texts) != len(steered_texts):
            raise ValueError("Baseline and steered text counts must match.")

        results = [
            self.evaluate_sample(
                baseline_text=baseline_texts[i],
                steered_text=steered_texts[i],
                avoid_style=avoid_style,
                target_style=target_style,
                response_format=response_format,
            )
            for i in range(len(baseline_texts))
        ]

        return self._aggregate_results(results)
    
    def _create_prompt(
        self,
        baseline_text: str,
        steered_text: str,
        avoid_style: str,
        target_style: Optional[str],
    ) -> str:
        template_str = self._select_prompt_template(
            target_style=target_style,
        )

        context = self._build_prompt_context(
            baseline_text=baseline_text,
            steered_text=steered_text,
            avoid_style=avoid_style,
            target_style=target_style,
        )

        return Template(template_str).render(context)
    
    
    def _select_prompt_template(
        self,
        target_style: None,
    ) -> str:
        """
        Return the correct prompt template string.
        """
        if target_style:
            return AVOID_AND_TOWARDS_EVALUATION_PROMPT_TEMPLATE_STR
        else:
            return AVOID_ONLY_EVALUATION_PROMPT_TEMPLATE_STR

    def _build_prompt_context(
        self,
        baseline_text: str,
        steered_text: str,
        avoid_style: str,
        target_style: Optional[str],
    ) -> Dict:
        """
        Default context builder. Subclasses may extend or override.
        """
        context = {
            "task": self.task,
            "baseline_text": baseline_text,
            "steered_text": steered_text,
            "avoid_style": avoid_style,
            "avoid_style_description": self.style_descriptions[avoid_style],
        }

        if target_style is not None:
            context.update({
                "target_style": target_style,
                "target_style_description": self.style_descriptions[target_style],
            })

        return context

    def _run_model(
        self,
        prompt: str,
        response_format: Type[BaseModel]=None,
        max_tokens: int = 1024,
        mode: str = "json"
    ) -> str:
        if "gpt" in self.model_name:
            api_fn = openai_api_call
        elif "claude" in self.model_name:
            api_fn = anthropic_api_call
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return api_fn(
            prompt=prompt,
            response_format=response_format,
            system_prompt=self.system_prompt,
            model=self.model_name,
            max_tokens=max_tokens,
            mode = mode
        )

    def _postprocess_result(self, parsed_output: Dict) -> Dict:
        """
        Convert raw judge JSON into the standardized output for this judge.
        """
        return {
            "steering_successful": parsed_output["steering_successful"],
            "steering_strength": parsed_output["steering_strength"],
        }

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Default aggregation logic.
        """
        n = len(results)
        if n == 0:
            return {"success_rate": 0.0, "average_strength": 0.0}

        success_count = sum(r.get("steering_successful", False) for r in results)
        total_strength = sum(r.get("steering_strength", 0) for r in results)

        return {
            "success_rate": success_count / n,
            "average_strength": total_strength / n,
        }
        
    def _parse_json_from_llm_output(self, raw_output: str) -> Dict:
        """
        Assumes model is instructed to return strict JSON.
        Override only if needed.
        """
        return json.loads(raw_output)
