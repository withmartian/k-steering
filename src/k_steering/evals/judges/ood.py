import asyncio

from k_steering.evals.judges.base_judge import BaseLLMJudge
from k_steering.utils.constants import OOD_JUDGE_SYSTEM_PROMPT


class OODJudge(BaseLLMJudge):
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini"
    ):
        """
        Initialize OOD Judge class
        
        Args:
            model_name: Judge Model Name
        """
        super().__init__(model_name)
        # Change the class name
        self.judge_name = "OODJudge"
        self.task = "ood"
        self.system_prompt = OOD_JUDGE_SYSTEM_PROMPT
        
    async def evaluate_sample(
        self,
        generation: str
        
    ) -> dict:
        """
        Evaluate a single (baseline, steered) pair.
        """
        prompt = self._create_prompt(
            generation=generation
        )

        raw_output = self._run_model(
            prompt=prompt,
            mode="logprob",
        )

        return raw_output

    async def evaluate_batch(
        self,
        generations: list[str]
    ) -> dict:
        """
        Evaluate a batch and return aggregate statistics.
        """

        results =  await asyncio.gather(*[self.evaluate_sample(generation=t) for t in generations])

        return results
    
    def _create_prompt(
        self,
        generation: str,
    ) -> str:
        template_str = OOD_JUDGE_SYSTEM_PROMPT
        
        return template_str.format(generation=generation)
    
    # def _aggregate_results(self, results: List[Dict]) -> Dict:
    #     """
    #     OOD aggregation logic.
    #     """
    #     try:
    #         top = results.choices[0].logprobs.content[0].top_logprobs
    #     except IndexError:
    #         return None
    #     result = {}
    #     for el in top:
    #         try:
    #             result[int(el.token)] = float(math.exp(el.logprob))
    #         except ValueError:
    #             continue
    #     total = sum(result.values())
    #     if total < 0.25:
    #         return None
    #     score = sum(k * v for k, v in result.items()) / total
    #     return score
        
        
        