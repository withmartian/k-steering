from k_steering.data.task_constants import (
    DEBATE_DESCRIPTIONS,
    DEBATE_JUDGE_SYSTEM_PROMPT,
)
from k_steering.evals.judges.base_judge import BaseLLMJudge


class DebateJudge(BaseLLMJudge):
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini"
    ):
        """
        Initialize Debate Judge class
        
        Args:
            model_name: Judge Model Name
        """
        super().__init__(model_name)
        self.judge_name = "DebateJudge"
        self.task = "debates"
        self.system_prompt = DEBATE_JUDGE_SYSTEM_PROMPT
        self.style_descriptions = DEBATE_DESCRIPTIONS
        
        