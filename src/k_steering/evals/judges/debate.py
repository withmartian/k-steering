from typing import Dict
from k_steering.evals.judges.base_judge import BaseLLMJudge
from k_steering.utils.constants import DEBATE_DESCRIPTIONS, DEBATE_JUDGE_SYSTEM_PROMPT

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
        self.task = "debates"
        self.system_prompt = DEBATE_JUDGE_SYSTEM_PROMPT
        self.style_descriptions = DEBATE_DESCRIPTIONS
        
        