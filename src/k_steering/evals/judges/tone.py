from typing import Dict
from k_steering.evals.judges.base_judge import BaseLLMJudge
from k_steering.utils.constants import TONE_DESCRIPTIONS, TONE_JUDGE_SYSTEM_PROMPT

class ToneJudge(BaseLLMJudge):
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini"
    ):
        """
        Initialize Tone Judge class
        
        Args:
            model_name: Judge Model Name
        """
        super().__init__(model_name)
        self.task = "tones"
        self.system_prompt = TONE_JUDGE_SYSTEM_PROMPT
        self.style_descriptions = TONE_DESCRIPTIONS
        
        