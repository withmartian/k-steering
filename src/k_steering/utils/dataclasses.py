

from pydantic import BaseModel


class StyleScore(BaseModel):
    avoid_style: str
    avoid_style_score: int
    target_style: str | None
    target_style_score: int | None
    
class AvoidBool(BaseModel):
    avoid_style: str
    avoid_style_bool: bool

class TargetBool(BaseModel):
    target_style: str
    target_style_bool: bool
    
    
class LLMJudgeAvoidOnly(BaseModel):
    baseline_scores: StyleScore
    steered_scores: StyleScore
    moved_away: AvoidBool
    steering_successful: bool
    steering_strength: int
    is_steered_text_coherent: bool
    explanation: str


class LLMJudgeAvoidTowards(BaseModel):
    baseline_scores: StyleScore
    steered_scores: StyleScore
    moved_away: AvoidBool
    moved_towards: TargetBool
    steering_successful: bool
    steering_strength: int
    is_steered_text_coherent: bool
    explanation: str
