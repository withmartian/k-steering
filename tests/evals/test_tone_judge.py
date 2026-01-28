import json
import pytest
from pydantic import BaseModel
from jinja2 import Template
from k_steering.evals.judges.tone import ToneJudge
from k_steering.evals.judges.base_judge import BaseLLMJudge
from k_steering.data.task_constants import (
    TONE_JUDGE_SYSTEM_PROMPT,
    TONE_DESCRIPTIONS,
)
from k_steering.data.eval_prompt_templates import (
    AVOID_ONLY_EVALUATION_PROMPT_TEMPLATE_STR,
)

class DummyResponse(BaseModel):
    steering_successful: bool
    steering_strength: float


@pytest.fixture
def judge():
    return ToneJudge(model_name="gpt-4o-mini")



def test_debate_judge_initialization(judge):
    assert isinstance(judge, BaseLLMJudge)
    assert judge.judge_name == "ToneJudge"

    assert judge.task == "tones"
    assert judge.system_prompt == TONE_JUDGE_SYSTEM_PROMPT
    assert judge.style_descriptions == TONE_DESCRIPTIONS
    assert judge.model_name == "gpt-4o-mini"


def test_debate_judge_has_required_styles(judge):
    # Ensures prompt context won't KeyError later
    assert isinstance(judge.style_descriptions, dict)
    assert len(judge.style_descriptions) > 0

    for style, desc in judge.style_descriptions.items():
        assert isinstance(style, str)
        assert isinstance(desc, str)


def test_debate_judge_evaluate_sample(monkeypatch, judge):
    fake_output = {
        "steering_successful": True,
        "steering_strength": 0.6,
    }

    def fake_run_model(*args, **kwargs):
        return json.dumps(fake_output)

    monkeypatch.setattr(judge, "_run_model", fake_run_model)

    result = judge.evaluate_sample(
        baseline_text="Original argument",
        steered_text="Rewritten argument",
        avoid_style=list(judge.style_descriptions.keys())[0],
        response_format=DummyResponse,
        target_style=None,
    )

    assert result["steering_successful"] is True
    assert result["steering_strength"] == 0.6


def test_debate_judge_prompt_contains_task_and_styles(judge):
    avoid_style = list(judge.style_descriptions.keys())[0]

    prompt = judge._create_prompt(
        baseline_text="Baseline debate text",
        steered_text="Steered debate text",
        avoid_style=avoid_style,
        target_style=None,
    )
    
    assert "Baseline debate text" in prompt
    assert "Steered debate text" in prompt
    assert avoid_style in prompt