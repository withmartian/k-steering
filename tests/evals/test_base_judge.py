import json
import pytest
from pydantic import BaseModel
from jinja2 import Template
from k_steering.evals.judges.base_judge import BaseLLMJudge
from k_steering.data.eval_prompt_templates import (
    AVOID_AND_TOWARDS_EVALUATION_PROMPT_TEMPLATE_STR,
    AVOID_ONLY_EVALUATION_PROMPT_TEMPLATE_STR,
)


class DummyResponse(BaseModel):
    steering_successful: bool
    steering_strength: float


class TestJudge(BaseLLMJudge):
    def __init__(self):
        super().__init__(model_name="gpt-4o-mini")
        self.task = "test_task"
        self.system_prompt = "Test system prompt"
        self.style_descriptions = {
            "formal": "Formal writing style",
            "casual": "Casual writing style",
        }


@pytest.fixture
def judge():
    return TestJudge()


def test_create_prompt_avoid_only(judge):
    prompt = judge._create_prompt(
        baseline_text="hello",
        steered_text="hi",
        avoid_style="formal",
        target_style=None,
    )
    context = {
            "task": judge.task,
            "baseline_text": "hello",
            "steered_text": "hi",
            "avoid_style": "formal",
            "avoid_style_description": judge.style_descriptions["formal"],
        }
    prompt_template = AVOID_ONLY_EVALUATION_PROMPT_TEMPLATE_STR
    test_prompt = Template(prompt_template).render(context)

    assert prompt == test_prompt


def test_create_prompt_avoid_and_target(judge):
    prompt = judge._create_prompt(
        baseline_text="hello",
        steered_text="hi",
        avoid_style="formal",
        target_style="casual",
    )
    
    context = {
            "task": judge.task,
            "baseline_text": "hello",
            "steered_text": "hi",
            "avoid_style": "formal",
            "avoid_style_description": judge.style_descriptions["formal"],
            "target_style": "casual",
            "target_style_description": judge.style_descriptions["casual"],
        }
    prompt_template = AVOID_AND_TOWARDS_EVALUATION_PROMPT_TEMPLATE_STR
    test_prompt = Template(prompt_template).render(context)
    
    assert prompt == test_prompt



def test_build_prompt_context(judge):
    ctx = judge._build_prompt_context(
        baseline_text="a",
        steered_text="b",
        avoid_style="formal",
        target_style="casual",
    )

    assert ctx["task"] == "test_task"
    assert ctx["avoid_style"] == "formal"
    assert ctx["target_style"] == "casual"
    assert "avoid_style_description" in ctx
    assert "target_style_description" in ctx


def test_evaluate_sample(monkeypatch, judge):
    fake_output = {
        "steering_successful": True,
        "steering_strength": 0.75,
    }

    def fake_run_model(*args, **kwargs):
        return json.dumps(fake_output)

    monkeypatch.setattr(judge, "_run_model", fake_run_model)

    result = judge.evaluate_sample(
        baseline_text="baseline",
        steered_text="steered",
        avoid_style="formal",
        target_style="casual",
        response_format=DummyResponse,
    )

    assert result == {
        "steering_successful": True,
        "steering_strength": 0.75,
    }


def test_evaluate_batch_success(judge, monkeypatch):
    fake_outputs = [
        {"steering_successful": True, "steering_strength": 1.0},
        {"steering_successful": False, "steering_strength": 0.5},
    ]

    def fake_run_model(*args, **kwargs):
        return json.dumps(fake_outputs.pop(0))

    monkeypatch.setattr(judge, "_run_model", fake_run_model)

    result = judge.evaluate_batch(
        baseline_texts=["a", "b"],
        steered_texts=["c", "d"],
        avoid_style="formal",
        target_style=None,
        response_format=DummyResponse,
    )

    assert result["success_rate"] == 0.5
    assert result["average_strength"] == pytest.approx(0.75)


def test_evaluate_batch_length_mismatch(judge):
    with pytest.raises(ValueError, match="must match"):
        judge.evaluate_batch(
            baseline_texts=["a"],
            steered_texts=["b", "c"],
            avoid_style="formal",
            response_format=DummyResponse,
        )


def test_aggregate_results_empty(judge):
    result = judge._aggregate_results([])

    assert result == {
        "success_rate": 0.0,
        "average_strength": 0.0,
    }


def test_parse_json_from_llm_output(judge):
    raw = '{"steering_successful": true, "steering_strength": 0.9}'
    parsed = judge._parse_json_from_llm_output(raw)

    assert parsed["steering_successful"] is True
    assert parsed["steering_strength"] == 0.9
