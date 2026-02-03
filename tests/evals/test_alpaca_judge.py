import json

import pandas as pd
import pytest
from pydantic import BaseModel

from k_steering.data.eval_prompt_templates import ALPACA_EVAL_PROMPT_TEMPLATE_STR
from k_steering.data.task_constants import ALPACA_JUDGE_SYSTEM_PROMPT
from k_steering.evals.judges.alpaca_judge import AlpacaJudge
from k_steering.evals.judges.base_judge import BaseLLMJudge


class DummyAlpacaResponse(BaseModel):
    is_acceptable: bool
    overall_quality: float
    coherence: float
    relevance: float
    fluency: float
    instruction_adherence: float
    factual_consistency: float


@pytest.fixture
def judge():
    return AlpacaJudge(model_name="gpt-4o-mini")


@pytest.fixture
def benchmark_df():
    return pd.DataFrame({
        "instruction": [
            "Summarize the text",
            "Translate to French",
        ],
        "output": [
            "Correct summary",
            "Traduction correcte",
        ],
    })


# -------------------------
# Tests
# -------------------------

def test_alpaca_judge_initialization(judge):
    assert isinstance(judge, BaseLLMJudge)
    assert judge.task is None
    assert judge.system_prompt == ALPACA_JUDGE_SYSTEM_PROMPT
    assert judge.style_descriptions is None
    assert judge.model_name == "gpt-4o-mini"


def test_select_prompt_template(judge):
    template = judge._select_prompt_template(target_style=None)
    assert template == ALPACA_EVAL_PROMPT_TEMPLATE_STR


def test_build_prompt_context(judge):
    ctx = judge._build_prompt_context(
        dataset_instruction="Do X",
        model_output="Model output",
        dataset_output="Reference output",
    )

    assert ctx == {
        "dataset_instruction": "Do X",
        "model_output": "Model output",
        "dataset_output": "Reference output",
    }


def test_create_prompt_contains_fields(judge):
    prompt = judge._create_prompt(
        dataset_instruction="Instruction",
        model_output="abc",
        dataset_output="cdf",
        target_style=None,
    )

    assert "Instruction" in prompt
    assert "abc" in prompt
    assert "cdf" in prompt


def test_evaluate_sample(monkeypatch, judge):
    fake_output = {
        "is_acceptable": True,
        "overall_quality": 4.0,
        "coherence": 4.5,
        "relevance": 4.0,
        "fluency": 5.0,
        "instruction_adherence": 4.5,
        "factual_consistency": 5.0,
    }

    def fake_run_model(*args, **kwargs):
        return json.dumps(fake_output)

    monkeypatch.setattr(judge, "_run_model", fake_run_model)

    result = judge.evaluate_sample(
        dataset_instruction="Summarize",
        model_output="Good summary",
        dataset_output="Reference summary",
        response_format=DummyAlpacaResponse,
    )

    assert result == fake_output


def test_evaluate_batch(monkeypatch, judge, benchmark_df):
    fake_outputs = [
        {
            "is_acceptable": True,
            "overall_quality": 4,
            "coherence": 4,
            "relevance": 4,
            "fluency": 5,
            "instruction_adherence": 4,
            "factual_consistency": 5,
        },
        {
            "is_acceptable": False,
            "overall_quality": 2,
            "coherence": 3,
            "relevance": 2,
            "fluency": 3,
            "instruction_adherence": 2,
            "factual_consistency": 3,
        },
    ]

    def fake_run_model(*args, **kwargs):
        return json.dumps(fake_outputs.pop(0))

    monkeypatch.setattr(judge, "_run_model", fake_run_model)

    result = judge.evaluate_batch(
        model_outputs=["out1", "out2"],
        benchmark_dataset=benchmark_df,
        response_format=DummyAlpacaResponse,
    )

    assert result["acceptance_rate"] == 0.5
    assert result["overall_quality"] == pytest.approx(3.0)
    assert result["fluency"] == pytest.approx(4.0)


def test_evaluate_batch_length_mismatch(judge, benchmark_df):
    with pytest.raises(ValueError, match="length must match"):
        judge.evaluate_batch(
            model_outputs=["only one"],
            benchmark_dataset=benchmark_df,
            response_format=DummyAlpacaResponse,
        )


def test_aggregate_results_empty(judge):
    result = judge._aggregate_results([])

    assert result == {
        "success_rate": 0.0,
        "average_strength": 0.0,
    }
