from unittest.mock import MagicMock

import pytest

from k_steering.data.task_constants import OOD_JUDGE_SYSTEM_PROMPT
from k_steering.evals.judges.ood import OODJudge


@pytest.fixture
def judge():
    j = OODJudge(model_name="dummy-model")
    # Patch _run_model so no real LLM is called
    j._run_model = MagicMock(return_value={"logprob": -1.23})
    return j


def test_ood_judge_init(judge):
    assert judge.judge_name == "OODJudge"
    assert judge.task == "ood"
    assert judge.system_prompt == OOD_JUDGE_SYSTEM_PROMPT


def test_create_prompt(judge):
    generation = "This is a test output"
    prompt = judge._create_prompt(generation=generation)

    expected = OOD_JUDGE_SYSTEM_PROMPT.format(generation=generation)
    assert prompt == expected


@pytest.mark.asyncio
async def test_evaluate_sample_calls_run_model(judge):
    generation = "hello world"

    result = await judge.evaluate_sample(generation)

    judge._run_model.assert_called_once()
    call_kwargs = judge._run_model.call_args.kwargs

    assert call_kwargs["mode"] == "logprob"
    assert generation in call_kwargs["prompt"]
    assert result == {"logprob": -1.23}


@pytest.mark.asyncio
async def test_evaluate_batch(judge):
    generations = ["a", "b", "c"]

    results = await judge.evaluate_batch(generations)

    assert len(results) == 3
    assert results == [{"logprob": -1.23}] * 3
    assert judge._run_model.call_count == 3


@pytest.mark.asyncio
async def test_evaluate_batch_preserves_order():
    judge = OODJudge()

    def side_effect(*, prompt, mode):
        return {"prompt": prompt}

    judge._run_model = MagicMock(side_effect=side_effect)

    generations = ["first", "second", "third"]
    results = await judge.evaluate_batch(generations)

    for gen, res in zip(generations, results, strict=True):
        assert gen in res["prompt"]
