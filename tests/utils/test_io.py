import math
from types import SimpleNamespace

import pytest

from k_steering.utils.io import anthropic_api_call, openai_api_call


class MockLogProb:
    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob


class MockChoiceLogprobs:
    def __init__(self, top_logprobs):
        self.content = [
            SimpleNamespace(top_logprobs=top_logprobs)
        ]


class MockChoice:
    def __init__(self, logprobs=None, content=None):
        self.logprobs = logprobs
        self.message = SimpleNamespace(content=content)


class MockCompletion:
    def __init__(self, choices):
        self.choices = choices


class MockOpenAIClient:
    def __init__(self, completion):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: completion
            )
        )


class MockAnthropicClient:
    def __init__(self, parsed_output):
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(
                parse=lambda **kwargs: SimpleNamespace(
                    parsed_output=parsed_output
                )
            )
        )

class DummySchema:
    @classmethod
    def model_json_schema(cls):
        return {"type": "object"}

    @classmethod
    def model_validate_json(cls, data):
        return {"parsed": data}

    @classmethod
    def model_validate(cls, data):
        return {"parsed": data}


def test_openai_logprob_mode_success(monkeypatch):
    top_logprobs = [
        MockLogProb("0", math.log(0.7)),
        MockLogProb("1", math.log(0.3)),
    ]

    completion = MockCompletion(
        choices=[
            MockChoice(
                logprobs=MockChoiceLogprobs(top_logprobs)
            )
        ]
    )

    monkeypatch.setattr(
        "k_steering.utils.io.OpenAI",
        lambda api_key=None: MockOpenAIClient(completion),
    )

    score = openai_api_call("test", mode="logprob")

    assert pytest.approx(score, rel=1e-3) == 0.3
    
def test_openai_logprob_mode_low_total(monkeypatch):
    top_logprobs = [
        MockLogProb("0", math.log(0.1)),
    ]

    completion = MockCompletion(
        choices=[
            MockChoice(
                logprobs=MockChoiceLogprobs(top_logprobs)
            )
        ]
    )

    monkeypatch.setattr(
        "k_steering.utils.io.OpenAI",
        lambda api_key=None: MockOpenAIClient(completion),
    )

    assert openai_api_call("test", mode="logprob") is None
    
def test_openai_logprob_mode_malformed(monkeypatch):
    completion = MockCompletion(
        choices=[MockChoice(logprobs=None)]
    )

    monkeypatch.setattr(
        "k_steering.utils.io.OpenAI",
        lambda api_key=None: MockOpenAIClient(completion),
    )

    assert openai_api_call("test", mode="logprob") is None

def test_openai_json_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "Mock-key")
    completion = MockCompletion(
        choices=[
            MockChoice(content='{"answer": 42}')
        ]
    )

    monkeypatch.setattr(
        "k_steering.utils.io.OpenAI",
        lambda api_key=None: MockOpenAIClient(completion),
    )

    result = openai_api_call(
        "test",
        mode="json",
        response_format=DummySchema,
    )

    assert result == {"parsed": '{"answer": 42}'}


def test_openai_json_mode_missing_schema(monkeypatch):
    with pytest.raises(ValueError, match="response_format"):
        monkeypatch.setenv("OPENAI_API_KEY", "Mock-key")
        openai_api_call("test", mode="json")

def test_anthropic_api_call(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "Mock-key")

    monkeypatch.setattr(
        "k_steering.utils.io.Anthropic",
        lambda api_key=None: MockAnthropicClient(
            parsed_output={"foo": "bar"}
        ),
    )

    result = anthropic_api_call(
        "test",
        response_format=DummySchema,
    )

    assert result == {"parsed": {"foo": "bar"}}
