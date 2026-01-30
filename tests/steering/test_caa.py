import numpy as np
import torch
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from k_steering.steering.caa import CAASteering

 

class DummyHandle:
    def remove(self):
        pass


class DummyLayer:
    def __init__(self):
        self.self_attn = self
        self.mlp = self

    def register_forward_pre_hook(self, fn):
        return DummyHandle()

    def register_forward_hook(self, fn):
        return DummyHandle()


class DummyHFModel:
    def __init__(self, n_layers=2):
        self.config = SimpleNamespace(num_hidden_layers=n_layers)
        self.model = SimpleNamespace(
            layers=[DummyLayer() for _ in range(n_layers)]
        )

    def __call__(self, **kwargs):
        return None

    def generate(self, **kwargs):
        return torch.ones(1, 6, dtype=torch.long)

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
    
    def __call__(self, texts, **kwargs):
        batch = len(texts) if isinstance(texts, list) else 1
        
        class TokenizerOutput(dict):
            def to(self, device):
                return self
        
        return TokenizerOutput({
            "input_ids": torch.ones(batch, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        })
    
    def decode(self, *args, **kwargs):
        return "generated text"

@pytest.fixture
def caa(monkeypatch):
    def fake_init(self, model_name, steering_config, trainer_config, device):
        self.model_name = model_name
        self.model = DummyHFModel()
        self.tokenizer = MockTokenizer()
        # self.tokenizer.return_value = {
        #     "input_ids": torch.ones(1, 3, dtype=torch.long)
        # }
        # self.tokenizer.decode = lambda x, **k: "decoded"

        self.device = "cpu"
        self.logger = MagicMock()
        self.steering_config = SimpleNamespace(
            output_dir=None,
            train_layer=0,
            pos=-1,
        )

        self.steering_vectors = {}
        self.cache = {}
        self._is_fitted = False

    monkeypatch.setattr(
        "k_steering.steering.caa.ActivationSteering.__init__",
        fake_init,
    )

    return CAASteering("dummy-model")



def test_format_prompt_no_style(caa):
    caa.style_instructions = {"a": "STYLE: "}
    prompts = ["hello", "world"]

    out = caa.format_prompt(prompts)

    assert out == prompts


def test_format_prompt_with_style(caa):
    caa.style_instructions = {"formal": "[FORMAL] "}
    prompts = ["hi"]

    out = caa.format_prompt(prompts, style="formal")

    assert out == ["[FORMAL] hi"]



def test_apply_steering_adds_vector(caa):
    caa.steering_vectors = {
        "pos": torch.ones(4),
    }

    hidden = torch.zeros(2, 3, 4)

    out = caa._apply_steering(
        hidden_states=hidden,
        target_labels=["pos"],
        steering_strength=2.0,
    )

    # normalized ones vector = ones/sqrt(4)=0.5 each → scaled by 2 → +1
    assert torch.allclose(out, torch.ones_like(hidden))


def test_apply_steering_with_avoid(caa):
    caa.steering_vectors = {
        "a": torch.ones(4),
        "b": torch.ones(4) * 0.5,
    }

    hidden = torch.zeros(1, 1, 4)

    out = caa._apply_steering(
        hidden_states=hidden,
        target_labels=["a"],
        avoid_labels=["b"],
        steering_strength=1.0,
    )

    assert out.shape == hidden.shape


def test_apply_steering_zero_norm_returns_input(caa):
    caa.steering_vectors = {
        "zero": torch.zeros(4),
    }

    hidden = torch.randn(1, 2, 4)

    out = caa._apply_steering(
        hidden_states=hidden,
        target_labels=["zero"],
        steering_strength=10.0,
    )

    assert torch.allclose(out, hidden)



def test_build_steering_trainer_creates_vectors(caa, monkeypatch):
    caa.style_instructions = {"s1": "", "s2": ""}

    def fake_cache(style, split_name, layer_idx, pos):
        if style == "neutral":
            return np.zeros((5, 4))
        return np.ones((5, 4))

    monkeypatch.setattr(caa, "get_layer_cache", fake_cache)

    caa.build_steering_trainer()

    assert "s1" in caa.steering_vectors
    assert isinstance(caa.steering_vectors["s1"], torch.Tensor)



def test_get_layer_cache_uses_cache(caa):
    acts = torch.randn(5, 3, 4)

    caa.cache = {
        "neutral": {
            "train": {
                ("resid_pre", 0): acts
            }
        }
    }

    out = caa.get_layer_cache(
        style="neutral",
        split_name="train",
        layer_idx=0,
        pos=-1,
        use_cached=True,
    )

    assert out.shape == (5, 4)
    assert isinstance(out, np.ndarray)



def test_make_hooks_returns_handles(caa):
    acts, handles = caa._make_hooks([0, 1])

    assert isinstance(acts, dict)
    assert len(handles) == 8   # 4 hooks × 2 layers



def test_generate_with_steering_basic(caa, monkeypatch):
    caa.steering_vectors = {"x": torch.ones(4)}

    monkeypatch.setattr(
        "k_steering.steering.caa.get_transformer_layers",
        lambda model: model.model.layers,
    )

    caa.tokenizer = MockTokenizer()
    

    result = caa._generate_with_steering(
        input_prompts=["hello"],
        steering_strength=1.0,
        target_labels=["x"],
        avoid_labels=None,
        target_layers=[0],
        layer_strengths={0: 1.0},
        generation_kwargs={},
    )

    assert result["text"] == ["generated text"]
    assert result["target_layers"] == [0]



def test_fit_requires_task_or_dataset(caa):
    with pytest.raises(ValueError):
        caa.fit()


def test_fit_requires_style_instructions_for_custom_dataset(caa):
    dataset = [{"text": "a", "label": "x"}]

    caa._get_prompts_from_dataset = lambda d: ["a"]
    caa._extract_labels = lambda d: ["x"]

    with pytest.raises(ValueError):
        caa.fit(dataset=dataset)
