import pytest
import torch
from unittest.mock import MagicMock
from types import SimpleNamespace

import torch
from k_steering.steering.base import ActivationSteering

class DummySteering(ActivationSteering):
    def _load_task(self, task_name: str, max_samples: int = None):
        dataset = [
            {"prompt": "Hello", "label": "A"},
            {"prompt": "Hi", "label": "B"},
        ]
        return dataset, ["A", "B"], ["Eval prompt"]

    def build_steering_trainer(self, eval: bool = False):
        self.k_clf = {"dummy": True}
        self.steering_vectors = {0: torch.ones(4)}

    def _apply_steering(self, hidden_states, layer_idx, steering_strength):
        return hidden_states

class FakeBatch(dict):
    def to(self, device):
        # Return self but ensure we're still a proper dict
        return self
    
    def __getitem__(self, key):
        # Make sure we return the actual tensor, not a mock
        return dict.__getitem__(self, key)
 


@pytest.fixture
def mock_model_and_tokenizer(monkeypatch):
    # ---- Tokenizer ----
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
    
    tokenizer = MockTokenizer()

    # ---- Model ----
    class MockModel:
        def __init__(self):
            self.config = SimpleNamespace(num_hidden_layers=2)
            self.generate = MagicMock(return_value=torch.ones(1, 6, dtype=torch.long))
        
        def eval(self):
            return self
        
        def __call__(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            
            if input_ids is None:
                raise ValueError(f"input_ids not found! kwargs: {kwargs}")
            
            B, T = input_ids.shape
            
            num_layers = 2
            hidden_states = tuple(torch.randn(B, T, 4) for _ in range(num_layers + 1))

            return SimpleNamespace(
                logits=torch.randn(B, T, 1000),
                hidden_states=hidden_states
            )
    
    model = MockModel()

    monkeypatch.setattr(
        "k_steering.steering.base.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "k_steering.steering.base.AutoModelForCausalLM.from_pretrained",
        lambda *_args, **_kwargs: model,
    )

    return model, tokenizer


def test_init_cpu_device(mock_model_and_tokenizer):

    steering = DummySteering("dummy-model", device="cpu")

    assert steering.device == "cpu"
    assert steering.model is not None
    assert steering.tokenizer is not None
    assert steering._is_fitted is False
    
def test_str_and_repr(mock_model_and_tokenizer):
    
    s = DummySteering("dummy")

    assert "Not Fitted" in str(s)
    assert "DummySteering(" in repr(s)


def test_extract_labels(mock_model_and_tokenizer):

    s = DummySteering("dummy")

    dataset = [
        {"label": "x"},
        {"label": "y"},
        {"label": "x"},
    ]

    labels = s._extract_labels(dataset)
    assert set(labels) == {"x", "y"}


def test_get_hidden_cache_shapes(mock_model_and_tokenizer):


    s = DummySteering("dummy")

    prompts = {"train": ["hello", "world"]}
    cache = s.get_hidden_cache(prompts, batch_size=1)

    assert "train" in cache
    assert 0 in cache["train"]
    assert "attention_mask" in cache["train"]

    hidden = cache["train"][0]
    attn = cache["train"]["attention_mask"]

    assert hidden.ndim == 3
    assert attn.ndim == 2

def test_get_layer_cache(mock_model_and_tokenizer):

    s = DummySteering("dummy")

    prompts = {"train": ["a", "b"]}
    vecs = s.get_layer_cache(
        split_name="train",
        layer_idx=0,
        prompts=prompts,
        batch_size=1,
        use_cached=False,
    )

    assert vecs.shape[0] == 2

def test_fit_sets_state(mock_model_and_tokenizer):

    s = DummySteering("dummy")
    s.fit(task="dummy-task")

    assert s._is_fitted is True
    assert s.cache is not None
    assert s.k_clf is not None


def test_generate_without_fit_raises(mock_model_and_tokenizer):

    s = DummySteering("dummy")

    with pytest.raises(RuntimeError):
        s.get_steered_output(["hello"])


# def test_save_and_load(tmp_path, mock_model_and_tokenizer):

#     s = DummySteering("dummy")
#     s.fit(task="dummy")

#     s.save(tmp_path, "test_model")

#     loaded = DummySteering.load(tmp_path, "test_model")

#     assert loaded.model_name == "dummy"
#     assert loaded._is_fitted is True
#     assert loaded.unique_labels == ["A", "B"]
    
def test_prepare_generation_kwargs(mock_model_and_tokenizer):

    s = DummySteering("dummy")

    kwargs = s._prepare_generation_kwargs(max_new_tokens=50)

    assert kwargs["max_new_tokens"] == 50
    assert "pad_token_id" in kwargs

