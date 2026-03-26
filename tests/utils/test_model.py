import pytest

from k_steering.utils.model import get_transformer_layers

# -------------------------
# Mock model definitions
# -------------------------

class MockLlamaInner:
    def __init__(self):
        self.layers = ["layer_0", "layer_1", "layer_2"]


class MockLlamaModel:
    def __init__(self):
        self.model = MockLlamaInner()


class MockGPTInner:
    def __init__(self):
        self.h = ["block_0", "block_1"]


class MockGPTModel:
    def __init__(self):
        self.transformer = MockGPTInner()


class UnsupportedModel:
    pass


# -------------------------
# Tests
# -------------------------

def test_llama_style_model():
    model = MockLlamaModel()

    layers = get_transformer_layers(model)

    assert layers == ["layer_0", "layer_1", "layer_2"]
    assert isinstance(layers, list)


def test_gpt_style_model():
    model = MockGPTModel()

    layers = get_transformer_layers(model)

    assert layers == ["block_0", "block_1"]
    assert isinstance(layers, list)


def test_unsupported_model_raises():
    model = UnsupportedModel()

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        get_transformer_layers(model)
