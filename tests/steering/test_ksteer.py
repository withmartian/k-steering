from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from k_steering.steering.k_steer import KSteering


@pytest.fixture
def mock_configs():
    steering_config = SimpleNamespace(
        output_dir=None,
        train_layer=0,
        eval_layer=1,
        steer_layers=[0, 1],
    )

    trainer_config = SimpleNamespace(
        input_dim=None,
        num_labels=None,
    )

    return steering_config, trainer_config


@pytest.fixture
def ksteer(monkeypatch, mock_configs):
    steering_config, trainer_config = mock_configs

    # patch parent __init__ to avoid HF loading
    def fake_parent_init(self, *args, **kwargs):
        self.model = MagicMock()
        self.steering_config = steering_config
        self.trainer_config = trainer_config
        self.model.config.num_hidden_layers = 2
        self.tokenizer = MagicMock()
        self.device = "cpu"
        self.logger = MagicMock()

    monkeypatch.setattr(
        "k_steering.steering.k_steer.ActivationSteering.__init__",
        fake_parent_init,
    )

    ks = KSteering("fake-model", steering_config, trainer_config, device="cpu")

    ks.dataset = [{"label": "a"}, {"label": "b"}]
    ks.unique_labels = ["a", "b"]
    ks.tone2idx = {"a": 0, "b": 1}

    return ks



def test_get_one_hot_basic(ksteer):
    idx = np.array([0, 1, 0])
    out = ksteer.get_one_hot(idx, num_classes=2)

    assert out.shape == (3, 2)
    assert np.allclose(out[0], [1, 0])
    assert np.allclose(out[1], [0, 1])



def test_build_steering_trainer(monkeypatch, ksteer):
    X = np.random.randn(2, 4)

    ksteer.get_layer_cache = MagicMock(return_value=X)

    fake_trainer = MagicMock()
    monkeypatch.setattr(
        "k_steering.steering.k_steer.ActivationSteeringTrainer",
        lambda cfg: fake_trainer,
    )

    ksteer.build_steering_trainer(eval=False)

    assert ksteer.trainer_config.input_dim == 4
    assert ksteer.trainer_config.num_labels == 2
    fake_trainer.fit.assert_called_once()



def test_apply_steering_runs(monkeypatch, ksteer):
    B, S, D = 2, 3, 4
    hidden = torch.randn(B, S, D)
    
    W = torch.randn(D, 2, requires_grad=False)

    def classifier(x):
        return x @ W

    fake_clf = MagicMock()
    fake_clf.classifier = classifier

    def fake_loss(logits, target_idx, avoid_idx):
        return logits.sum(dim=1)

    fake_clf._compute_steering_loss = fake_loss
    ksteer.k_clf = fake_clf

    out = ksteer._apply_steering(
        hidden_states=hidden,
        target_idx=[0],
        steering_strength=0.1,
        steps=1,
    )

    assert isinstance(out, torch.Tensor)
    assert out.shape == hidden.shape


def test_apply_steering_numpy_input(ksteer):
    arr = np.random.randn(1, 2, 3)
    
    D = arr.shape[-1]
    
    W = torch.randn(D, 2, requires_grad=False)

    def classifier(x):
        return x @ W

    fake_clf = MagicMock()
    fake_clf.classifier = classifier
    def fake_loss(logits, target_idx, avoid_idx):
        return logits.sum(dim=1)

    fake_clf._compute_steering_loss = fake_loss

    ksteer.k_clf = fake_clf

    out = ksteer._apply_steering(arr, target_idx=[0], steering_strength=0.1)
    assert isinstance(out, torch.Tensor)



def test_generate_with_steering(monkeypatch, ksteer):

    # fake layers + hooks
    handle = MagicMock()
    layer = MagicMock()
    layer.register_forward_hook.return_value = handle

    monkeypatch.setattr(
        "k_steering.steering.k_steer.get_transformer_layers",
        lambda model: [layer, layer],
    )

    # tokenizer mock with .to()
    class FakeBatch(dict):
        def to(self, device):
            return self

    ksteer.tokenizer = MagicMock()
    ksteer.tokenizer.return_value = FakeBatch(
        {"input_ids": torch.ones(1, 3, dtype=torch.long)}
    )
    ksteer.tokenizer.decode.return_value = "gen"

    ksteer.model.generate = MagicMock(
        return_value=torch.ones(1, 5, dtype=torch.long)
    )

    out = ksteer._generate_with_steering(
        input_prompts=["hi"],
        steering_strength=1.0,
        target_labels=["a"],
        avoid_labels=None,
        target_layers=[0],
        layer_strengths={0: 1.0},
        generation_kwargs={},
    )

    assert out["text"] == ["gen"]
    handle.remove.assert_called_once()



@pytest.mark.asyncio
async def test_sweep_alpha(monkeypatch, ksteer):

    ksteer._get_prompts_from_dataset = MagicMock(return_value=["p1"])
    ksteer._prepare_generation_kwargs = MagicMock(return_value={})

    ksteer._generate_with_steering = MagicMock(return_value=["text"])

    async def fake_is_ood(gens, judge):
        return False

    async def fake_calibrate(fn):
        return 3.14

    monkeypatch.setattr(
        "k_steering.steering.k_steer.is_ood",
        fake_is_ood,
    )
    monkeypatch.setattr(
        "k_steering.steering.k_steer.calibrate_alpha_ood_only",
        fake_calibrate,
    )

    judge = MagicMock()

    res = await ksteer.sweep_alpha(judge, target_labels=["a"])

    assert res == {0: 3.14, 1: 3.14}
