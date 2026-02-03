import numpy as np
import pytest
import torch

# adjust imports to your module path
from k_steering.steering.trainer import (
    ActivationSteeringTrainer,
    MultiLabelSteeringModel,
)


class DummyTrainerConfig:
    def __init__(
        self,
        input_dim=8,
        hidden_dim=16,
        num_labels=3,
        clf_type="mlp",
        lr=1e-2,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.clf_type = clf_type
        self.lr = lr


@pytest.mark.parametrize("clf_type", ["linear", "mlp"])
def test_model_forward_shape(clf_type):
    model = MultiLabelSteeringModel(
        input_dim=8,
        hidden_dim=16,
        num_labels=4,
        clf_type=clf_type,
    )

    x = torch.randn(5, 8)
    out = model(x)

    assert out.shape == (5, 4)


def test_model_invalid_type():
    with pytest.raises(NotImplementedError):
        MultiLabelSteeringModel(8, 16, 3, clf_type="bad")


def test_trainer_initialization():
    cfg = DummyTrainerConfig()
    trainer = ActivationSteeringTrainer(cfg)

    assert trainer.classifier is not None
    assert trainer.loss_fn is not None
    assert trainer.optimizer is not None


def test_fit_changes_weights():
    cfg = DummyTrainerConfig()
    trainer = ActivationSteeringTrainer(cfg)

    X = np.random.randn(64, cfg.input_dim).astype(np.float32)
    Y = np.random.randint(0, 2, size=(64, cfg.num_labels)).astype(np.float32)

    before = [p.clone() for p in trainer.classifier.parameters()]

    trainer.fit(X, Y, epochs=2, batch_size=16)

    after = list(trainer.classifier.parameters())

    # at least one param tensor changed
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))


def test_predict_proba_range_and_shape():
    cfg = DummyTrainerConfig()
    trainer = ActivationSteeringTrainer(cfg)

    X = np.random.randn(10, cfg.input_dim).astype(np.float32)

    probs = trainer.predict_proba(X)

    assert probs.shape == (10, cfg.num_labels)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_compute_steering_loss_basic():
    cfg = DummyTrainerConfig(num_labels=4)
    trainer = ActivationSteeringTrainer(cfg)

    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )

    loss = trainer._compute_steering_loss(
        logits,
        target_idx=[2, 3],
        avoid_idx=[0],
    )

    # manual:
    # avoid = mean([1.0]) = 1.0
    # target = mean([3,4]) = 3.5
    # loss = 1.0 - 3.5 = -2.5
    assert torch.allclose(loss[0], torch.tensor(-2.5))


def test_compute_steering_loss_empty_indices():
    cfg = DummyTrainerConfig(num_labels=3)
    trainer = ActivationSteeringTrainer(cfg)

    logits = torch.randn(4, 3)

    loss = trainer._compute_steering_loss(
        logits,
        target_idx=[],
        avoid_idx=[],
    )

    assert torch.allclose(loss, torch.zeros(4, device=loss.device))


def test_steer_activations_shape_preserved():
    cfg = DummyTrainerConfig()
    trainer = ActivationSteeringTrainer(cfg)

    acts = np.random.randn(6, cfg.input_dim).astype(np.float32)

    steered = trainer.steer_activations(
        acts,
        target_idx=[0],
        avoid_idx=[1],
        alpha=0.1,
        steps=2,
    )

    assert isinstance(steered, torch.Tensor)
    assert steered.shape == acts.shape


def test_steer_activations_changes_values():
    cfg = DummyTrainerConfig()
    trainer = ActivationSteeringTrainer(cfg)

    acts = torch.randn(5, cfg.input_dim)

    steered = trainer.steer_activations(
        acts,
        target_idx=[0],
        avoid_idx=[],
        alpha=0.5,
        steps=1,
    )

    assert not torch.allclose(acts, steered)


def test_steer_accepts_numpy_and_tensor():
    cfg = DummyTrainerConfig()
    trainer = ActivationSteeringTrainer(cfg)

    acts_np = np.random.randn(3, cfg.input_dim).astype(np.float32)
    acts_t = torch.randn(3, cfg.input_dim)

    out_np = trainer.steer_activations(acts_np, target_idx=[0])
    out_t = trainer.steer_activations(acts_t, target_idx=[0])

    assert out_np.shape == acts_np.shape
    assert out_t.shape == acts_t.shape
