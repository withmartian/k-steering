
from typing import Any


class SteeringConfig:
    """Configuration for steering behavior"""

    def __init__(
        self,
        train_layer: int | None=-1,
        steering_strength: float = 1.0,
        eval_layer: int | None = None,
        steer_layers: list[int] | None = None,
        layer_strengths: dict[int, float] | None = None,
        output_dir: str | None = None,
        pos: int | None = None,
        **kwargs
    ):
        self.steering_strength = steering_strength
        self.eval_layer = eval_layer
        self.train_layer = train_layer
        self.steer_layers = steer_layers
        self.layer_strengths = layer_strengths or {}
        self.pos = pos
        self.output_dir = output_dir
        self.extra_config = kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            'steering_strength': self.steering_strength,
            'eval_layer': self.eval_layer,
            'train_layer':self.train_layer,
            'steer_layers': self.steer_layers,
            'layer_strengths': self.layer_strengths,
            'pos': self.pos,
            'output_dir': self.output_dir,
            **self.extra_config
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'SteeringConfig':
        return cls(**config_dict)
    

class TrainerConfig:
    """Configuration for training classifier"""

    def __init__(
        self,
        input_dim: int=None,
        num_labels: int=None,
        hidden_dim: int = 128,
        clf_type: str = "mlp",
        lr: float = 1e-3,
        **kwargs
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.clf_type = clf_type
        self.lr = lr
        self.extra_config = kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_labels': self.num_labels,
            'clf_type': self.clf_type,
            'lr': self.lr,
            **self.extra_config
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'TrainerConfig':
        return cls(**config_dict)
