
from typing import Optional, Dict, Any, List


class SteeringConfig:
    """Configuration for steering behavior"""

    def __init__(
        self,
        steering_strength: float = 1.0,
        eval_layers: Optional[List[int]] = None,
        layer_strengths: Optional[Dict[int, float]] = None,
        **kwargs
    ):
        self.steering_strength = steering_strength
        self.eval_layers = eval_layers
        self.layer_strengths = layer_strengths or {}
        self.extra_config = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'steering_strength': self.steering_strength,
            'eval_layers': self.eval_layers,
            'layer_strengths': self.layer_strengths,
            **self.extra_config
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SteeringConfig':
        return cls(**config_dict)
