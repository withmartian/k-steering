import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

from src.k_steering.steering.base import ActivationSteering
from src.k_steering.steering.trainer import ActivationSteeringTrainer
from src.k_steering.steering.config import SteeringConfig


class KSteering(ActivationSteering):
    """
    K-Steering: Activation steering using K-Linear Steering
    
    Inherits common functionality from ActivationSteering and implements
    K-Linear Steering based steering logic.
    
    Example:
        >>> config = SteeringConfig(steering_strength=1.5, layers=[10, 15, 20])
        >>> k_steer = KSteering("gpt2", config)
        >>> k_steer.fit(task="tones")
        >>> output = k_steer.get_steered_output("Hello, how are you?")
        >>> k_steer.save("./models", "my_steering_model")
    """
    
    def __init__(
        self, 
        model_name: str, 
        steering_config: Optional[SteeringConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize K-Linear Steering class
        
        Args:
            model_name: HuggingFace model name
            steering_config: Steering configuration
            device: Device to use
        """
        super().__init__(model_name, steering_config, device)
        self.k_clf = ActivationSteeringTrainer(SteeringConfig)
        
    def build_steering_trainer(
        self, cache: Dict[str, torch.Tensor], eval: bool = False
    ):
        """
        Build steering classifier/vectors from cached activations

        Args:
            cache: Cached hidden states
            eval: Whether to build evaluation classifier

        Returns:
            Classifier/steering vectors (subclass-specific)
        """
        if not eval:
            layer_idx = self.steering_config.layer
        else:
            layer_idx = len(self.dataset) + self.steering_config.layer
            
        X_layer = self.get_layer_cache(cache, layer_idx)
        y_layer = np.array([self.unique_labels.index(row["label"]) for row in self.dataset], dtype=np.int64)
        self.k_clf.fit(X_layer,self.get_one_hot(y_layer, len(self.unique_labels)), epochs=1, batch=64)

    def get_one_hot(self, indices: np.ndarray, num_classes: int) -> np.ndarray:
        out = np.zeros((len(indices), num_classes), dtype=np.float32)
        out[np.arange(len(indices)), indices] = 1.0
        return out
    
    def _apply_steering(
        self,
        hidden_states: torch.Tensor,
        layer_idx: List[int],
        steering_strength: float,
        avoid_idx: List[int] | None = None,
        steps: int = 1,
        step_size_decay: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply K-steering to hidden states
        
        Args:
            hidden_states: Original hidden states [batch, seq_len, hidden_dim]
            layer_idx: Current layer index
            steering_strength: Strength multiplier
            
        Returns:
            Steered hidden states
        """
        if avoid_idx is None:
            avoid_idx = []
        if isinstance(hidden_states, np.ndarray):
            acts_t = torch.as_tensor(hidden_states, dtype=torch.float32, device=self.device)
        else:
            acts_t = hidden_states.to(self.device, dtype=torch.float32)

        steered = acts_t.detach().clone()
        for step in range(steps):
            curr = steered.clone().requires_grad_(True)
            logits = self.k_clf.classifier(curr)
            loss_vec = self.k_clf._compute_steering_loss(
                logits, target_idx=layer_idx, avoid_idx=avoid_idx
            )
            loss = loss_vec.mean()
            grads = torch.autograd.grad(loss, curr, retain_graph=False)[0]
            current_alpha = steering_strength * (step_size_decay ** step)
            steered = (curr - current_alpha * grads).detach()
        return steered
    
    def _generate_with_steering(
        self,
        input_prompt: str,
        steering_strength: float,
        target_layers: Optional[List[int]],
        layer_strengths: Dict[int, float],
        generation_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate text with K-steering applied
        
        Args:
            input_prompt: Input text
            steering_strength: Global steering strength
            target_layers: Layers to steer
            layer_strengths: Per-layer strengths
            generation_kwargs: Generation parameters
            
        Returns:
            Generation results dictionary
        """
        
        # Hook to apply steering during generation
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Get effective strength for this layer
                strength = layer_strengths.get(layer_idx, steering_strength)
                
                # Apply steering
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    steered = self._apply_steering(hidden_states, layer_idx, strength)
                    return (steered,) + output[1:]
                else:
                    return self._apply_steering(output, layer_idx, strength)
            return hook
        
        # Register hooks on target layers
        for layer_idx in target_layers:
            if layer_idx < len(self.model.transformer.h):
                handle = self.model.transformer.h[layer_idx].register_forward_hook(
                    make_hook(layer_idx)
                )
                hooks.append(handle)
        
        try:
            # Generate with steering
            inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
        finally:
            # Remove hooks
            for handle in hooks:
                handle.remove()
        
        return {
            'text': generated_text,
            'input_prompt': input_prompt,
            'steering_strength': steering_strength,
            'target_layers': target_layers,
            'layer_strengths': layer_strengths
        }


        
        
        