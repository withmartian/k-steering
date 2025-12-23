import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

from src.k_steering.steering.base import ActivationSteering
from src.k_steering.steering.trainer import ActivationSteeringTrainer
from src.k_steering.steering.config import SteeringConfig, TrainerConfig
from src.k_steering.utils.data import load_task
from src.k_steering.utils.model import get_transformer_layers


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
        trainer_config: Optional[TrainerConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize K-Linear Steering class
        
        Args:
            model_name: HuggingFace model name
            steering_config: Steering configuration
            device: Device to use
        """
        super().__init__(model_name, steering_config,trainer_config, device)
        
    def build_steering_trainer(
        self, eval: bool = False
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
            split_name = "train"
            layer_idx = self.steering_config.train_layer
        else:
            split_name = "eval"
            layer_idx = self.steering_config.eval_layer
        
        
        
        X_layer = self.get_layer_cache(split_name, layer_idx)
        y_layer = np.array([self.unique_labels.index(row["label"]) for row in self.dataset], dtype=np.int64)
        self.trainer_config.input_dim=X_layer.shape[1]
        self.trainer_config.num_labels=len(self.unique_labels)
        self.k_clf = ActivationSteeringTrainer(self.trainer_config)
        
        self.k_clf.fit(X_layer,self.get_one_hot(y_layer, len(self.unique_labels)), epochs=1, batch_size=64)

    def get_one_hot(self, indices: np.ndarray, num_classes: int) -> np.ndarray:
        out = np.zeros((len(indices), num_classes), dtype=np.float32)
        out[np.arange(len(indices)), indices] = 1.0
        return out
    
    def _apply_steering(
        self,
        hidden_states: torch.Tensor,
        target_idx: List[int],
        steering_strength: float,
        rest: Optional[torch.Tensor] = None,
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
            # acts_t = torch.as_tensor(hidden_states, dtype=torch.float32, device=self.device)
            acts_t = torch.as_tensor(hidden_states, device=self.device)
        else:
            # acts_t = hidden_states.to(self.device, dtype=torch.float32)
            acts_t = hidden_states.to(self.device)
            
        B, S, D = acts_t.shape

        h_current = acts_t.reshape(-1, D).float()
        for step in range(steps):
            h_step = h_current.clone()
            h_step.requires_grad_(True)
            with torch.enable_grad():
                logits = self.k_clf.classifier(h_step)
                logits = logits.view(B, S, -1).mean(dim=1)
                loss_vec = self.k_clf._compute_steering_loss(
                    logits, target_idx=target_idx, avoid_idx=avoid_idx
                )
                if loss_vec.numel() > 0:
                    grad = torch.autograd.grad(
                        outputs=loss_vec,
                        inputs=h_step,
                        grad_outputs=torch.ones_like(loss_vec),
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    current_alpha = steering_strength * (step_size_decay ** step)
                    grad = grad.view(B * S, D)
                    h_current = (h_step - current_alpha * grad).detach()
                else:
                    h_current = h_step.detach()
        h_new = h_current.reshape(B, S, D).to(acts_t.dtype)
        if rest is None:
            return h_new
        return (h_new,) + rest
    
    def _generate_with_steering(
        self,
        input_prompts: List[str],
        steering_strength: float,
        target_labels: List[str],
        avoid_labels: Optional[List[str]],
        target_layers: Optional[List[int]],
        layer_strengths: Dict[int, float],
        generation_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate text with K-steering applied
        
        Args:
            input_prompts: Input text
            steering_strength: Global steering strength
            target_layers: Layers to steer
            layer_strengths: Per-layer strengths
            generation_kwargs: Generation parameters
            
        Returns:
            Generation results dictionary
        """
        
        # Hook to apply steering during generation
        hooks = []
        
        def make_hook(layer_idx, target_idx, avoid_idx):
            def hook(module, input, output):
                # Get effective strength for this layer
                strength = layer_strengths.get(layer_idx, steering_strength)
                
                # Apply steering
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest = output[1:]
                    steered = self._apply_steering(hidden_states=hidden_states,
                                                   target_idx=target_idx,
                                                   avoid_idx=avoid_idx,
                                                   steering_strength=strength,
                                                   rest=rest)
                    return (steered,) + output[1:]
                else:
                    rest = None
                    return self._apply_steering(hidden_states=output,
                                                target_idx=target_idx,
                                                avoid_idx=avoid_idx,
                                                steering_strength=strength,
                                                rest=rest)

            return hook
        
        model_layers = get_transformer_layers(self.model)
        # Register hooks on target layers
        target_idx = [self.tone2idx[t] for t in target_labels]
        
        avoid_idx = None
        if avoid_labels:
            avoid_idx = [self.tone2idx[t] for t in avoid_labels]
            print(f"Generating Output for {target_labels} target labels and {avoid_labels} avoid labels")
        else:
            print(f"Generating Output for {target_labels} target labels")
            
        for layer_idx in target_layers:
            if layer_idx < self.model.config.num_hidden_layers:
                handle = model_layers[layer_idx].register_forward_hook(
                    make_hook(layer_idx, target_idx, avoid_idx)
                )
                hooks.append(handle)
        
        try:
            # Generate with steering
            inputs = self.tokenizer(input_prompts, return_tensors="pt").to(self.device)
            
            # with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
            
            generated_text = [self.tokenizer.decode(
                output[inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ) for output in outputs]
            
        finally:
            # Remove hooks
            for handle in hooks:
                handle.remove()
        
        return {
            'text': generated_text,
            'input_prompt': input_prompts,
            'steering_strength': steering_strength,
            'target_layers': target_layers,
            'layer_strengths': layer_strengths
        }

    def _load_task(self, task_name: str) -> Tuple[Any, List[str], List[str]]:
        """
        Load predefined task dataset

        Args:
            task_name: Name of task to load

        Returns:
            Tuple of (dataset, unique_labels, eval_prompts)
        """
        print(f"Loading Task: {task_name}")
        dataset, unique_labels, eval_prompts = load_task(task_name)
        return dataset, unique_labels, eval_prompts
        
        
        