import asyncio
import logging
import os
from typing import Any

import numpy as np
import torch

from k_steering.evals.judges.base_judge import BaseLLMJudge
from k_steering.steering.base import ActivationSteering
from k_steering.steering.config import SteeringConfig, TrainerConfig
from k_steering.steering.trainer import ActivationSteeringTrainer
from k_steering.utils.data import load_task
from k_steering.utils.model import get_transformer_layers
from k_steering.utils.sweep import calibrate_alpha_ood_only, is_ood

_LOGGER = logging.getLogger(__name__)

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
        steering_config: SteeringConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        device: str | None = None
    ):
        """
        Initialize K-Linear Steering class
        
        Args:
            model_name (str): HuggingFace model name or path
            steering_config (SteeringConfig): Configuration for steering behavior
            trainer_config (TrainerConfig): Configuration for training classifier
            device (str): Device to load model on (auto-detected if None)
        """
        super().__init__(model_name, steering_config,trainer_config, device)
        if not self.logger:
            self.logger = _LOGGER
            logging.basicConfig(level=logging.INFO)
            output_dir = self.steering_config.output_dir if self.steering_config.output_dir else "./outputs"
            os.makedirs(output_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(output_dir, "steering_log.log")
            )
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
        
    def build_steering_trainer(
        self, eval: bool = False
    ):
        """
        Build steering classifier/vectors from cached activations

        Args:
            cache (dict): Cached hidden states
            eval (bool): Whether to build evaluation classifier

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
        """
        Get One hot encoded array for unique classes

        Args:
            indices (np.ndarray): Input array containing labels
            num_classes (int): Number of classes

        Returns:
            np.ndarray: One hot encoded array
        """
        out = np.zeros((len(indices), num_classes), dtype=np.float32)
        out[np.arange(len(indices)), indices] = 1.0
        return out
    
    def _apply_steering(
        self,
        hidden_states: torch.Tensor,
        target_idx: list[int],
        steering_strength: float,
        rest: torch.Tensor | None = None,
        avoid_idx: list[int] | None = None,
        steps: int = 1,
        step_size_decay: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply K-steering to hidden states
        
        Args:
            hidden_states (torch.Tensor): Original hidden states [batch, seq_len, hidden_dim]
            layer_idx (int): Current layer index
            steering_strength (float): Strength multiplier
            
        Returns:
            torch.Tensor: Steered hidden states
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
        input_prompts: list[str],
        steering_strength: float,
        target_labels: list[str],
        layer_strengths: dict[int, float],
        generation_kwargs: dict[str, Any],
        avoid_labels: list[str] | None=None,
        target_layers: list[int] | None=None,
    ) -> dict[str, Any]:
        """
        Generate text with K-steering applied
        
        Args:
            input_prompt (List[str]): Input text
            steering_strength (float): Global steering strength
            target_labels (List[str]): Labels for steering behaviour towards
            avoid_labels (List[str]): Labels for steering behaviour away from 
            target_layers (List[str]): Layers to apply steering
            layer_strengths (Dict[int, float]): Layer-specific strengths
            generation_kwargs (Dict[str, Any]): Generation parameters
            
        Returns:
            Dict[str, Any]: Generation results dictionary
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
                    return steered
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
            self.logger.info(f"Generating Output for {target_labels} target labels and {avoid_labels} avoid labels")
        else:
            self.logger.info(f"Generating Output for {target_labels} target labels")
            
        for layer_idx in target_layers:
            if layer_idx < self.model.config.num_hidden_layers:
                handle = model_layers[layer_idx].register_forward_hook(
                    make_hook(layer_idx, target_idx, avoid_idx)
                )
                hooks.append(handle)
                
        # input_prompts = input_prompts[:2]
        
        try:
            # Generate with steering
            print(f"Tokenizing {len(input_prompts)} examples")
            inputs = self.tokenizer(input_prompts, return_tensors="pt", padding=True).to(self.device)
            
            # with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
            
            generated_text = [self.tokenizer.decode(
                output[inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ) for output in outputs]
            
            print("Generation Completed!!")
            
        finally:
            # Remove hooks
            print("Removing Hooks")
            for handle in hooks:
                handle.remove()
        
        return {
            'text': generated_text,
            'input_prompt': input_prompts,
            'steering_strength': steering_strength,
            'target_layers': target_layers,
            'layer_strengths': layer_strengths
        }

    def _load_task(self, task_name: str, max_samples:int = None) -> tuple[Any, list[str], list[str]]:
        """
        Load predefined task dataset

        Args:
            task_name (str): Name of task to load

        Returns:
            Tuple[Any, List[str], List[str]]: Tuple of (dataset, unique_labels, eval_prompts)
        """
        self.logger.info(f"Loading Task: {task_name}")
        dataset, unique_labels, eval_prompts = load_task(task_name, max_samples)
        return dataset, unique_labels, eval_prompts
    
    
    async def sweep_alpha(self, judge: BaseLLMJudge,
                          target_labels: list[str] | None = None,
                          avoid_labels: list[str] | None = None,
                          max_new_tokens: int = 100,
                          **generation_kwargs) -> dict[Any, list]:
        
        """
        Parameter Sweep Feature for calibrating optimal values of alpha / steering strength
        
        Args:
            judge (BaseLLMJudge): LLM Judge object for evaluating the coherence of the output
            target_labels (List): Labels for steering behaviour towards
            avoid_labels (List): Labels for steering behaviour away from 
            max_new_tokens (int) : Maximum new tokens to be generated by the LLM

        Returns:
            Dict[Any, List]: Layer Wise alpha/steering strength dictionary
        """
        self.gen_semaphore = asyncio.Semaphore(1)
        
        input_prompts = self._get_prompts_from_dataset(self.dataset)
        
        self.gen_kwargs = self._prepare_generation_kwargs(
            max_new_tokens=max_new_tokens, **generation_kwargs
        )
        
        layer_wise_alpha = {}
        
        for layer_idx in self.steering_config.steer_layers:
            print(f"Calibrating Alpha for Layer: {layer_idx} ")
            async def _ood_steer(alpha: float, layer_idx=layer_idx):
                async with self.gen_semaphore:
                    
                    gens = self._generate_with_steering(
                        input_prompts=input_prompts,
                        steering_strength=alpha,
                        target_labels=target_labels,
                        avoid_labels=avoid_labels,
                        target_layers=[layer_idx],
                        layer_strengths={layer_idx: alpha},
                        generation_kwargs=self.gen_kwargs,
                    )

                # Only the judge call is async-parallel
                return await is_ood(gens, judge=judge)
            
            optim_alpha = await calibrate_alpha_ood_only(_ood_steer)
            layer_wise_alpha[layer_idx] = optim_alpha
        
        return layer_wise_alpha
            
            
            
            
        
        
        
        
        
        
        