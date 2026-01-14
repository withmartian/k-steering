import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional, Dict, Any, List, Tuple, Union
from collections import defaultdict
from sklearn.model_selection import train_test_split
import logging
from src.k_steering.steering.base import ActivationSteering
from src.k_steering.steering.trainer import ActivationSteeringTrainer
from src.k_steering.steering.config import SteeringConfig, TrainerConfig
from src.k_steering.utils.data import load_task
from src.k_steering.utils.model import get_transformer_layers
from src.k_steering.utils.constants import DEBATE_DESCRIPTIONS, TONE_DESCRIPTIONS
_LOGGER = logging.getLogger(__name__)


class CAASteering(ActivationSteering):
    """
    CAA Steering: Activation steering using Contrastive Activation Addition
    
    Inherits common functionality from ActivationSteering and implements
    CAA Steering based steering logic.
    
    Example:
        >>> config = SteeringConfig(steering_strength=1.5, layers=[10, 15, 20])
        >>> k_steer = CAASteering("gpt2", config)
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
        self.cache = defaultdict(dict)
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
        
    def format_prompt(self,prompts, style: str=None):
        
        def get_formatted(example):
            instruction = self.style_instructions[style] if style else ""
            
            return f"{instruction}{example}"
        
        return list(map(get_formatted, prompts))
        
    def fit(
        self,
        task: Optional[str] = None,
        dataset: Optional[Any] = None,
        batch_size: int = 64,
        style_instructions: Dict = None
    ) -> "ActivationSteering":
        """
        Fit the steering model on a task or dataset

        Args:
            task: Name of predefined task
            dataset: Custom dataset
            eval_prompts: Optional evaluation prompts
            batch_size: Batch size for caching

        Returns:
            self for method chaining
        """
        # Load data
        if task is not None:
            _, self.unique_labels, self.prompts = self._load_task(task)
        elif dataset is not None:
            self.prompts = self._get_prompts_from_dataset(dataset)
            self.unique_labels = self._extract_labels(dataset)
        else:
            raise ValueError("Either 'task' or 'dataset' must be provided")
        
        if task == "tones":
            self.style_instructions = TONE_DESCRIPTIONS
        elif task == "debates":
            self.style_instructions = {key: DEBATE_DESCRIPTIONS[key] for key in DEBATE_DESCRIPTIONS.keys() if key in ['Empirical Grounding','Straw Man Reframing']}
        elif style_instructions:
            self.style_instructions = style_instructions
        else:
            raise ValueError("'style_instructions' must be provided")
        
            
                
            
        # Prepare prompts
        self.neutral_prompts = self.format_prompt(self.prompts)
        neutral_train_prompts, neutral_eval_prompts = train_test_split(self.neutral_prompts, test_size=0.2, random_state=42)
        
        # self.n_train_prompts = len(train_prompts)
        all_neutral_prompts = {
                            "train": neutral_train_prompts,
                            "eval": neutral_eval_prompts,
                        }
        
        # Cache hidden states
        self.logger.info(f"Caching hidden states for {all_neutral_prompts.keys()} neutral prompt style...")
        self.cache['neutral'] = self.get_hidden_cache(all_neutral_prompts, batch_size=batch_size)
        
        for style in self.style_instructions.keys():
        
            formatted_prompts = self.format_prompt(self.prompts, style)
        
            formatted_train_prompts, formatted_eval_prompts = train_test_split(formatted_prompts, test_size=0.2, random_state=42)
    
            all_formatted_prompts = {
                                "train": formatted_train_prompts,
                                "eval": formatted_eval_prompts,
                            }

        
        
            self.logger.info(f"Caching hidden states for {all_formatted_prompts.keys()} in {style} prompt style...")
            self.cache[style] = self.get_hidden_cache(all_formatted_prompts, batch_size=batch_size)

        # Build classifier/steering vectors
        self.logger.info("Building Steering Trainer...")
        self.build_steering_trainer(eval=False)
        self._is_fitted = True

        self.logger.info("Training complete!")
        return self
    
    def get_hidden_cache(
        self,
        prompts: Dict[str, List[str]],  # e.g. {"train": [...], "eval": [...]}
        batch_size: int = 64,
    ) -> Dict[str, Dict[Union[int, str], torch.Tensor]]:
        """
        Get cached hidden states for multiple named prompt splits.

        Args:
            prompts: Dict mapping split name -> list of prompts
            batch_size: Batch size for processing
            return_attention_mask: Whether to include attention masks

        Returns:
            {
            split_name: {
                layer_idx: Tensor [B, T, D],
                ...
                "attention_mask": Tensor [B, T]
            }
            }
        """
        
        num_layers = self.model.config.num_hidden_layers
        all_caches: Dict[str, Dict[Union[int, str], torch.Tensor]] = {}

        with torch.no_grad():
            for split_name, split_prompts in prompts.items():
                # cache = {i: [] for i in range(num_layers + 1)}

                for i in range(0, len(split_prompts), batch_size):
                    batch = split_prompts[i : i + batch_size]

                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)
                    
                    acts, handles = self._make_hooks(list(range(num_layers)))

                    _ = self.model(**inputs)
                    
                    for h in handles:
                        h.remove()
                        
                final_acts = {}
                for key, tensors in acts.items():
                    final_acts[key] = torch.cat(tensors, dim=0)
                    
                for layer in range(num_layers):
                    final_acts[('resid_mid', layer)] = (
                        final_acts[('resid_pre', layer)] +
                        final_acts[('attn_out', layer)]
    )

                all_caches[split_name] = final_acts

        return all_caches
    
    def get_layer_cache(
        self,
        style: str,
        split_name: str,
        layer_idx: int,
        pos: int = -1,
        act_type: str = "resid_pre",
        prompts: Optional[Dict[str, List[str]]] = None,
        batch_size: int = 64,
        use_cached: bool = True,
    ) -> torch.Tensor:
        """
        Extract the last non-padding token representation for a given
        layer and named split.

        Returns:
            Tensor of shape [num_prompts, hidden_dim]
        """
        # ---------- ensure cache for split exists ----------
        if (
            not use_cached 
            or self.cache is None 
            or style not in self.cache
        ):
            if prompts is None or split_name not in prompts:
                raise ValueError(
                    f"Must provide prompts for split '{split_name}' "
                    "if cache is not available"
                )

            # compute cache only for this split
            self.cache[style] = {}
            if style == "neutral":
                train_prompts, eval_prompts = train_test_split(self.neutral_prompts, test_size=0.2, random_state=42)
            else:
                formatted_prompts = self.format_prompt(self.prompts, style)
                train_prompts, eval_prompts = train_test_split(formatted_prompts, test_size=0.2, random_state=42)
            
            prompts = {
                            "train": train_prompts,
                            "eval": eval_prompts,
                        }
            self.cache[style].update(
                self.get_hidden_cache(
                    {split_name: prompts[split_name]},
                    batch_size=batch_size
                )
            )
        
        split_cache = self.cache[style][split_name]
        acts = (
                    split_cache[('resid_pre', layer_idx)][:, pos, :]
                    .detach()
                    .cpu()
                    .numpy()
                )

        return acts
        
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
        # if not eval:
        split_name = "train"
        layer_idx = self.steering_config.train_layer
        pos = self.steering_config.pos if self.steering_config.pos else -1
        # else:
        #     split_name = "eval"
        #     layer_idx = self.steering_config.eval_layer
        
        
        
        neutral_cache = self.get_layer_cache(style = "neutral",split_name=split_name, layer_idx = layer_idx, pos=pos)
        
        
        for style in self.style_instructions.keys():
            formatted_cache = self.get_layer_cache(style = style,split_name=split_name, layer_idx = layer_idx, pos=pos)
            self.steering_vectors[style] = torch.tensor(
                                            np.mean(formatted_cache, axis=0) - np.mean(neutral_cache, axis=0),
                                            dtype=torch.float32
                                        )
    
    def _apply_steering(
        self,
        hidden_states: torch.Tensor,
        target_labels: List[str],
        steering_strength: float = None,
        avoid_labels: List[str] | None = None,
        steps: int = 1,
        step_size_decay: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply CAA-steering to hidden states
        
        Args:
            hidden_states: Original hidden states [batch, seq_len, hidden_dim]
            layer_idx: Current layer index
            steering_strength: Strength multiplier
            
        Returns:
            Steered hidden states
        """
        
        if isinstance(hidden_states, np.ndarray):
            # acts_t = torch.as_tensor(hidden_states, dtype=torch.float32, device=self.device)
            acts_t = torch.as_tensor(hidden_states, device=self.device)
        else:
            # acts_t = hidden_states.to(self.device, dtype=torch.float32)
            acts_t = hidden_states.to(self.device)
            
        if avoid_labels is not None:
            combined_vector = sum(self.steering_vectors[c] for c in target_labels) - sum(self.steering_vectors[c] for c in avoid_labels)
        else:
            combined_vector = sum(self.steering_vectors[c] for c in target_labels)
        
        # Normalize the vector (L2)
        norm = combined_vector.norm(p=2)
        if norm == 0:
            # Avoid division by zero (no steering)
            return hidden_states
        normalized_vector = combined_vector / norm
        normalized_vector = normalized_vector.to(dtype = hidden_states.dtype)
        # Apply scaled direction
        return hidden_states + steering_strength * normalized_vector.to(hidden_states.device)
    
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
        
        def make_hook(layer_idx, target_labels, avoid_labels):
            def hook(module, input, output):
                # Get effective strength for this layer
                strength = layer_strengths.get(layer_idx, steering_strength)
                
                # Apply steering
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest = output[1:]
                    steered = self._apply_steering(hidden_states=hidden_states,
                                                   target_labels=target_labels,
                                                   avoid_labels=avoid_labels,
                                                   steering_strength=strength)
                    return (steered,) + output[1:]
                else:
                    rest = None
                    return self._apply_steering(hidden_states=output,
                                                target_labels=target_labels,
                                                avoid_labels=avoid_labels,
                                                steering_strength=strength)

            return hook
        
        model_layers = get_transformer_layers(self.model)
        # Register hooks on target layers
        
            
        for layer_idx in target_layers:
            if layer_idx < self.model.config.num_hidden_layers:
                handle = model_layers[layer_idx].register_forward_hook(
                    make_hook(layer_idx, target_labels, avoid_labels)
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
        self.logger.info(f"Loading Task: {task_name}")
        dataset, unique_labels, eval_prompts = load_task(task_name)
        return dataset, unique_labels, eval_prompts
    
    def _make_hooks(self, layers):
        acts = defaultdict(list)
        handles = []

        for layer in layers:
            block = self.model.model.layers[layer]

            def resid_pre_hook(module, inputs, layer=layer):
                acts[('resid_pre', layer)].append(inputs[0].detach().cpu())

            def attn_out_hook(module, inputs, output, layer=layer):
                acts[('attn_out', layer)].append(output[0])

            def mlp_out_hook(module, inputs, output, layer=layer):
                acts[('mlp_out', layer)].append(output[0])

            def resid_post_hook(module, inputs, output, layer=layer):
                acts[('resid_post', layer)].append(output[0])

            handles += [
                block.register_forward_pre_hook(resid_pre_hook),
                block.self_attn.register_forward_hook(attn_out_hook),
                block.mlp.register_forward_hook(mlp_out_hook),
                block.register_forward_hook(resid_post_hook),
            ]

        return acts, handles

    
    def _compute_resid_mid(acts, layers):
        for layer in layers:
            acts[('resid_mid', layer)] = (
                acts[('resid_pre', layer)] +
                acts[('attn_out', layer)]
            )
    
