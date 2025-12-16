from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle

from src.k_steering.steering.dataset import TaskDataset
from src.k_steering.steering.config import SteeringConfig, TrainerConfig


class ActivationSteering(ABC):
    """
    Base class for activation steering methods

    Provides common functionality for:
    - Model loading
    - Caching hidden states
    - Save/load operations
    - Basic generation interface

    Subclasses should implement:
    - build_classifier()
    - _apply_steering()
    """

    def __init__(
        self,
        model_name: str,
        steering_config: Optional[SteeringConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize activation steering base

        Args:
            model_name: HuggingFace model name or path
            steering_config: Configuration for steering behavior
            device: Device to load model on (auto-detected if None)
        """
        self.model_name = model_name
        self.steering_config = steering_config or SteeringConfig()
        self.trainer_config = trainer_config or TrainerConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_hf_model(model_name)

        # State variables (populated during fit)
        self.dataset = None
        self.unique_labels = None
        self.eval_prompts = None
        self.cache = None
        self._is_fitted = False

        # Steering components (subclass-specific)
        self.steering_vectors = None
        self.classifier = None

    def _load_hf_model(
        self,
        model_name: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load HuggingFace model and tokenizer

        Args:
            model_name: Model identifier

        Returns:
            Tuple of (model, tokenizer)
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            _attn_implementation="eager",
            output_hidden_states=True,
            device_map="auto" if self.device == "cuda" else None
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        return model, tokenizer

    def _load_task(self, task_name: str) -> Tuple[Any, List[str], List[str]]:
        """
        Load predefined task dataset

        Args:
            task_name: Name of task to load

        Returns:
            Tuple of (dataset, unique_labels, eval_prompts)
        """
        # TODO: Implement task loading from registry
        raise NotImplementedError(
            f"Task loading for '{task_name}' not implemented. "
            "Override this method or provide custom dataset."
        )

    def get_hidden_cache(
        self,
        prompts: List[str],
        batch_size: int = 8,
        return_attention_mask: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Get cached hidden states for all prompts

        Args:
            prompts: List of input prompts
            batch_size: Batch size for processing
            return_attention_mask: Whether to include attention masks

        Returns:
            Dictionary mapping layer indices to hidden states
        """
        cache = {i: [] for i in range(len(self.model.transformer.h))}
        attention_masks = [] if return_attention_mask else None

        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt",
                    padding=True, 
                    truncation=True
                ).to(self.device)

                if return_attention_mask:
                    attention_masks.append(inputs['attention_mask'])

                outputs = self.model(
                    **inputs, 
                    output_hidden_states=True,
                    return_dict=True
                )

                # Cache hidden states from each layer
                for layer_idx, hidden_state in enumerate(outputs.hidden_states
                                                         ):
                    cache[layer_idx].append(hidden_state.cpu())

        # Concatenate batches
        cache = {k: torch.cat(v, dim=0) for k, v in cache.items()}

        if return_attention_mask:
            cache['attention_mask'] = torch.cat(attention_masks, dim=0)

        self.cache = cache

        return cache

    def get_layer_cache(
        self,
        layer_idx: int,
        prompts: Optional[List[str]] = None,
        batch_size: int = 8,
        use_cached: bool = True,
    ) -> torch.Tensor:
        """
        Replicates get_hidden_cached_old using get_hidden_cache output.
        Extracts the hidden state of the last non-padding token for each
        prompt.

        Returns:
            Tensor of shape [num_prompts, hidden_dim]
        """
        # Ensure cache exists and contains attention_mask
        if not (use_cached and self.cache is not None and "attention_mask" in
                self.cache):
            if prompts is None:
                raise ValueError("Must provide prompts if cache is not \
                    available")

            self.cache = self.get_hidden_cache(
                prompts,
                batch_size=batch_size,
                return_attention_mask=True,
            )

        hidden = self.cache[layer_idx]
        attention_mask = self.cache["attention_mask"]

        # last non-padding token index
        lengths = attention_mask.sum(dim=1) - 1

        # collect last-token representations
        vectors = [
            hidden[i, idx] for i, idx in enumerate(lengths)
        ]

        return torch.stack(vectors, dim=0)

    @abstractmethod
    def build_steering_trainer(self, cache: Dict[str, torch.Tensor],
                               eval: bool = False):
        """
        Build steering classifier/vectors from cached activations

        Args:
            cache: Cached hidden states
            eval: Whether to build evaluation classifier

        Returns:
            Classifier/steering vectors (subclass-specific)
        """
        pass


